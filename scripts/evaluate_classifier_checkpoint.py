#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from scripts.hc3_utils import (
        DEFAULT_HUMAN_ANSWERS_COLUMN,
        DEFAULT_QUESTION_COLUMN,
        DEFAULT_SELECTED_HUMAN_COLUMN,
        DEFAULT_SOURCE_COLUMN,
        configure_logging,
        load_hc3_dataset,
        sanitize_identifier,
    )
    from scripts.train_classifier import (
        LABEL2ID,
        ID2LABEL,
        compute_metrics,
        flatten_split,
        is_main_process,
        resolve_ai_columns,
        write_eval_artifacts,
        write_json,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from hc3_utils import (
        DEFAULT_HUMAN_ANSWERS_COLUMN,
        DEFAULT_QUESTION_COLUMN,
        DEFAULT_SELECTED_HUMAN_COLUMN,
        DEFAULT_SOURCE_COLUMN,
        configure_logging,
        load_hc3_dataset,
        sanitize_identifier,
    )
    from train_classifier import (
        LABEL2ID,
        ID2LABEL,
        compute_metrics,
        flatten_split,
        is_main_process,
        resolve_ai_columns,
        write_eval_artifacts,
        write_json,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained classifier checkpoint on one or more AI answer columns.",
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Saved classifier directory, e.g. outputs/.../ai_qwen25_3b_actual_human_reference.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help="Tokenizer path/name. Defaults to --checkpoint-dir.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-split", default="eval")
    parser.add_argument("--question-column", default=DEFAULT_QUESTION_COLUMN)
    parser.add_argument("--human-answers-column", default=DEFAULT_HUMAN_ANSWERS_COLUMN)
    parser.add_argument("--selected-human-column", default=DEFAULT_SELECTED_HUMAN_COLUMN)
    parser.add_argument("--source-column", default=DEFAULT_SOURCE_COLUMN)
    parser.add_argument(
        "--ai-answer-columns",
        nargs="+",
        required=True,
        help="AI answer column(s) to evaluate the checkpoint on.",
    )
    parser.add_argument("--ai-column-prefix", default="ai_")
    parser.add_argument("--text-mode", choices=("answer", "question_answer"), default="answer")
    parser.add_argument(
        "--answer-window-words",
        type=int,
        default=None,
        help="Must match training if the checkpoint was trained with answer windows.",
    )
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--max-bad-examples", type=int, default=200)
    parser.add_argument("--bad-example-text-chars", type=int, default=1200)
    parser.add_argument("--save-all-eval-predictions", action="store_true")
    parser.add_argument("--log-file", default=None)
    return parser


def make_classifier_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        train_split=args.eval_split,
        eval_split=args.eval_split,
        question_column=args.question_column,
        human_answers_column=args.human_answers_column,
        selected_human_column=args.selected_human_column,
        source_column=args.source_column,
        ai_answer_columns=args.ai_answer_columns,
        ai_column_prefix=args.ai_column_prefix,
        text_mode=args.text_mode,
        answer_window_words=args.answer_window_words,
        max_train_samples=None,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
    )


def tokenize_eval_dataset(eval_dataset: Dataset, tokenizer: Any, max_length: int) -> Dataset:
    remove_columns = [column for column in eval_dataset.column_names if column != "label"]

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return eval_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=remove_columns,
        desc="Tokenizing eval dataset",
    )


def build_eval_training_args(args: argparse.Namespace, run_output_dir: Path) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    kwargs: dict[str, Any] = {
        "output_dir": str(run_output_dir / "trainer_tmp"),
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "report_to": [],
        "seed": args.seed,
    }
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"
    if "ddp_find_unused_parameters" in signature.parameters:
        kwargs["ddp_find_unused_parameters"] = False
    return TrainingArguments(**kwargs)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Pass --overwrite-output-dir or choose a new output dir."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def evaluate_column(
    *,
    dataset_dict: DatasetDict,
    ai_column: str,
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    classifier_args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any] | None:
    eval_dataset = flatten_split(
        dataset_dict[args.eval_split],
        split_name=args.eval_split,
        ai_column=ai_column,
        args=classifier_args,
    )
    if args.max_eval_samples is not None:
        if args.max_eval_samples <= 0:
            raise ValueError("--max-eval-samples must be > 0.")
        eval_dataset = eval_dataset.shuffle(seed=args.seed).select(
            range(min(args.max_eval_samples, len(eval_dataset)))
        )

    tokenized_eval = tokenize_eval_dataset(eval_dataset, tokenizer, args.max_length)
    run_output_dir = output_dir / sanitize_identifier(ai_column, fallback="ai_column")
    training_args = build_eval_training_args(args, run_output_dir)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "eval_dataset": tokenized_eval,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    logging.info("Evaluating checkpoint %s on column %s", args.checkpoint_dir, ai_column)
    prediction_output = trainer.predict(tokenized_eval, metric_key_prefix="eval")
    metrics = None
    if is_main_process():
        run_output_dir.mkdir(parents=True, exist_ok=True)
        metrics = write_eval_artifacts(
            run_output_dir=run_output_dir,
            ai_column=ai_column,
            eval_dataset=eval_dataset,
            prediction_output=prediction_output,
            args=argparse.Namespace(
                report_to="none",
                max_bad_examples=args.max_bad_examples,
                bad_example_text_chars=args.bad_example_text_chars,
                save_all_eval_predictions=args.save_all_eval_predictions,
            ),
        )
        write_json(
            run_output_dir / "eval_config.json",
            {
                "dataset_dir": args.dataset_dir,
                "checkpoint_dir": args.checkpoint_dir,
                "tokenizer_name": args.tokenizer_name or args.checkpoint_dir,
                "eval_split": args.eval_split,
                "ai_column": ai_column,
                "text_mode": args.text_mode,
                "answer_window_words": args.answer_window_words,
                "max_length": args.max_length,
                "max_eval_samples": args.max_eval_samples,
                "seed": args.seed,
            },
        )
        logging.info("Saved eval logs to %s", run_output_dir / "eval_logs")
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_file, rank=int(os.environ.get("RANK", "0")))
    set_seed(args.seed)

    if args.max_length <= 0:
        raise ValueError("--max-length must be > 0.")
    output_dir = Path(args.output_dir)
    if is_main_process():
        prepare_output_dir(output_dir, args.overwrite_output_dir)

    dataset_dict = load_hc3_dataset(dataset_dir=args.dataset_dir)
    if args.eval_split not in dataset_dict:
        raise ValueError(f"Split '{args.eval_split}' not found. Available: {', '.join(dataset_dict.keys())}")
    classifier_args = make_classifier_args(args)
    ai_columns = resolve_ai_columns(dataset_dict, classifier_args)

    tokenizer_name = args.tokenizer_name or args.checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=args.trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint_dir,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=args.trust_remote_code,
    )

    if is_main_process():
        logging.info("Checkpoint: %s", args.checkpoint_dir)
        logging.info("Tokenizer: %s", tokenizer_name)
        logging.info("Eval columns: %s", ", ".join(ai_columns))

    all_metrics: dict[str, Any] = {}
    for ai_column in ai_columns:
        metrics = evaluate_column(
            dataset_dict=dataset_dict,
            ai_column=ai_column,
            model=model,
            tokenizer=tokenizer,
            args=args,
            classifier_args=classifier_args,
            output_dir=output_dir,
        )
        if metrics is not None:
            all_metrics[ai_column] = metrics

    if is_main_process():
        write_json(output_dir / "all_eval_metrics.json", all_metrics)
        logging.info("Saved aggregate eval metrics to %s", output_dir / "all_eval_metrics.json")


if __name__ == "__main__":
    main()
