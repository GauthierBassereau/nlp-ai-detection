#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import inspect
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"matplotlib-{os.getuid()}"
    try:
        matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(matplotlib_config_dir)
    except OSError:
        pass

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
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
        ensure_text_list,
        load_hc3_dataset,
        pick_deterministic_text,
        safe_text,
        sanitize_identifier,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from hc3_utils import (
        DEFAULT_HUMAN_ANSWERS_COLUMN,
        DEFAULT_QUESTION_COLUMN,
        DEFAULT_SELECTED_HUMAN_COLUMN,
        DEFAULT_SOURCE_COLUMN,
        configure_logging,
        ensure_text_list,
        load_hc3_dataset,
        pick_deterministic_text,
        safe_text,
        sanitize_identifier,
    )


ID2LABEL = {0: "human", 1: "ai"}
LABEL2ID = {"human": 0, "ai": 1}
MODEL_CHOICES = {
    "bert-base": "google-bert/bert-base-uncased",
    "roberta-base": "FacebookAI/roberta-base",
    "modernbert-base": "answerdotai/ModernBERT-base",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one human-vs-AI classifier per generated answer column.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--dataset-dir", required=True)
    dataset_group.add_argument("--train-split", default="train")
    dataset_group.add_argument("--eval-split", default="eval")
    dataset_group.add_argument("--question-column", default=DEFAULT_QUESTION_COLUMN)
    dataset_group.add_argument("--human-answers-column", default=DEFAULT_HUMAN_ANSWERS_COLUMN)
    dataset_group.add_argument("--selected-human-column", default=DEFAULT_SELECTED_HUMAN_COLUMN)
    dataset_group.add_argument("--source-column", default=DEFAULT_SOURCE_COLUMN)
    dataset_group.add_argument(
        "--ai-answer-columns",
        nargs="+",
        default=None,
        help="Generated answer columns to train against. Defaults to all columns with --ai-column-prefix.",
    )
    dataset_group.add_argument("--ai-column-prefix", default="ai_")
    dataset_group.add_argument(
        "--text-mode",
        choices=("answer", "question_answer"),
        default="answer",
        help="Use answer only or prepend the question to each answer.",
    )
    dataset_group.add_argument(
        "--answer-window-words",
        type=int,
        default=None,
        help=(
            "If set, replace each answer with a deterministic random contiguous "
            "window of this many whitespace tokens before tokenization."
        ),
    )
    dataset_group.add_argument("--max-train-samples", type=int, default=None)
    dataset_group.add_argument("--max-eval-samples", type=int, default=None)

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model-choice", choices=tuple(MODEL_CHOICES.keys()), default="modernbert-base")
    model_group.add_argument("--model-name", default=None, help="Explicit checkpoint. Overrides --model-choice.")
    model_group.add_argument("--trust-remote-code", action="store_true")
    model_group.add_argument("--max-length", type=int, default=256)
    model_group.add_argument("--gradient-checkpointing", action="store_true")

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--output-dir", required=True)
    train_group.add_argument("--num-train-epochs", type=float, default=3.0)
    train_group.add_argument("--learning-rate", type=float, default=2e-5)
    train_group.add_argument("--weight-decay", type=float, default=0.01)
    train_group.add_argument("--warmup-ratio", type=float, default=0.1)
    train_group.add_argument("--per-device-train-batch-size", type=int, default=8)
    train_group.add_argument("--per-device-eval-batch-size", type=int, default=16)
    train_group.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_group.add_argument("--logging-steps", type=int, default=25)
    train_group.add_argument("--dataloader-num-workers", type=int, default=0)
    train_group.add_argument("--mixed-precision", choices=("auto", "no", "fp16", "bf16"), default="auto")
    train_group.add_argument("--report-to", choices=("none", "wandb"), default="wandb")
    train_group.add_argument("--wandb-project", default=None)
    train_group.add_argument("--wandb-entity", default=None)
    train_group.add_argument("--run-name", default=None)
    train_group.add_argument("--save-total-limit", type=int, default=2)
    train_group.add_argument("--overwrite-output-dir", action="store_true")
    train_group.add_argument("--skip-existing-runs", action="store_true")
    train_group.add_argument("--seed", type=int, default=42)

    logging_group = parser.add_argument_group("local eval logs")
    logging_group.add_argument("--max-bad-examples", type=int, default=200)
    logging_group.add_argument("--bad-example-text-chars", type=int, default=1200)
    logging_group.add_argument("--save-all-eval-predictions", action="store_true")
    logging_group.add_argument("--log-file", default=None)
    return parser


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def resolve_model_name(args: argparse.Namespace) -> str:
    return args.model_name or MODEL_CHOICES[args.model_choice]


def resolve_ai_columns(dataset_dict: DatasetDict, args: argparse.Namespace) -> list[str]:
    train_columns = set(dataset_dict[args.train_split].column_names)
    eval_columns = set(dataset_dict[args.eval_split].column_names)
    common_columns = train_columns.intersection(eval_columns)

    if args.ai_answer_columns:
        columns: list[str] = []
        seen: set[str] = set()
        for raw_value in args.ai_answer_columns:
            for column in raw_value.split(","):
                column = column.strip()
                if column and column not in seen:
                    seen.add(column)
                    columns.append(column)
        missing = sorted(column for column in columns if column not in common_columns)
        if missing:
            raise ValueError(f"AI answer column(s) missing from train/eval splits: {', '.join(missing)}")
        return columns

    columns = sorted(
        column
        for column in common_columns
        if column.startswith(args.ai_column_prefix) and not column.startswith("prompt_")
    )
    if not columns:
        raise ValueError(
            f"No AI answer columns found with prefix '{args.ai_column_prefix}'. "
            "Pass --ai-answer-columns explicitly."
        )
    return columns


def validate_dataset(dataset_dict: DatasetDict, args: argparse.Namespace) -> None:
    missing_splits = [split for split in (args.train_split, args.eval_split) if split not in dataset_dict]
    if missing_splits:
        raise ValueError(f"Dataset is missing required split(s): {', '.join(missing_splits)}")
    for split_name in (args.train_split, args.eval_split):
        columns = set(dataset_dict[split_name].column_names)
        missing = {args.question_column}.difference(columns)
        if args.selected_human_column not in columns and args.human_answers_column not in columns:
            missing.add(f"{args.selected_human_column} or {args.human_answers_column}")
        if missing:
            raise ValueError(f"Split '{split_name}' is missing required columns: {', '.join(sorted(missing))}")


def build_text(question: str, answer: str, text_mode: str) -> str:
    if text_mode == "answer" or not question:
        return answer
    return f"Question: {question}\n\nAnswer: {answer}"


def answer_window(
    answer: str,
    *,
    window_words: int | None,
    seed: int,
    split_name: str,
    row_index: int,
    source_key: str,
) -> tuple[str, int, int, int]:
    words = safe_text(answer).split()
    original_word_count = len(words)
    if window_words is None:
        return safe_text(answer), 0, original_word_count, original_word_count
    if window_words <= 0:
        raise ValueError("--answer-window-words must be > 0 when set.")
    if original_word_count <= window_words:
        return safe_text(answer), 0, original_word_count, original_word_count

    rng = random.Random(f"{seed}:{split_name}:{row_index}:{source_key}:answer_window:{window_words}")
    start = rng.randrange(0, original_word_count - window_words + 1)
    end = start + window_words
    return " ".join(words[start:end]), start, end, original_word_count


def selected_human_answer(
    example: dict[str, Any],
    *,
    args: argparse.Namespace,
    split_name: str,
    row_index: int,
) -> str:
    if args.selected_human_column in example:
        answer = safe_text(example.get(args.selected_human_column))
        if answer:
            return answer
    answer, _ = pick_deterministic_text(
        example.get(args.human_answers_column),
        seed=args.seed,
        split_name=split_name,
        row_index=row_index,
        salt=args.selected_human_column,
    )
    return answer


def selected_ai_answer(
    example: dict[str, Any],
    *,
    args: argparse.Namespace,
    split_name: str,
    row_index: int,
    ai_column: str,
) -> str:
    values = ensure_text_list(example.get(ai_column))
    if not values:
        return ""
    answer, _ = pick_deterministic_text(
        values,
        seed=args.seed,
        split_name=split_name,
        row_index=row_index,
        salt=ai_column,
    )
    return answer


def flatten_split(split_dataset: Dataset, *, split_name: str, ai_column: str, args: argparse.Namespace) -> Dataset:
    records: dict[str, list[Any]] = defaultdict(list)
    skipped = 0

    for row_index in range(len(split_dataset)):
        example = split_dataset[row_index]
        question = safe_text(example.get(args.question_column))
        human_answer = selected_human_answer(
            example,
            args=args,
            split_name=split_name,
            row_index=row_index,
        )
        ai_answer = selected_ai_answer(
            example,
            args=args,
            split_name=split_name,
            row_index=row_index,
            ai_column=ai_column,
        )
        if not question or not human_answer or not ai_answer:
            skipped += 1
            continue

        source = safe_text(example.get(args.source_column)) or "unknown"
        hc3_source_split = safe_text(example.get("hc3_source_split"))
        hc3_source_index = example.get("hc3_source_index", None)

        examples = [
            ("human", 0, args.selected_human_column, human_answer),
            ("ai", 1, ai_column, ai_answer),
        ]
        for author_type, label, answer_source_column, answer in examples:
            answer_for_model, window_start, window_end, original_word_count = answer_window(
                answer,
                window_words=args.answer_window_words,
                seed=args.seed,
                split_name=split_name,
                row_index=row_index,
                source_key=answer_source_column,
            )
            records["example_id"].append(f"{split_name}-{row_index}-{answer_source_column}-{author_type}")
            records["row_index"].append(row_index)
            records["hc3_source_split"].append(hc3_source_split)
            records["hc3_source_index"].append(hc3_source_index)
            records["question"].append(question)
            records["answer"].append(answer_for_model)
            records["full_answer"].append(answer)
            records["answer_window_start"].append(window_start)
            records["answer_window_end"].append(window_end)
            records["answer_original_word_count"].append(original_word_count)
            records["text"].append(build_text(question, answer_for_model, args.text_mode))
            records["source"].append(source)
            records["author_type"].append(author_type)
            records["answer_source_column"].append(answer_source_column)
            records["label"].append(label)

    if skipped:
        logging.info(
            "Skipped %s %s rows without usable human/AI answers for %s.",
            f"{skipped:,}",
            split_name,
            ai_column,
        )
    if not records:
        raise ValueError(f"No usable classification examples for split={split_name} column={ai_column}")
    return Dataset.from_dict(dict(records))


def build_classification_dataset(dataset_dict: DatasetDict, ai_column: str, args: argparse.Namespace) -> DatasetDict:
    train_dataset = flatten_split(dataset_dict[args.train_split], split_name=args.train_split, ai_column=ai_column, args=args)
    eval_dataset = flatten_split(dataset_dict[args.eval_split], split_name=args.eval_split, ai_column=ai_column, args=args)
    train_dataset = subsample_dataset(train_dataset, args.max_train_samples, args.seed)
    eval_dataset = subsample_dataset(eval_dataset, args.max_eval_samples, args.seed)
    return DatasetDict({"train": train_dataset, "eval": eval_dataset})


def subsample_dataset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None:
        return dataset.shuffle(seed=seed)
    if max_samples <= 0:
        raise ValueError("Max sample arguments must be > 0.")
    if max_samples >= len(dataset):
        return dataset.shuffle(seed=seed)
    return dataset.shuffle(seed=seed).select(range(max_samples))


def tokenize_dataset(dataset_dict: DatasetDict, tokenizer: Any, max_length: int) -> DatasetDict:
    remove_columns = [column for column in dataset_dict["train"].column_names if column != "label"]

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return dataset_dict.map(
        tokenize_batch,
        batched=True,
        remove_columns=remove_columns,
        desc="Tokenizing classification dataset",
    )


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def detect_backend() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_precision(mode: str, backend: str) -> dict[str, bool]:
    settings = {"fp16": False, "bf16": False}
    if backend != "cuda":
        return settings
    if mode == "fp16":
        settings["fp16"] = True
    elif mode == "bf16":
        settings["bf16"] = True
    elif mode == "auto":
        settings["bf16" if torch.cuda.is_bf16_supported() else "fp16"] = True
    return settings


def build_training_arguments(
    args: argparse.Namespace,
    *,
    run_output_dir: Path,
    run_name: str,
    backend: str,
) -> TrainingArguments:
    precision = resolve_precision(args.mixed_precision, backend)
    signature = inspect.signature(TrainingArguments.__init__)
    kwargs: dict[str, Any] = {
        "output_dir": str(run_output_dir),
        "overwrite_output_dir": args.overwrite_output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "logging_strategy": "steps",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": backend == "cuda",
        "fp16": precision["fp16"],
        "bf16": precision["bf16"],
        "seed": args.seed,
        "report_to": [] if args.report_to == "none" else [args.report_to],
        "run_name": run_name,
        "save_total_limit": args.save_total_limit,
        "logging_dir": str(run_output_dir / "trainer_logs"),
    }
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    if "use_mps_device" in signature.parameters and backend == "mps":
        kwargs["use_mps_device"] = True
    if "ddp_find_unused_parameters" in signature.parameters:
        kwargs["ddp_find_unused_parameters"] = False
    return TrainingArguments(**kwargs)


def build_trainer(
    *,
    model: Any,
    training_args: TrainingArguments,
    tokenized_dataset: DatasetDict,
    tokenizer: Any,
) -> Trainer:
    trainer_signature = inspect.signature(Trainer.__init__)
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_dataset["train"],
        "eval_dataset": tokenized_dataset["eval"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    if "processing_class" in trainer_signature.parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def prepare_run_output_dir(run_output_dir: Path, args: argparse.Namespace) -> bool:
    if run_output_dir.exists() and any(run_output_dir.iterdir()):
        if args.skip_existing_runs:
            logging.info("Skipping existing run directory: %s", run_output_dir)
            return False
        if args.overwrite_output_dir:
            shutil.rmtree(run_output_dir)
        else:
            raise FileExistsError(
                f"{run_output_dir} already exists. Pass --overwrite-output-dir or --skip-existing-runs."
            )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    return True


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        file_handle.write(",".join(columns) + "\n")
        for row in rows:
            values = [csv_escape(row.get(column, "")) for column in columns]
            file_handle.write(",".join(values) + "\n")


def csv_escape(value: Any) -> str:
    text = safe_text(value)
    if any(char in text for char in [",", "\"", "\n", "\r"]):
        return "\"" + text.replace("\"", "\"\"") + "\""
    return text


def source_metric_rows(eval_dataset: Dataset, labels: np.ndarray, predictions: np.ndarray) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"labels": [], "predictions": []})
    for index, example in enumerate(eval_dataset):
        source = safe_text(example.get("source")) or "unknown"
        grouped[source]["labels"].append(int(labels[index]))
        grouped[source]["predictions"].append(int(predictions[index]))

    rows: list[dict[str, Any]] = []
    for source, values in sorted(grouped.items()):
        group_labels = np.array(values["labels"])
        group_predictions = np.array(values["predictions"])
        precision, recall, f1, support = precision_recall_fscore_support(
            group_labels,
            group_predictions,
            labels=[0, 1],
            average=None,
            zero_division=0,
        )
        row: dict[str, Any] = {
            "source": source,
            "n": int(len(group_labels)),
            "accuracy": float(accuracy_score(group_labels, group_predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(group_labels, group_predictions)),
            "macro_f1": float(np.mean(f1)),
            "predicted_human_n": int((group_predictions == 0).sum()),
            "predicted_ai_n": int((group_predictions == 1).sum()),
        }
        for label_id, label_name in ID2LABEL.items():
            mask = group_labels == label_id
            row[f"{label_name}_n"] = int(mask.sum())
            row[f"{label_name}_accuracy"] = (
                float(accuracy_score(group_labels[mask], group_predictions[mask])) if mask.any() else None
            )
            row[f"{label_name}_precision"] = float(precision[label_id])
            row[f"{label_name}_recall"] = float(recall[label_id])
            row[f"{label_name}_f1"] = float(f1[label_id])
            row[f"{label_name}_support"] = int(support[label_id])
        rows.append(row)
    return rows


def plot_source_metrics(source_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not source_rows:
        return
    sorted_rows = sorted(source_rows, key=lambda row: row["source"])
    sources = [row["source"] for row in sorted_rows]
    metrics = [
        ("accuracy", "Accuracy"),
        ("ai_precision", "AI precision"),
        ("ai_recall", "AI recall"),
        ("ai_f1", "AI F1"),
    ]
    x_positions = np.arange(len(sources))
    bar_width = min(0.18, 0.8 / len(metrics))
    figure_width = max(10, len(sources) * 1.15)

    plt.figure(figsize=(figure_width, 6.5))
    for metric_index, (metric_key, metric_label) in enumerate(metrics):
        values = [float(row.get(metric_key) or 0.0) for row in sorted_rows]
        offsets = x_positions + (metric_index - (len(metrics) - 1) / 2) * bar_width
        plt.bar(offsets, values, width=bar_width, label=metric_label)

    plt.xticks(x_positions, sources, rotation=35, ha="right")
    plt.ylim(0, 1.02)
    plt.ylabel("Score")
    plt.title("Evaluation Metrics By Source")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def bad_classification_rows(
    eval_dataset: Dataset,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    max_rows: int,
    answer_chars: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wrong_indices = np.where(labels != predictions)[0]
    for index in wrong_indices[:max_rows]:
        example = eval_dataset[int(index)]
        row = {
            "example_id": safe_text(example.get("example_id")),
            "row_index": example.get("row_index"),
            "source": safe_text(example.get("source")) or "unknown",
            "answer_source_column": safe_text(example.get("answer_source_column")),
            "label": ID2LABEL[int(labels[index])],
            "prediction": ID2LABEL[int(predictions[index])],
            "confidence": float(np.max(probabilities[index])),
            "prob_human": float(probabilities[index][0]),
            "prob_ai": float(probabilities[index][1]),
            "question": safe_text(example.get("question")),
            "answer": safe_text(example.get("answer"))[:answer_chars],
            "full_answer": safe_text(example.get("full_answer"))[:answer_chars],
            "answer_window_start": example.get("answer_window_start"),
            "answer_window_end": example.get("answer_window_end"),
            "answer_original_word_count": example.get("answer_original_word_count"),
        }
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        for row in rows:
            file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_eval_artifacts(
    *,
    run_output_dir: Path,
    ai_column: str,
    eval_dataset: Dataset,
    prediction_output: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    logits = prediction_output.predictions
    labels = prediction_output.label_ids.astype(int)
    predictions = np.argmax(logits, axis=-1).astype(int)
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    metrics = {
        **{key: float(value) for key, value in prediction_output.metrics.items()},
        "eval_column": ai_column,
        "eval_examples": int(len(labels)),
    }
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=[ID2LABEL[0], ID2LABEL[1]],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()
    source_rows = source_metric_rows(eval_dataset, labels, predictions)
    bad_rows = bad_classification_rows(
        eval_dataset,
        labels,
        predictions,
        probabilities,
        max_rows=args.max_bad_examples,
        answer_chars=args.bad_example_text_chars,
    )

    log_dir = run_output_dir / "eval_logs"
    write_json(log_dir / "metrics.json", metrics)
    write_json(log_dir / "classification_report.json", report)
    write_json(log_dir / "confusion_matrix.json", {"labels": [ID2LABEL[0], ID2LABEL[1]], "matrix": matrix})
    write_json(log_dir / "source_metrics.json", source_rows)
    write_json(log_dir / "source_accuracy.json", source_rows)
    write_csv(
        log_dir / "source_metrics.csv",
        source_rows,
        [
            "source",
            "n",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "human_n",
            "human_precision",
            "human_recall",
            "human_f1",
            "ai_n",
            "ai_precision",
            "ai_recall",
            "ai_f1",
            "predicted_human_n",
            "predicted_ai_n",
        ],
    )
    write_csv(
        log_dir / "source_accuracy.csv",
        source_rows,
        ["source", "n", "accuracy", "human_n", "human_accuracy", "ai_n", "ai_accuracy"],
    )
    plot_source_metrics(source_rows, log_dir / "source_metrics.png")
    write_jsonl(log_dir / "bad_classifications.jsonl", bad_rows)

    if args.save_all_eval_predictions:
        prediction_rows = []
        for index, example in enumerate(eval_dataset):
            prediction_rows.append(
                {
                    "example_id": safe_text(example.get("example_id")),
                    "label": ID2LABEL[int(labels[index])],
                    "prediction": ID2LABEL[int(predictions[index])],
                    "prob_human": float(probabilities[index][0]),
                    "prob_ai": float(probabilities[index][1]),
                    "source": safe_text(example.get("source")) or "unknown",
                    "answer_source_column": safe_text(example.get("answer_source_column")),
                    "answer_window_start": example.get("answer_window_start"),
                    "answer_window_end": example.get("answer_window_end"),
                    "answer_original_word_count": example.get("answer_original_word_count"),
                }
            )
        write_jsonl(log_dir / "eval_predictions.jsonl", prediction_rows)

    maybe_log_wandb_tables(args, source_rows, bad_rows)
    return metrics


def maybe_log_wandb_tables(
    args: argparse.Namespace,
    source_rows: list[dict[str, Any]],
    bad_rows: list[dict[str, Any]],
) -> None:
    if args.report_to != "wandb" or not is_main_process():
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    source_columns = [
        "source",
        "n",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "human_precision",
        "human_recall",
        "human_f1",
        "ai_precision",
        "ai_recall",
        "ai_f1",
    ]
    bad_columns = [
        "example_id",
        "source",
        "answer_source_column",
        "label",
        "prediction",
        "confidence",
        "prob_human",
        "prob_ai",
        "question",
        "answer",
    ]
    wandb.log(
        {
            "eval/source_accuracy": wandb.Table(
                columns=source_columns,
                data=[[row.get(column) for column in source_columns] for row in source_rows],
            ),
            "eval/bad_classifications": wandb.Table(
                columns=bad_columns,
                data=[[row.get(column) for column in bad_columns] for row in bad_rows],
            ),
        }
    )


def finish_wandb_if_needed(args: argparse.Namespace) -> None:
    if args.report_to != "wandb" or not is_main_process():
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.finish()


def set_wandb_env(args: argparse.Namespace) -> None:
    if args.report_to != "wandb":
        return
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity


def train_one_column(
    *,
    dataset_dict: DatasetDict,
    ai_column: str,
    model_name: str,
    tokenizer: Any,
    args: argparse.Namespace,
    backend: str,
) -> dict[str, Any] | None:
    run_id = sanitize_identifier(ai_column, fallback="ai_column")
    run_output_dir = Path(args.output_dir) / run_id
    run_exists = run_output_dir.exists() and any(run_output_dir.iterdir())
    if run_exists and args.skip_existing_runs:
        logging.info("Skipping existing run directory: %s", run_output_dir)
        return None
    if run_exists and not args.overwrite_output_dir:
        raise FileExistsError(
            f"{run_output_dir} already exists. Pass --overwrite-output-dir or --skip-existing-runs."
        )
    if is_main_process() and not prepare_run_output_dir(run_output_dir, args):
        return None

    classification_dataset = build_classification_dataset(dataset_dict, ai_column, args)
    tokenized_dataset = tokenize_dataset(classification_dataset, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=args.trust_remote_code,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    run_name = f"{args.run_name}-{run_id}" if args.run_name else f"{sanitize_identifier(model_name, fallback='model')}-{run_id}"
    training_args = build_training_arguments(
        args,
        run_output_dir=run_output_dir,
        run_name=run_name,
        backend=backend,
    )
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    logging.info("Training column %s", ai_column)
    logging.info(
        "Examples for %s: train=%s eval=%s",
        ai_column,
        f"{len(classification_dataset['train']):,}",
        f"{len(classification_dataset['eval']):,}",
    )
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(str(run_output_dir))
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    prediction_output = trainer.predict(tokenized_dataset["eval"], metric_key_prefix="eval")
    trainer.log_metrics("eval", prediction_output.metrics)
    trainer.save_metrics("eval", prediction_output.metrics)

    local_metrics = None
    if is_main_process():
        local_metrics = write_eval_artifacts(
            run_output_dir=run_output_dir,
            ai_column=ai_column,
            eval_dataset=classification_dataset["eval"],
            prediction_output=prediction_output,
            args=args,
        )
        logging.info("Saved local eval logs to %s", run_output_dir / "eval_logs")

    finish_wandb_if_needed(args)
    del trainer, model, tokenized_dataset, classification_dataset
    cleanup_memory()
    return local_metrics


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_file, rank=int(os.environ.get("RANK", "0")))
    set_seed(args.seed)
    set_wandb_env(args)

    if args.max_length <= 0:
        raise ValueError("--max-length must be > 0.")
    dataset_dict = load_hc3_dataset(dataset_dir=args.dataset_dir)
    validate_dataset(dataset_dict, args)
    ai_columns = resolve_ai_columns(dataset_dict, args)
    model_name = resolve_model_name(args)
    backend = detect_backend()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)

    if is_main_process():
        logging.info("Model checkpoint: %s", model_name)
        logging.info("Detected backend: %s", backend)
        logging.info("AI columns: %s", ", ".join(ai_columns))

    all_metrics: dict[str, Any] = {}
    for ai_column in ai_columns:
        metrics = train_one_column(
            dataset_dict=dataset_dict,
            ai_column=ai_column,
            model_name=model_name,
            tokenizer=tokenizer,
            args=args,
            backend=backend,
        )
        if metrics is not None:
            all_metrics[ai_column] = metrics

    if is_main_process():
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "all_eval_metrics.json", all_metrics)
        logging.info("Saved aggregate metrics to %s", output_dir / "all_eval_metrics.json")


if __name__ == "__main__":
    main()
