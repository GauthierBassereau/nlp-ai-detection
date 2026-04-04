#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    import evaluate
except ImportError:  # pragma: no cover
    evaluate = None


ID2LABEL = {0: "human", 1: "ai"}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}
DEFAULT_DATASET_REVISION = "refs/convert/parquet"
MODEL_CHOICES = {
    "bert-base": "google-bert/bert-base-uncased",
    "roberta-base": "FacebookAI/roberta-base",
    "modernbert-base": "answerdotai/ModernBERT-base",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a transformer classifier for human-vs-AI text classification.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument(
        "--dataset-dir",
        default=None,
        help="Path to a dataset saved with datasets.save_to_disk().",
    )
    dataset_group.add_argument(
        "--dataset-name",
        default="Hello-SimpleAI/HC3",
        help="HF dataset id used when --dataset-dir is not provided.",
    )
    dataset_group.add_argument(
        "--dataset-config",
        default="all",
        help="Subset/data directory used when --dataset-dir is not provided.",
    )
    dataset_group.add_argument(
        "--cache-dir",
        default=None,
        help="Optional HF cache directory.",
    )
    dataset_group.add_argument(
        "--question-column",
        default="question",
        help="Column name for the question in raw HC3-style data.",
    )
    dataset_group.add_argument(
        "--human-answers-column",
        default="human_answers",
        help="Column name for human-written answers in raw HC3-style data.",
    )
    dataset_group.add_argument(
        "--ai-answers-column",
        default="chatgpt_answers",
        help="Column name for AI-written answers in raw HC3-style data.",
    )
    dataset_group.add_argument(
        "--source-column",
        default="source",
        help="Optional source/domain column name in raw HC3-style data.",
    )
    dataset_group.add_argument(
        "--text-mode",
        choices=("answer", "question_answer"),
        default="answer",
        help="Whether to train on answer text only or question+answer.",
    )
    dataset_group.add_argument(
        "--validation-size",
        type=float,
        default=0.1,
        help="Validation proportion after flattening the dataset.",
    )
    dataset_group.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Test proportion after flattening the dataset.",
    )
    dataset_group.add_argument(
        "--train-subsample-ratio",
        type=float,
        default=1.0,
        help="Use less than 1.0 to train on a fraction of the train split.",
    )
    dataset_group.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional hard cap for the number of training examples.",
    )
    dataset_group.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help="Optional hard cap for the number of validation examples.",
    )
    dataset_group.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional hard cap for the number of test examples.",
    )
    dataset_group.add_argument(
        "--save-splits-dir",
        default=None,
        help="Optional path for saving the flattened train/validation/test DatasetDict.",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--model-choice",
        choices=tuple(MODEL_CHOICES.keys()),
        default="modernbert-base",
        help="Named baseline model to fine-tune.",
    )
    model_group.add_argument(
        "--model-name",
        default=None,
        help="Optional explicit HF checkpoint. Overrides --model-choice.",
    )
    model_group.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer truncation length.",
    )
    model_group.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage.",
    )

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--output-dir", required=True, help="Where to save checkpoints and logs.")
    train_group.add_argument("--num-train-epochs", type=float, default=3.0)
    train_group.add_argument("--learning-rate", type=float, default=2e-5)
    train_group.add_argument("--weight-decay", type=float, default=0.01)
    train_group.add_argument("--warmup-ratio", type=float, default=0.1)
    train_group.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Explicit warmup steps. Overrides --warmup-ratio when set.",
    )
    train_group.add_argument("--per-device-train-batch-size", type=int, default=8)
    train_group.add_argument("--per-device-eval-batch-size", type=int, default=16)
    train_group.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_group.add_argument("--logging-steps", type=int, default=25)
    train_group.add_argument("--dataloader-num-workers", type=int, default=0)
    train_group.add_argument(
        "--report-to",
        choices=("none", "wandb", "tensorboard"),
        default="wandb",
        help="Experiment tracker to use.",
    )
    train_group.add_argument(
        "--run-name",
        default=None,
        help="Optional experiment run name for the tracker UI.",
    )
    train_group.add_argument(
        "--torch-empty-cache-steps",
        type=int,
        default=None,
        help="Call torch empty_cache periodically during training. Useful on MPS.",
    )
    train_group.add_argument(
        "--mixed-precision",
        choices=("auto", "no", "fp16", "bf16"),
        default="auto",
        help="CUDA mixed precision mode. Auto enables bf16/fp16 only on CUDA.",
    )
    train_group.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Optional checkpoint path for resuming training.",
    )
    train_group.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Allow overwriting an existing output directory.",
    )
    train_group.add_argument("--seed", type=int, default=42)

    return parser


def detect_backend() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_precision_mode(mode: str, backend: str) -> dict[str, bool]:
    settings = {"fp16": False, "bf16": False}
    if backend != "cuda":
        return settings

    if mode == "fp16":
        settings["fp16"] = True
        return settings
    if mode == "bf16":
        settings["bf16"] = True
        return settings
    if mode == "auto":
        if torch.cuda.is_bf16_supported():
            settings["bf16"] = True
        else:
            settings["fp16"] = True
    return settings


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def ensure_list_of_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            text = safe_text(item)
            if text:
                items.append(text)
        return items
    text = safe_text(value)
    return [text] if text else []


def build_text(question: str, answer: str, text_mode: str) -> str:
    if text_mode == "answer" or not question:
        return answer
    return f"Question: {question}\n\nAnswer: {answer}"


def normalize_dataset_object(dataset_obj: Dataset | DatasetDict) -> DatasetDict:
    if isinstance(dataset_obj, DatasetDict):
        return dataset_obj
    return DatasetDict({"train": dataset_obj})


def load_source_dataset(args: argparse.Namespace) -> DatasetDict:
    if args.dataset_dir:
        dataset_path = Path(args.dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        dataset_obj = load_from_disk(str(dataset_path))
    else:
        load_kwargs = {
            "path": args.dataset_name,
            "cache_dir": args.cache_dir,
            "revision": DEFAULT_DATASET_REVISION,
        }
        if args.dataset_config:
            load_kwargs["data_dir"] = args.dataset_config
        dataset_obj = load_dataset(**load_kwargs)
    return normalize_dataset_object(dataset_obj)


def flatten_hc3_split(split_dataset: Dataset, args: argparse.Namespace, split_name: str) -> Dataset:
    if {"text", "label"}.issubset(split_dataset.column_names):
        return normalize_flat_split(split_dataset)

    required_columns = {
        args.question_column,
        args.human_answers_column,
        args.ai_answers_column,
    }
    missing_columns = required_columns.difference(split_dataset.column_names)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Split '{split_name}' is missing required columns: {missing_list}"
        )

    def explode_batch(batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
        size = len(indices)
        sources = batch.get(args.source_column, [None] * size)

        flattened = {
            "example_id": [],
            "question": [],
            "answer": [],
            "text": [],
            "source": [],
            "author_type": [],
            "label": [],
        }

        for idx_in_batch, row_index in enumerate(indices):
            question = safe_text(batch[args.question_column][idx_in_batch])
            source = safe_text(sources[idx_in_batch]) if sources[idx_in_batch] is not None else None

            human_answers = ensure_list_of_text(batch[args.human_answers_column][idx_in_batch])
            ai_answers = ensure_list_of_text(batch[args.ai_answers_column][idx_in_batch])

            for answer_idx, answer in enumerate(human_answers):
                flattened["example_id"].append(f"{split_name}-{row_index}-human-{answer_idx}")
                flattened["question"].append(question)
                flattened["answer"].append(answer)
                flattened["text"].append(build_text(question, answer, args.text_mode))
                flattened["source"].append(source)
                flattened["author_type"].append("human")
                flattened["label"].append(0)

            for answer_idx, answer in enumerate(ai_answers):
                flattened["example_id"].append(f"{split_name}-{row_index}-ai-{answer_idx}")
                flattened["question"].append(question)
                flattened["answer"].append(answer)
                flattened["text"].append(build_text(question, answer, args.text_mode))
                flattened["source"].append(source)
                flattened["author_type"].append("ai")
                flattened["label"].append(1)

        return flattened

    return split_dataset.map(
        explode_batch,
        batched=True,
        with_indices=True,
        remove_columns=split_dataset.column_names,
        desc=f"Flattening {split_name} into classification examples",
    )


def normalize_flat_split(split_dataset: Dataset) -> Dataset:
    normalized = split_dataset
    if "example_id" not in normalized.column_names:
        normalized = normalized.map(
            lambda _, idx: {"example_id": f"flat-{idx}"},
            with_indices=True,
            desc="Adding example ids",
        )
    if "answer" not in normalized.column_names:
        normalized = normalized.map(
            lambda example: {"answer": safe_text(example["text"])},
            desc="Adding answer column",
        )
    if "question" not in normalized.column_names:
        normalized = normalized.map(
            lambda _: {"question": ""},
            desc="Adding question column",
        )
    if "source" not in normalized.column_names:
        normalized = normalized.map(
            lambda _: {"source": None},
            desc="Adding source column",
        )

    label_feature = normalized.features.get("label")
    if isinstance(label_feature, ClassLabel):
        name_map = {name.lower(): idx for idx, name in enumerate(label_feature.names)}

        def remap_class_label(example: dict[str, Any]) -> dict[str, int]:
            label_name = label_feature.int2str(example["label"]).lower()
            if label_name in ("human", "human_written"):
                return {"label": 0}
            if label_name in ("ai", "machine", "chatgpt", "generated"):
                return {"label": 1}
            if label_name in name_map and name_map[label_name] in (0, 1):
                return {"label": name_map[label_name]}
            raise ValueError(f"Unsupported label name: {label_name}")

        normalized = normalized.map(remap_class_label, desc="Normalizing class labels")
    else:

        def remap_label(example: dict[str, Any]) -> dict[str, int]:
            raw_label = example["label"]
            if isinstance(raw_label, str):
                label = raw_label.strip().lower()
                if label in ("0", "human", "human_written"):
                    return {"label": 0}
                if label in ("1", "ai", "machine", "chatgpt", "generated"):
                    return {"label": 1}
                raise ValueError(f"Unsupported string label: {raw_label}")
            if raw_label in (0, 1):
                return {"label": int(raw_label)}
            raise ValueError(f"Unsupported numeric label: {raw_label}")

        normalized = normalized.map(remap_label, desc="Normalizing labels")

    if "author_type" not in normalized.column_names:
        normalized = normalized.map(
            lambda example: {"author_type": label_to_name(int(example["label"]))},
            desc="Adding author_type from labels",
        )

    return normalized


def flatten_all_splits(dataset_dict: DatasetDict, args: argparse.Namespace) -> Dataset:
    flattened_parts = []
    for split_name, split_dataset in dataset_dict.items():
        flat_split = flatten_hc3_split(split_dataset, args=args, split_name=split_name)
        flattened_parts.append(flat_split)
    if not flattened_parts:
        raise ValueError("No dataset splits were found.")
    flattened = flattened_parts[0] if len(flattened_parts) == 1 else concatenate_datasets(flattened_parts)
    flattened = flattened.filter(lambda example: bool(safe_text(example["text"])), desc="Dropping empty texts")
    flattened = flattened.cast_column("label", ClassLabel(names=[ID2LABEL[0], ID2LABEL[1]]))
    return flattened.shuffle(seed=args.seed)


def split_dataset(flattened_dataset: Dataset, args: argparse.Namespace) -> DatasetDict:
    if not 0.0 < args.validation_size < 1.0:
        raise ValueError("--validation-size must be between 0 and 1.")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.validation_size + args.test_size >= 1.0:
        raise ValueError("--validation-size + --test-size must be < 1.")

    first_split = flattened_dataset.train_test_split(
        test_size=args.test_size,
        seed=args.seed,
        stratify_by_column="label",
    )
    validation_ratio = args.validation_size / (1.0 - args.test_size)
    second_split = first_split["train"].train_test_split(
        test_size=validation_ratio,
        seed=args.seed,
        stratify_by_column="label",
    )

    return DatasetDict(
        {
            "train": second_split["train"],
            "validation": second_split["test"],
            "test": first_split["test"],
        }
    )


def subsample_split(dataset_split: Dataset, *, ratio: float | None, max_samples: int | None, seed: int) -> Dataset:
    result = dataset_split
    target_size = len(result)

    if ratio is not None and ratio < 1.0:
        if ratio <= 0.0:
            raise ValueError("Subsample ratios must be > 0.")
        target_size = min(target_size, max(1, int(round(len(result) * ratio))))
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("Max sample arguments must be > 0.")
        target_size = min(target_size, max_samples)

    if target_size >= len(result):
        return result

    try:
        return result.train_test_split(
            train_size=target_size,
            seed=seed,
            stratify_by_column="label",
        )["train"]
    except ValueError:
        return result.shuffle(seed=seed).select(range(target_size))


def apply_split_caps(dataset_dict: DatasetDict, args: argparse.Namespace) -> DatasetDict:
    return DatasetDict(
        {
            "train": subsample_split(
                dataset_dict["train"],
                ratio=args.train_subsample_ratio,
                max_samples=args.max_train_samples,
                seed=args.seed,
            ),
            "validation": subsample_split(
                dataset_dict["validation"],
                ratio=None,
                max_samples=args.max_validation_samples,
                seed=args.seed,
            ),
            "test": subsample_split(
                dataset_dict["test"],
                ratio=None,
                max_samples=args.max_test_samples,
                seed=args.seed,
            ),
        }
    )


def label_to_name(label: int) -> str:
    return ID2LABEL[int(label)]


def build_metric_function():
    hf_metrics = None
    if evaluate is not None:
        try:
            hf_metrics = {
                "accuracy": evaluate.load("accuracy"),
                "precision": evaluate.load("precision"),
                "recall": evaluate.load("recall"),
                "f1": evaluate.load("f1"),
            }
        except Exception:
            hf_metrics = None

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        if hf_metrics is not None:
            return {
                "accuracy": hf_metrics["accuracy"].compute(
                    predictions=predictions,
                    references=labels,
                )["accuracy"],
                "precision": hf_metrics["precision"].compute(
                    predictions=predictions,
                    references=labels,
                    average="binary",
                )["precision"],
                "recall": hf_metrics["recall"].compute(
                    predictions=predictions,
                    references=labels,
                    average="binary",
                )["recall"],
                "f1": hf_metrics["f1"].compute(
                    predictions=predictions,
                    references=labels,
                    average="binary",
                )["f1"],
            }

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return compute_metrics


def validate_tracker_setup(args: argparse.Namespace) -> None:
    if args.report_to != "wandb":
        return

    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "W&B tracking was requested but the 'wandb' package is not installed. "
            "Install it with 'pip install wandb' or use '--report-to none'."
        ) from exc


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.model_name:
        return args.model_name
    return MODEL_CHOICES[args.model_choice]


def resolve_warmup_steps(args: argparse.Namespace, train_size: int) -> int:
    if args.warmup_steps is not None:
        if args.warmup_steps < 0:
            raise ValueError("--warmup-steps must be >= 0.")
        return args.warmup_steps

    effective_batch = max(1, args.per_device_train_batch_size * args.gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(train_size / effective_batch))
    total_steps = max(1, math.ceil(steps_per_epoch * args.num_train_epochs))
    return int(total_steps * args.warmup_ratio)


def resolve_empty_cache_steps(args: argparse.Namespace, backend: str) -> int | None:
    if args.torch_empty_cache_steps is not None:
        if args.torch_empty_cache_steps <= 0:
            raise ValueError("--torch-empty-cache-steps must be > 0.")
        return args.torch_empty_cache_steps
    if backend == "mps":
        return 10
    return None


def build_training_arguments(
    args: argparse.Namespace,
    backend: str,
    train_size: int,
) -> TrainingArguments:
    precision_settings = resolve_precision_mode(args.mixed_precision, backend)
    signature = inspect.signature(TrainingArguments.__init__)
    warmup_steps = resolve_warmup_steps(args, train_size)
    empty_cache_steps = resolve_empty_cache_steps(args, backend)

    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": warmup_steps,
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
        "fp16": precision_settings["fp16"],
        "bf16": precision_settings["bf16"],
        "seed": args.seed,
        "report_to": [] if args.report_to == "none" else [args.report_to],
        "save_total_limit": 2,
    }

    if args.run_name:
        kwargs["run_name"] = args.run_name

    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    if "use_mps_device" in signature.parameters and backend == "mps":
        kwargs["use_mps_device"] = True

    if empty_cache_steps is not None and "torch_empty_cache_steps" in signature.parameters:
        kwargs["torch_empty_cache_steps"] = empty_cache_steps

    return TrainingArguments(**kwargs)


def build_trainer(
    model: Any,
    training_args: TrainingArguments,
    tokenized_dataset: DatasetDict,
    tokenizer: Any,
) -> Trainer:
    trainer_signature = inspect.signature(Trainer.__init__)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_dataset["train"],
        "eval_dataset": tokenized_dataset["validation"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": build_metric_function(),
    }

    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    return Trainer(**trainer_kwargs)


def prepare_output_dir(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        return
    if args.resume_from_checkpoint:
        return
    if args.overwrite_output_dir:
        shutil.rmtree(output_dir)
        return
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} already exists and is not empty. "
            "Pass --overwrite-output-dir or choose a new --output-dir."
        )


def cleanup_device_memory(backend: str) -> None:
    gc.collect()
    if backend == "cuda":
        torch.cuda.empty_cache()
    elif backend == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def tokenize_splits(
    dataset_dict: DatasetDict,
    tokenizer: Any,
    max_length: int,
) -> DatasetDict:
    columns_to_remove = [column for column in dataset_dict["train"].column_names if column != "label"]

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    return dataset_dict.map(
        tokenize_batch,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing dataset",
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def save_prediction_records(
    raw_split: Dataset,
    prediction_output: Any,
    output_path: Path,
) -> None:
    probabilities = softmax(prediction_output.predictions)
    predicted_labels = np.argmax(prediction_output.predictions, axis=-1)

    with output_path.open("w", encoding="utf-8") as file_handle:
        for example, predicted_label, probability_vector in zip(
            raw_split,
            predicted_labels,
            probabilities,
        ):
            gold_label = int(example["label"])
            record = {
                "example_id": example["example_id"],
                "source": example["source"],
                "author_type": example["author_type"],
                "question": example["question"],
                "answer": example["answer"],
                "text": example["text"],
                "gold_label": label_to_name(gold_label),
                "predicted_label": label_to_name(int(predicted_label)),
                "confidence": float(probability_vector[int(predicted_label)]),
                "probabilities": {
                    "human": float(probability_vector[0]),
                    "ai": float(probability_vector[1]),
                },
                "correct": int(predicted_label) == gold_label,
            }
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_misclassified_only(predictions_path: Path, output_path: Path) -> None:
    with predictions_path.open("r", encoding="utf-8") as input_handle, output_path.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line in input_handle:
            record = json.loads(line)
            if not record["correct"]:
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    validate_tracker_setup(args)
    prepare_output_dir(args)

    backend = detect_backend()
    source_dataset = load_source_dataset(args)
    flattened_dataset = flatten_all_splits(source_dataset, args)
    split_dataset_dict = split_dataset(flattened_dataset, args)
    split_dataset_dict = apply_split_caps(split_dataset_dict, args)
    model_name = resolve_model_name(args)

    if args.save_splits_dir:
        save_splits_path = Path(args.save_splits_dir)
        save_splits_path.parent.mkdir(parents=True, exist_ok=True)
        split_dataset_dict.save_to_disk(str(save_splits_path))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = tokenize_splits(split_dataset_dict, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    enable_gradient_checkpointing = args.gradient_checkpointing or backend == "mps"
    if enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    cleanup_device_memory(backend)
    training_args = build_training_arguments(args, backend, len(split_dataset_dict["train"]))
    trainer = build_trainer(model, training_args, tokenized_dataset, tokenizer)

    print(f"Device backend: {backend}")
    print(f"Model checkpoint: {model_name}")
    print(
        "Split sizes: "
        f"train={len(split_dataset_dict['train']):,}, "
        f"validation={len(split_dataset_dict['validation']):,}, "
        f"test={len(split_dataset_dict['test']):,}"
    )
    print(f"Max sequence length: {args.max_length}")
    print(f"Gradient checkpointing: {enable_gradient_checkpointing}")

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    validation_metrics = trainer.evaluate(
        eval_dataset=tokenized_dataset["validation"],
        metric_key_prefix="validation",
    )
    cleanup_device_memory(backend)
    trainer.log_metrics("validation", validation_metrics)
    trainer.save_metrics("validation", validation_metrics)

    test_output = trainer.predict(
        tokenized_dataset["test"],
        metric_key_prefix="test",
    )
    cleanup_device_memory(backend)
    trainer.log_metrics("test", test_output.metrics)
    trainer.save_metrics("test", test_output.metrics)
    trainer.save_metrics(
        "all",
        {
            **train_result.metrics,
            **validation_metrics,
            **test_output.metrics,
        },
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "test_predictions.jsonl"
    misclassified_path = output_dir / "test_misclassified.jsonl"
    save_prediction_records(split_dataset_dict["test"], test_output, predictions_path)
    write_misclassified_only(predictions_path, misclassified_path)

    print(f"Saved model and metrics to: {output_dir}")
    print(f"Saved test predictions to: {predictions_path}")
    print(f"Saved misclassified examples to: {misclassified_path}")


if __name__ == "__main__":
    main()
