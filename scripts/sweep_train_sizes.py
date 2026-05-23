#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import logging
import os
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict
from transformers import AutoTokenizer, set_seed

try:
    from scripts import train_classifier as classifier
    from scripts.hc3_utils import configure_logging, load_hc3_dataset, sanitize_identifier
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    import train_classifier as classifier
    from hc3_utils import configure_logging, load_hc3_dataset, sanitize_identifier


def build_parser() -> argparse.ArgumentParser:
    parser = classifier.build_parser()
    parser.description = (
        "Train the same classifier setup on multiple train row counts to measure data scaling."
    )
    sweep_group = parser.add_argument_group("size sweep")
    sweep_group.add_argument(
        "--train-row-sizes",
        "--train-sizes",
        nargs="+",
        type=int,
        required=True,
        help=(
            "Numbers of original dataset train rows to use. Each row contributes one human "
            "and one AI classification example for the selected AI column."
        ),
    )
    sweep_group.add_argument(
        "--size-dir-prefix",
        default="train_rows_",
        help="Directory prefix used under --output-dir for each size.",
    )
    return parser


def distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def validate_train_sizes(train_sizes: list[int], train_row_count: int) -> list[int]:
    if not train_sizes:
        raise ValueError("Pass at least one value to --train-row-sizes.")

    validated: list[int] = []
    seen: set[int] = set()
    for size in train_sizes:
        if size <= 0:
            raise ValueError("--train-row-sizes values must be > 0.")
        if size > train_row_count:
            raise ValueError(
                f"Requested {size:,} train rows, but split only has {train_row_count:,} rows."
            )
        if size not in seen:
            seen.add(size)
            validated.append(size)
    return validated


def dataset_for_train_size(dataset_dict: DatasetDict, args: argparse.Namespace, train_rows: int) -> DatasetDict:
    subset_dict = DatasetDict()
    for split_name, split_dataset in dataset_dict.items():
        if split_name == args.train_split:
            subset_dict[split_name] = split_dataset.shuffle(seed=args.seed).select(range(train_rows))
        else:
            subset_dict[split_name] = split_dataset
    return subset_dict


def run_name_for_size(base_run_name: str | None, train_rows: int) -> str:
    size_name = f"rows{train_rows}"
    if base_run_name:
        return f"{base_run_name}-{size_name}"
    return size_name


def metric_row(size_key: str, train_rows: int, run_output_dir: Path, metrics: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "size_key": size_key,
        "train_rows": train_rows,
        "run_output_dir": str(run_output_dir),
    }
    if metrics is None:
        row["status"] = "skipped"
        return row
    row["status"] = "completed"
    row.update(metrics)
    return row


def write_size_metrics_csv(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    preferred_columns = [
        "size_key",
        "status",
        "train_rows",
        "eval_examples",
        "eval_accuracy",
        "eval_balanced_accuracy",
        "eval_precision",
        "eval_recall",
        "eval_f1",
        "eval_loss",
        "eval_column",
        "run_output_dir",
    ]
    all_columns = set().union(*(row.keys() for row in rows)) if rows else set()
    columns = preferred_columns + sorted(all_columns.difference(preferred_columns))
    classifier.write_csv(output_dir / "size_metrics.csv", rows, columns)


def plot_size_metrics(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    completed_rows = [
        row
        for row in rows
        if row.get("status") == "completed" and row.get("eval_accuracy") is not None
    ]
    if not completed_rows:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    completed_rows = sorted(completed_rows, key=lambda row: int(row["train_rows"]))
    train_rows = [int(row["train_rows"]) for row in completed_rows]
    accuracy = [float(row["eval_accuracy"]) for row in completed_rows]
    f1 = [float(row["eval_f1"]) for row in completed_rows if row.get("eval_f1") is not None]

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(train_rows, accuracy, marker="o", label="accuracy")
    if len(f1) == len(train_rows):
        axis.plot(train_rows, f1, marker="o", label="f1")
    axis.set_xlabel("Training rows")
    axis.set_ylabel("Eval score")
    axis.set_title("Classifier Data Scaling")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "size_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_file, rank=int(os.environ.get("RANK", "0")))
    set_seed(args.seed)
    classifier.set_wandb_env(args)

    if args.max_train_samples is not None and classifier.is_main_process():
        logging.info(
            "Ignoring --max-train-samples=%s because --train-row-sizes controls row counts.",
            args.max_train_samples,
        )
    if args.max_length <= 0:
        raise ValueError("--max-length must be > 0.")

    dataset_dict = load_hc3_dataset(dataset_dir=args.dataset_dir)
    classifier.validate_dataset(dataset_dict, args)
    ai_columns = classifier.resolve_ai_columns(dataset_dict, args)
    if len(ai_columns) != 1:
        raise ValueError(
            "The size sweep trains exactly one AI column. "
            "Pass one value to --ai-answer-columns, for example "
            "--ai-answer-columns ai_qwen25_3b_actual_human_reference."
        )
    ai_column = ai_columns[0]

    train_sizes = validate_train_sizes(args.train_row_sizes, len(dataset_dict[args.train_split]))
    model_name = classifier.resolve_model_name(args)
    backend = classifier.detect_backend()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)
    base_output_dir = Path(args.output_dir)

    if classifier.is_main_process():
        base_output_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Model checkpoint: %s", model_name)
        logging.info("Detected backend: %s", backend)
        logging.info("AI column: %s", ai_column)
        logging.info("Train row sizes: %s", ", ".join(f"{size:,}" for size in train_sizes))

    all_rows: list[dict[str, Any]] = []
    for train_rows in train_sizes:
        size_key = f"{args.size_dir_prefix}{train_rows}"
        run_output_base = base_output_dir / sanitize_identifier(size_key, fallback=f"rows_{train_rows}")
        run_args = copy.deepcopy(args)
        run_args.output_dir = str(run_output_base)
        run_args.max_train_samples = None
        run_args.run_name = run_name_for_size(args.run_name, train_rows)

        if classifier.is_main_process():
            logging.info("Starting size sweep run: %s train rows -> %s", f"{train_rows:,}", run_output_base)

        size_dataset = dataset_for_train_size(dataset_dict, args, train_rows)
        metrics = classifier.train_one_column(
            dataset_dict=size_dataset,
            ai_column=ai_column,
            model_name=model_name,
            tokenizer=tokenizer,
            args=run_args,
            backend=backend,
        )
        distributed_barrier()

        if classifier.is_main_process():
            row = metric_row(size_key, train_rows, run_output_base / sanitize_identifier(ai_column), metrics)
            all_rows.append(row)
            classifier.write_json(base_output_dir / "all_size_metrics.json", all_rows)
            write_size_metrics_csv(base_output_dir, all_rows)
            plot_size_metrics(base_output_dir, all_rows)
            logging.info("Finished size sweep run for %s train rows.", f"{train_rows:,}")

    if classifier.is_main_process():
        logging.info("Saved size sweep metrics to %s", base_output_dir / "all_size_metrics.json")


if __name__ == "__main__":
    main()
