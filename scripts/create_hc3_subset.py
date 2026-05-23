#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets

try:
    from scripts.hc3_utils import (
        DEFAULT_DATASET_CONFIG,
        DEFAULT_DATASET_NAME,
        DEFAULT_DATASET_REVISION,
        DEFAULT_HUMAN_ANSWERS_COLUMN,
        DEFAULT_QUESTION_COLUMN,
        DEFAULT_SELECTED_HUMAN_COLUMN,
        DEFAULT_SELECTED_HUMAN_INDEX_COLUMN,
        DEFAULT_SOURCE_COLUMN,
        configure_logging,
        ensure_text_list,
        load_hc3_dataset,
        pick_deterministic_text,
        resolve_split_names,
        safe_text,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from hc3_utils import (
        DEFAULT_DATASET_CONFIG,
        DEFAULT_DATASET_NAME,
        DEFAULT_DATASET_REVISION,
        DEFAULT_HUMAN_ANSWERS_COLUMN,
        DEFAULT_QUESTION_COLUMN,
        DEFAULT_SELECTED_HUMAN_COLUMN,
        DEFAULT_SELECTED_HUMAN_INDEX_COLUMN,
        DEFAULT_SOURCE_COLUMN,
        configure_logging,
        ensure_text_list,
        load_hc3_dataset,
        pick_deterministic_text,
        resolve_split_names,
        safe_text,
    )


ORIGIN_SPLIT_COLUMN = "hc3_source_split"
ORIGIN_INDEX_COLUMN = "hc3_source_index"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a reproducible train/eval subset from Hello-SimpleAI/HC3.",
    )
    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--dataset-dir", default=None, help="Load an existing saved dataset.")
    dataset_group.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    dataset_group.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    dataset_group.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    dataset_group.add_argument("--cache-dir", default=None)
    dataset_group.add_argument(
        "--source-splits",
        nargs="+",
        default=None,
        help="Input splits to sample from. Defaults to all available splits.",
    )
    dataset_group.add_argument("--question-column", default=DEFAULT_QUESTION_COLUMN)
    dataset_group.add_argument("--human-answers-column", default=DEFAULT_HUMAN_ANSWERS_COLUMN)
    dataset_group.add_argument("--source-column", default=DEFAULT_SOURCE_COLUMN)
    dataset_group.add_argument(
        "--selected-human-column",
        default=DEFAULT_SELECTED_HUMAN_COLUMN,
        help="Column added with one deterministic human answer per row.",
    )
    dataset_group.add_argument(
        "--selected-human-index-column",
        default=DEFAULT_SELECTED_HUMAN_INDEX_COLUMN,
        help="Column added with the index of the selected human answer.",
    )

    sampling_group = parser.add_argument_group("sampling")
    sampling_group.add_argument("--train-size", type=int, required=True)
    sampling_group.add_argument("--eval-size", type=int, required=True)
    sampling_group.add_argument("--seed", type=int, default=42)

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", required=True)
    output_group.add_argument("--overwrite-output-dir", action="store_true")
    output_group.add_argument("--log-file", default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.train_size <= 0:
        raise ValueError("--train-size must be > 0.")
    if args.eval_size <= 0:
        raise ValueError("--eval-size must be > 0.")
    if args.selected_human_column == args.selected_human_index_column:
        raise ValueError("Selected human text and index columns must be different.")


def validate_split_columns(split_dataset: Dataset, args: argparse.Namespace, split_name: str) -> None:
    required = {args.question_column, args.human_answers_column}
    missing = sorted(required.difference(split_dataset.column_names))
    if missing:
        raise ValueError(f"Split '{split_name}' is missing required columns: {', '.join(missing)}")


def with_origin_columns(split_dataset: Dataset, split_name: str) -> Dataset:
    if ORIGIN_SPLIT_COLUMN in split_dataset.column_names:
        split_dataset = split_dataset.remove_columns([ORIGIN_SPLIT_COLUMN])
    if ORIGIN_INDEX_COLUMN in split_dataset.column_names:
        split_dataset = split_dataset.remove_columns([ORIGIN_INDEX_COLUMN])

    def add_origin(batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
        return {
            ORIGIN_SPLIT_COLUMN: [split_name] * len(indices),
            ORIGIN_INDEX_COLUMN: list(indices),
        }

    return split_dataset.map(
        add_origin,
        batched=True,
        with_indices=True,
        desc=f"Adding origin columns to {split_name}",
    )


def row_is_usable(example: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(safe_text(example.get(args.question_column))) and bool(
        ensure_text_list(example.get(args.human_answers_column))
    )


def add_selected_human_answer(split_dataset: Dataset, args: argparse.Namespace, split_name: str) -> Dataset:
    removable = [
        column
        for column in (args.selected_human_column, args.selected_human_index_column)
        if column in split_dataset.column_names
    ]
    if removable:
        split_dataset = split_dataset.remove_columns(removable)

    def select_answer(example: dict[str, Any], row_index: int) -> dict[str, Any]:
        answer, answer_index = pick_deterministic_text(
            example.get(args.human_answers_column),
            seed=args.seed,
            split_name=split_name,
            row_index=row_index,
            salt=args.selected_human_column,
        )
        return {
            args.selected_human_column: answer,
            args.selected_human_index_column: -1 if answer_index is None else answer_index,
        }

    return split_dataset.map(
        select_answer,
        with_indices=True,
        desc=f"Selecting one human answer for {split_name}",
    )


def build_combined_source(dataset_dict: DatasetDict, args: argparse.Namespace) -> Dataset:
    split_names = resolve_split_names(dataset_dict, args.source_splits)
    prepared_splits: list[Dataset] = []

    for split_name in split_names:
        split_dataset = dataset_dict[split_name]
        validate_split_columns(split_dataset, args, split_name)
        split_dataset = with_origin_columns(split_dataset, split_name)
        split_dataset = split_dataset.filter(
            lambda example: row_is_usable(example, args),
            desc=f"Filtering usable rows from {split_name}",
        )
        prepared_splits.append(split_dataset)
        logging.info("Input split %s has %s usable rows.", split_name, f"{len(split_dataset):,}")

    if not prepared_splits:
        raise ValueError("No source splits were selected.")
    return prepared_splits[0] if len(prepared_splits) == 1 else concatenate_datasets(prepared_splits)


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if not path.exists():
        return
    if overwrite:
        shutil.rmtree(path)
        return
    if any(path.iterdir()):
        raise FileExistsError(
            f"{path} already exists and is not empty. Pass --overwrite-output-dir to replace it."
        )


def write_metadata(output_dir: Path, args: argparse.Namespace, source_rows: int) -> Path:
    metadata_path = output_dir / "subset_config.json"
    payload = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "dataset_dir": args.dataset_dir,
        "source_splits": args.source_splits,
        "question_column": args.question_column,
        "human_answers_column": args.human_answers_column,
        "source_column": args.source_column,
        "selected_human_column": args.selected_human_column,
        "selected_human_index_column": args.selected_human_index_column,
        "seed": args.seed,
        "source_rows_after_filtering": source_rows,
        "train_size": args.train_size,
        "eval_size": args.eval_size,
    }
    with metadata_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    return metadata_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    configure_logging(args.log_file)

    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, overwrite=args.overwrite_output_dir)

    dataset_dict = load_hc3_dataset(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_revision=args.dataset_revision,
        cache_dir=args.cache_dir,
    )
    combined = build_combined_source(dataset_dict, args)
    requested_total = args.train_size + args.eval_size
    if requested_total > len(combined):
        raise ValueError(
            f"Requested {requested_total:,} rows but only {len(combined):,} usable rows are available."
        )

    sampled = combined.shuffle(seed=args.seed).select(range(requested_total))
    train_dataset = sampled.select(range(args.train_size))
    eval_dataset = sampled.select(range(args.train_size, requested_total))
    result = DatasetDict(
        {
            "train": add_selected_human_answer(train_dataset, args, "train"),
            "eval": add_selected_human_answer(eval_dataset, args, "eval"),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result.save_to_disk(str(output_dir))
    metadata_path = write_metadata(output_dir, args, source_rows=len(combined))

    logging.info("Saved subset to %s", output_dir)
    logging.info("Rows: train=%s eval=%s", f"{len(result['train']):,}", f"{len(result['eval']):,}")
    logging.info("Saved subset metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
