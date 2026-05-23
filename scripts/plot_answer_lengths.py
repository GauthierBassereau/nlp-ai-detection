#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.hc3_utils import (
        DEFAULT_HUMAN_ANSWERS_COLUMN,
        DEFAULT_SELECTED_HUMAN_COLUMN,
        configure_logging,
        ensure_text_list,
        load_hc3_dataset,
        resolve_split_names,
        safe_text,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from hc3_utils import (
        DEFAULT_HUMAN_ANSWERS_COLUMN,
        DEFAULT_SELECTED_HUMAN_COLUMN,
        configure_logging,
        ensure_text_list,
        load_hc3_dataset,
        resolve_split_names,
        safe_text,
    )


DEFAULT_ANSWER_COLUMNS = {
    DEFAULT_HUMAN_ANSWERS_COLUMN,
    DEFAULT_SELECTED_HUMAN_COLUMN,
    "chatgpt_answers",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot answer-length distributions for answer columns in a saved dataset.",
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=None, help="Splits to use. Defaults to all splits.")
    parser.add_argument(
        "--answer-columns",
        nargs="+",
        default=None,
        help="Answer columns to plot. Defaults to human/chatgpt/ai_* columns found in the dataset.",
    )
    parser.add_argument(
        "--ai-column-prefix",
        default="ai_",
        help="Prefix used to auto-detect generated AI answer columns.",
    )
    parser.add_argument(
        "--length-unit",
        choices=("words", "chars", "hf_tokens"),
        default="words",
        help="Length measure. 'words' is a simple whitespace/punctuation token count.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help="Required when --length-unit hf_tokens. Example: answerdotai/ModernBERT-base.",
    )
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional x-axis cap. Longer answers are excluded from the visible curves.",
    )
    parser.add_argument(
        "--percentile-cap",
        type=float,
        default=99.5,
        help="Auto x-axis cap percentile when --max-length is not set. Set 100 to disable.",
    )
    parser.add_argument("--title", default="Answer Length Distributions")
    parser.add_argument("--output-file", default="outputs/answer_length_distribution.png")
    parser.add_argument(
        "--stats-file",
        default=None,
        help="Optional CSV stats path. Defaults to output PNG path with .csv suffix.",
    )
    parser.add_argument("--log-file", default=None)
    return parser


def resolve_answer_columns(dataset_columns: set[str], args: argparse.Namespace) -> list[str]:
    if args.answer_columns:
        columns: list[str] = []
        seen: set[str] = set()
        for raw_value in args.answer_columns:
            for column in raw_value.split(","):
                column = column.strip()
                if column and column not in seen:
                    seen.add(column)
                    columns.append(column)
        missing = sorted(column for column in columns if column not in dataset_columns)
        if missing:
            raise ValueError(f"Requested answer column(s) not found: {', '.join(missing)}")
        return columns

    columns = [
        column
        for column in sorted(dataset_columns)
        if column in DEFAULT_ANSWER_COLUMNS
        or (column.startswith(args.ai_column_prefix) and not column.startswith("prompt_"))
    ]
    if not columns:
        raise ValueError("No answer columns found. Pass --answer-columns explicitly.")
    return columns


def build_length_function(args: argparse.Namespace) -> Callable[[str], int]:
    if args.length_unit == "chars":
        return lambda text: len(text)
    if args.length_unit == "words":
        word_pattern = re.compile(r"\b\w+\b")
        return lambda text: len(word_pattern.findall(text))

    if not args.tokenizer_name:
        raise ValueError("--tokenizer-name is required when --length-unit hf_tokens.")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def collect_lengths(
    *,
    dataset_dir: str,
    split_names: list[str] | None,
    answer_columns: list[str] | None,
    args: argparse.Namespace,
) -> dict[str, list[int]]:
    dataset_dict = load_hc3_dataset(dataset_dir=dataset_dir)
    resolved_splits = resolve_split_names(dataset_dict, split_names)
    common_columns = set(dataset_dict[resolved_splits[0]].column_names)
    for split_name in resolved_splits[1:]:
        common_columns &= set(dataset_dict[split_name].column_names)

    runtime_args = argparse.Namespace(**vars(args))
    runtime_args.answer_columns = answer_columns
    columns = resolve_answer_columns(common_columns, runtime_args)
    length_fn = build_length_function(args)
    lengths_by_column: dict[str, list[int]] = {column: [] for column in columns}

    for split_name in resolved_splits:
        split_dataset = dataset_dict[split_name]
        logging.info("Reading split %s with %s rows.", split_name, f"{len(split_dataset):,}")
        for example in split_dataset:
            for column in columns:
                for answer in ensure_text_list(example.get(column)):
                    answer = safe_text(answer)
                    if answer:
                        lengths_by_column[column].append(length_fn(answer))

    for column, lengths in lengths_by_column.items():
        logging.info("%s: %s answers", column, f"{len(lengths):,}")
    return lengths_by_column


def resolve_plot_cap(lengths_by_column: dict[str, list[int]], args: argparse.Namespace) -> int | None:
    if args.max_length is not None:
        if args.max_length <= 0:
            raise ValueError("--max-length must be > 0.")
        return args.max_length
    if args.percentile_cap >= 100:
        return None
    if not 0 < args.percentile_cap <= 100:
        raise ValueError("--percentile-cap must be in (0, 100].")
    all_lengths = [length for lengths in lengths_by_column.values() for length in lengths]
    if not all_lengths:
        return None
    return max(1, int(math.ceil(np.percentile(all_lengths, args.percentile_cap))))


def histogram_curve(lengths: list[int], *, bins: int, max_length: int | None) -> tuple[np.ndarray, np.ndarray]:
    visible_lengths = np.asarray(
        [length for length in lengths if max_length is None or length <= max_length],
        dtype=np.float64,
    )
    if visible_lengths.size == 0:
        return np.asarray([]), np.asarray([])
    upper = max_length if max_length is not None else int(visible_lengths.max())
    upper = max(1, upper)
    counts, edges = np.histogram(visible_lengths, bins=bins, range=(0, upper), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, counts


def plot_distributions(lengths_by_column: dict[str, list[int]], args: argparse.Namespace) -> Path:
    if args.bins <= 0:
        raise ValueError("--bins must be > 0.")
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_length = resolve_plot_cap(lengths_by_column, args)

    plt.figure(figsize=(12, 7))
    for column, lengths in lengths_by_column.items():
        if not lengths:
            logging.warning("Skipping %s because it has no answers.", column)
            continue
        x_values, y_values = histogram_curve(lengths, bins=args.bins, max_length=max_length)
        if x_values.size == 0:
            logging.warning("Skipping %s because no answers are within the visible range.", column)
            continue
        plt.plot(x_values, y_values, linewidth=2, label=f"{column} (n={len(lengths):,})")

    unit_label = {
        "words": "Word Count",
        "chars": "Character Count",
        "hf_tokens": "Tokenizer Token Count",
    }[args.length_unit]
    plt.title(args.title)
    plt.xlabel(unit_label)
    plt.ylabel("Density")
    if max_length is not None:
        plt.xlim(0, max_length)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def write_stats(lengths_by_column: dict[str, list[int]], stats_path: Path) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=["column", "n", "min", "mean", "median", "p90", "p95", "p99", "max"],
        )
        writer.writeheader()
        for column, lengths in lengths_by_column.items():
            if not lengths:
                writer.writerow({"column": column, "n": 0})
                continue
            values = np.asarray(lengths, dtype=np.float64)
            writer.writerow(
                {
                    "column": column,
                    "n": len(lengths),
                    "min": int(values.min()),
                    "mean": round(float(values.mean()), 3),
                    "median": round(float(np.median(values)), 3),
                    "p90": round(float(np.percentile(values, 90)), 3),
                    "p95": round(float(np.percentile(values, 95)), 3),
                    "p99": round(float(np.percentile(values, 99)), 3),
                    "max": int(values.max()),
                }
            )


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_file)
    lengths_by_column = collect_lengths(
        dataset_dir=args.dataset_dir,
        split_names=args.splits,
        answer_columns=args.answer_columns,
        args=args,
    )
    output_path = plot_distributions(lengths_by_column, args)
    stats_path = Path(args.stats_file) if args.stats_file else output_path.with_suffix(".csv")
    write_stats(lengths_by_column, stats_path)
    logging.info("Saved plot to %s", output_path)
    logging.info("Saved stats to %s", stats_path)


if __name__ == "__main__":
    main()
