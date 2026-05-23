#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

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
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export random augmented dataset examples with question, one human answer, "
            "and every selected AI answer column."
        ),
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--split",
        default=None,
        help="Split to sample from. Defaults to eval if present, otherwise train.",
    )
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", default="outputs/random_augmented_examples.txt")
    parser.add_argument("--question-column", default=DEFAULT_QUESTION_COLUMN)
    parser.add_argument("--human-answers-column", default=DEFAULT_HUMAN_ANSWERS_COLUMN)
    parser.add_argument("--selected-human-column", default=DEFAULT_SELECTED_HUMAN_COLUMN)
    parser.add_argument("--source-column", default=DEFAULT_SOURCE_COLUMN)
    parser.add_argument(
        "--ai-answer-columns",
        nargs="+",
        default=None,
        help="AI columns to show. Defaults to chatgpt_answers plus generated ai_* columns.",
    )
    parser.add_argument("--ai-column-prefix", default="ai_")
    parser.add_argument(
        "--exclude-chatgpt",
        action="store_true",
        help="Do not include the original HC3 chatgpt_answers column in auto-detected AI columns.",
    )
    parser.add_argument(
        "--max-answers-per-column",
        type=int,
        default=3,
        help="For list-valued answer columns, show at most this many answers.",
    )
    parser.add_argument(
        "--max-answer-chars",
        type=int,
        default=None,
        help="Optional character cap per answer in the output text file.",
    )
    parser.add_argument("--log-file", default=None)
    return parser


def resolve_split(dataset_dict: Any, requested_split: str | None) -> str:
    if requested_split:
        if requested_split not in dataset_dict:
            raise ValueError(
                f"Split '{requested_split}' not found. Available splits: {', '.join(dataset_dict.keys())}"
            )
        return requested_split
    if "eval" in dataset_dict:
        return "eval"
    if "validation" in dataset_dict:
        return "validation"
    if "train" in dataset_dict:
        return "train"
    return next(iter(dataset_dict.keys()))


def resolve_ai_columns(column_names: list[str], args: argparse.Namespace) -> list[str]:
    available = set(column_names)
    if args.ai_answer_columns:
        columns: list[str] = []
        seen: set[str] = set()
        for raw_value in args.ai_answer_columns:
            for column in raw_value.split(","):
                column = column.strip()
                if column and column not in seen:
                    seen.add(column)
                    columns.append(column)
        missing = sorted(column for column in columns if column not in available)
        if missing:
            raise ValueError(f"Requested AI column(s) not found: {', '.join(missing)}")
        return columns

    columns = []
    if not args.exclude_chatgpt and "chatgpt_answers" in available:
        columns.append("chatgpt_answers")
    columns.extend(
        column
        for column in sorted(column_names)
        if column.startswith(args.ai_column_prefix) and not column.startswith("prompt_")
    )
    if not columns:
        raise ValueError("No AI answer columns found. Pass --ai-answer-columns explicitly.")
    return columns


def truncate_text(text: str, max_chars: int | None) -> str:
    text = safe_text(text)
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def selected_human_answer(example: dict[str, Any], args: argparse.Namespace, split_name: str, row_index: int) -> str:
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


def sample_indices(total_rows: int, num_examples: int, seed: int) -> list[int]:
    if num_examples <= 0:
        raise ValueError("--num-examples must be > 0.")
    if total_rows <= 0:
        raise ValueError("Cannot sample from an empty split.")
    sample_size = min(num_examples, total_rows)
    return sorted(random.Random(seed).sample(range(total_rows), sample_size))


def format_answer_block(
    *,
    label: str,
    answers: list[str],
    max_answers: int,
    max_chars: int | None,
) -> list[str]:
    lines = [f"{label}:"]
    if not answers:
        lines.append("  [missing]")
        return lines

    selected_answers = answers[:max_answers]
    if len(selected_answers) == 1:
        lines.append(f"  {truncate_text(selected_answers[0], max_chars)}")
    else:
        for answer_index, answer in enumerate(selected_answers, start=1):
            lines.append(f"  [{answer_index}] {truncate_text(answer, max_chars)}")
    if len(answers) > max_answers:
        lines.append(f"  [... {len(answers) - max_answers} more answer(s) omitted]")
    return lines


def write_examples(args: argparse.Namespace) -> Path:
    dataset_dict = load_hc3_dataset(dataset_dir=args.dataset_dir)
    split_name = resolve_split(dataset_dict, args.split)
    split_dataset = dataset_dict[split_name]
    ai_columns = resolve_ai_columns(split_dataset.column_names, args)
    indices = sample_indices(len(split_dataset), args.num_examples, args.seed)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "Random Augmented Dataset Examples",
        f"Dataset: {args.dataset_dir}",
        f"Split: {split_name}",
        f"Rows in split: {len(split_dataset):,}",
        f"Sampled rows: {', '.join(str(index) for index in indices)}",
        f"AI columns: {', '.join(ai_columns)}",
        "",
    ]

    for example_number, row_index in enumerate(indices, start=1):
        example = split_dataset[row_index]
        question = truncate_text(example.get(args.question_column), args.max_answer_chars)
        source = safe_text(example.get(args.source_column)) or "unknown"
        human_answer = selected_human_answer(example, args, split_name, row_index)

        lines.extend(
            [
                "=" * 100,
                f"Example {example_number} | row={row_index} | source={source}",
                "",
                "Question:",
                f"  {question}",
                "",
            ]
        )
        lines.extend(
            format_answer_block(
                label="Human answer",
                answers=[human_answer] if human_answer else [],
                max_answers=1,
                max_chars=args.max_answer_chars,
            )
        )

        for column in ai_columns:
            lines.append("")
            lines.extend(
                format_answer_block(
                    label=f"AI answer - {column}",
                    answers=ensure_text_list(example.get(column)),
                    max_answers=args.max_answers_per_column,
                    max_chars=args.max_answer_chars,
                )
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Saved %s sampled examples to %s", f"{len(indices):,}", output_path)
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_file)
    if args.max_answers_per_column <= 0:
        raise ValueError("--max-answers-per-column must be > 0.")
    write_examples(args)


if __name__ == "__main__":
    main()
