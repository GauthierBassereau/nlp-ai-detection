#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import random
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

try:
    from scripts.hc3_utils import (
        DEFAULT_HUMAN_ANSWERS_COLUMN,
        DEFAULT_QUESTION_COLUMN,
        DEFAULT_SELECTED_HUMAN_COLUMN,
        DEFAULT_SOURCE_COLUMN,
        configure_logging,
        load_hc3_dataset,
        safe_text,
    )
    from scripts.train_classifier import (
        ID2LABEL,
        MODEL_CHOICES,
        flatten_split,
        resolve_ai_columns,
        resolve_model_name,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from hc3_utils import (
        DEFAULT_HUMAN_ANSWERS_COLUMN,
        DEFAULT_QUESTION_COLUMN,
        DEFAULT_SELECTED_HUMAN_COLUMN,
        DEFAULT_SOURCE_COLUMN,
        configure_logging,
        load_hc3_dataset,
        safe_text,
    )
    from train_classifier import (
        ID2LABEL,
        MODEL_CHOICES,
        flatten_split,
        resolve_ai_columns,
        resolve_model_name,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the exact text and tokenized inputs used by the classifier.",
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument(
        "--ai-answer-columns",
        nargs="+",
        default=None,
        help="AI columns to inspect. Defaults to generated ai_* columns.",
    )
    parser.add_argument("--ai-column-prefix", default="ai_")
    parser.add_argument("--num-rows", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", default="outputs/classifier_input_examples.txt")

    parser.add_argument("--question-column", default=DEFAULT_QUESTION_COLUMN)
    parser.add_argument("--human-answers-column", default=DEFAULT_HUMAN_ANSWERS_COLUMN)
    parser.add_argument("--selected-human-column", default=DEFAULT_SELECTED_HUMAN_COLUMN)
    parser.add_argument("--source-column", default=DEFAULT_SOURCE_COLUMN)
    parser.add_argument("--text-mode", choices=("answer", "question_answer"), default="answer")
    parser.add_argument(
        "--answer-window-words",
        type=int,
        default=None,
        help="Use the same deterministic random answer windowing as training.",
    )

    parser.add_argument("--model-choice", choices=tuple(MODEL_CHOICES.keys()), default="modernbert-base")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-token-display", type=int, default=160)
    parser.add_argument("--max-text-chars", type=int, default=3000)
    parser.add_argument("--log-file", default=None)
    return parser


def make_training_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        train_split="train",
        eval_split=args.split,
        question_column=args.question_column,
        human_answers_column=args.human_answers_column,
        selected_human_column=args.selected_human_column,
        source_column=args.source_column,
        ai_answer_columns=args.ai_answer_columns,
        ai_column_prefix=args.ai_column_prefix,
        text_mode=args.text_mode,
        answer_window_words=args.answer_window_words,
        max_train_samples=None,
        max_eval_samples=None,
        seed=args.seed,
        model_choice=args.model_choice,
        model_name=args.model_name,
    )


def group_flat_examples_by_row(flat_examples: Any) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for example in flat_examples:
        row_index = int(example["row_index"])
        label_name = ID2LABEL[int(example["label"])]
        grouped[row_index][label_name] = example
    return {
        row_index: examples
        for row_index, examples in grouped.items()
        if "human" in examples and "ai" in examples
    }


def sample_row_indices(grouped_examples: dict[int, dict[str, dict[str, Any]]], args: argparse.Namespace) -> list[int]:
    if args.num_rows <= 0:
        raise ValueError("--num-rows must be > 0.")
    available_indices = sorted(grouped_examples.keys())
    if not available_indices:
        raise ValueError("No paired human/AI rows were found for the requested split and column.")
    sample_size = min(args.num_rows, len(available_indices))
    return sorted(random.Random(args.seed).sample(available_indices, sample_size))


def truncate_text(text: str, max_chars: int) -> str:
    text = safe_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def whitespace_word_count(text: str) -> int:
    text = safe_text(text)
    return len(text.split()) if text else 0


def invisible_character_summary(text: str) -> list[str]:
    summaries: list[str] = []
    for index, character in enumerate(text):
        category = unicodedata.category(character)
        if category.startswith("C") and character not in ("\n", "\t", "\r"):
            name = unicodedata.name(character, "UNKNOWN")
            summaries.append(f"index={index} U+{ord(character):04X} {name} category={category}")
    return summaries


def tokenize_text(tokenizer: Any, text: str, max_length: int) -> dict[str, Any]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True,
    )
    input_ids = encoded["input_ids"]
    return {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"],
        "special_tokens_mask": encoded.get("special_tokens_mask", []),
        "tokens": tokenizer.convert_ids_to_tokens(input_ids),
        "decoded": tokenizer.decode(input_ids, skip_special_tokens=False),
        "truncated": len(tokenizer.encode(text, add_special_tokens=True, truncation=False)) > len(input_ids),
    }


def format_tokenized_example(
    *,
    label: str,
    example: dict[str, Any],
    tokenizer: Any,
    args: argparse.Namespace,
) -> list[str]:
    text = safe_text(example["text"])
    tokenized = tokenize_text(tokenizer, text, args.max_length)
    input_ids = tokenized["input_ids"]
    tokens = tokenized["tokens"]
    display_count = min(args.max_token_display, len(input_ids))
    invisible_summary = invisible_character_summary(text)

    lines = [
        f"{label.upper()} EXAMPLE",
        f"  label_id: {example['label']}",
        f"  answer_source_column: {example['answer_source_column']}",
        f"  source: {example['source']}",
        f"  raw_character_count: {len(text)}",
        f"  whitespace_word_count: {whitespace_word_count(text)}",
        f"  answer_window_start: {example.get('answer_window_start')}",
        f"  answer_window_end: {example.get('answer_window_end')}",
        f"  answer_original_word_count: {example.get('answer_original_word_count')}",
        f"  tokenizer_token_count_with_specials: {len(input_ids)}",
        f"  truncated_at_max_length_{args.max_length}: {tokenized['truncated']}",
        f"  invisible_control_chars: {len(invisible_summary)}",
    ]
    if invisible_summary:
        lines.extend(f"    {item}" for item in invisible_summary[:20])
        if len(invisible_summary) > 20:
            lines.append(f"    ... {len(invisible_summary) - 20} more")

    lines.extend(
        [
            "",
            "  EXACT TEXT FIELD PASSED TO TOKENIZER:",
            "  " + truncate_text(text, args.max_text_chars).replace("\n", "\n  "),
            "",
            "  PYTHON repr(TEXT):",
            "  " + repr(truncate_text(text, args.max_text_chars)),
            "",
            f"  TOKENS first {display_count}/{len(tokens)}:",
            "  " + " ".join(tokens[:display_count]),
            "",
            f"  TOKEN IDS first {display_count}/{len(input_ids)}:",
            "  " + " ".join(str(token_id) for token_id in input_ids[:display_count]),
            "",
            f"  SPECIAL TOKEN MASK first {display_count}/{len(input_ids)}:",
            "  " + " ".join(str(value) for value in tokenized["special_tokens_mask"][:display_count]),
            "",
            "  DECODED FROM INPUT IDS:",
            "  " + truncate_text(tokenized["decoded"], args.max_text_chars).replace("\n", "\n  "),
        ]
    )
    return lines


def export_for_column(
    *,
    dataset_dict: Any,
    ai_column: str,
    tokenizer: Any,
    args: argparse.Namespace,
) -> list[str]:
    training_args = make_training_namespace(args)
    flat_examples = flatten_split(
        dataset_dict[args.split],
        split_name=args.split,
        ai_column=ai_column,
        args=training_args,
    )
    grouped = group_flat_examples_by_row(flat_examples)
    sampled_rows = sample_row_indices(grouped, args)

    lines = [
        "=" * 120,
        f"AI COLUMN: {ai_column}",
        f"Paired rows available: {len(grouped):,}",
        f"Sampled rows: {', '.join(str(row_index) for row_index in sampled_rows)}",
        "",
    ]
    for sample_number, row_index in enumerate(sampled_rows, start=1):
        human_example = grouped[row_index]["human"]
        ai_example = grouped[row_index]["ai"]
        lines.extend(
            [
                "-" * 120,
                f"Sample {sample_number} | dataset row_index={row_index} | source={human_example['source']}",
                "",
                "QUESTION:",
                "  " + safe_text(human_example["question"]).replace("\n", "\n  "),
                "",
            ]
        )
        lines.extend(format_tokenized_example(label="human", example=human_example, tokenizer=tokenizer, args=args))
        lines.append("")
        lines.extend(format_tokenized_example(label="ai", example=ai_example, tokenizer=tokenizer, args=args))
        lines.append("")
    return lines


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_file)
    if args.max_length <= 0:
        raise ValueError("--max-length must be > 0.")
    if args.max_token_display <= 0:
        raise ValueError("--max-token-display must be > 0.")

    dataset_dict = load_hc3_dataset(dataset_dir=args.dataset_dir)
    if args.split not in dataset_dict:
        raise ValueError(f"Split '{args.split}' not found. Available splits: {', '.join(dataset_dict.keys())}")
    training_args = make_training_namespace(args)
    model_name = resolve_model_name(training_args)
    ai_columns = resolve_ai_columns(dataset_dict, training_args)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Classifier Input Inspection",
        f"Dataset: {args.dataset_dir}",
        f"Split: {args.split}",
        f"Text mode: {args.text_mode}",
        f"Answer window words: {args.answer_window_words}",
        f"Tokenizer/model: {model_name}",
        f"Max length: {args.max_length}",
        f"AI columns: {', '.join(ai_columns)}",
        "",
        "The classifier Trainer tokenizes only the TEXT FIELD shown below, then removes all metadata columns.",
        "",
    ]
    for ai_column in ai_columns:
        lines.extend(export_for_column(dataset_dict=dataset_dict, ai_column=ai_column, tokenizer=tokenizer, args=args))
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Saved classifier input inspection to %s", output_path)


if __name__ == "__main__":
    main()
