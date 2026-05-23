#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
)

try:
    from scripts.hc3_utils import (
        DEFAULT_DATASET_CONFIG,
        DEFAULT_DATASET_NAME,
        DEFAULT_DATASET_REVISION,
        DEFAULT_HUMAN_ANSWERS_COLUMN,
        DEFAULT_QUESTION_COLUMN,
        DEFAULT_SELECTED_HUMAN_COLUMN,
        DEFAULT_SOURCE_COLUMN,
        build_generation_column_name,
        configure_logging,
        ensure_text_list,
        load_hc3_dataset,
        load_prompt_specs,
        pick_deterministic_text,
        render_template,
        resolve_split_names,
        safe_text,
        sanitize_identifier,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from hc3_utils import (
        DEFAULT_DATASET_CONFIG,
        DEFAULT_DATASET_NAME,
        DEFAULT_DATASET_REVISION,
        DEFAULT_HUMAN_ANSWERS_COLUMN,
        DEFAULT_QUESTION_COLUMN,
        DEFAULT_SELECTED_HUMAN_COLUMN,
        DEFAULT_SOURCE_COLUMN,
        build_generation_column_name,
        configure_logging,
        ensure_text_list,
        load_hc3_dataset,
        load_prompt_specs,
        pick_deterministic_text,
        render_template,
        resolve_split_names,
        safe_text,
        sanitize_identifier,
    )


@dataclass(frozen=True)
class DistributedInfo:
    rank: int
    local_rank: int
    world_size: int
    initialized: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class PromptTask:
    name: str
    spec: dict[str, Any]
    answer_column: str
    prompt_column: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append generated answer columns to an HC3 subset using one or more prompts.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--dataset-dir", default=None, help="Input dataset saved with save_to_disk().")
    dataset_group.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    dataset_group.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    dataset_group.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    dataset_group.add_argument("--cache-dir", default=None)
    dataset_group.add_argument("--splits", nargs="+", default=None, help="Splits to augment. Defaults to all.")
    dataset_group.add_argument("--question-column", default=DEFAULT_QUESTION_COLUMN)
    dataset_group.add_argument("--human-answers-column", default=DEFAULT_HUMAN_ANSWERS_COLUMN)
    dataset_group.add_argument("--selected-human-column", default=DEFAULT_SELECTED_HUMAN_COLUMN)
    dataset_group.add_argument("--source-column", default=DEFAULT_SOURCE_COLUMN)

    prompt_group = parser.add_argument_group("prompts")
    prompt_group.add_argument(
        "--prompt-file",
        default="configs/hc3_generation_prompts.json",
        help="JSON file containing prompt templates.",
    )
    prompt_group.add_argument("--prompt-names", nargs="+", default=None, help="Prompt names to run.")
    prompt_group.add_argument("--column-prefix", default="ai")
    prompt_group.add_argument(
        "--model-alias",
        default=None,
        help="Short model name used in generated column names. Defaults to sanitized --model-name.",
    )
    prompt_group.add_argument("--save-prompts", action="store_true", help="Save rendered prompts as columns.")
    prompt_group.add_argument("--disable-chat-template", action="store_true")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model-name", required=True)
    model_group.add_argument("--trust-remote-code", action="store_true")
    model_group.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    model_group.add_argument("--torch-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    model_group.add_argument("--max-input-length", type=int, default=1024)

    generation_group = parser.add_argument_group("generation")
    generation_group.add_argument("--batch-size", type=int, default=4)
    generation_group.add_argument("--max-new-tokens", type=int, default=170)
    generation_group.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    generation_group.add_argument("--temperature", type=float, default=0.7)
    generation_group.add_argument("--top-p", type=float, default=0.9)
    generation_group.add_argument("--num-beams", type=int, default=1)
    generation_group.add_argument("--num-return-sequences", type=int, default=1)
    generation_group.add_argument("--seed", type=int, default=42)

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Where to save the augmented dataset. If it already contains a saved dataset, "
            "that dataset is loaded and new columns are appended."
        ),
    )
    output_group.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Ignore an existing output dataset and rebuild from --dataset-dir/HF source.",
    )
    output_group.add_argument("--overwrite-columns", action="store_true")
    output_group.add_argument("--skip-existing-columns", action="store_true")
    output_group.add_argument("--keep-temp-shards", action="store_true")
    output_group.add_argument("--log-file", default=None)
    return parser


def init_distributed() -> DistributedInfo:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    initialized = False

    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available but WORLD_SIZE > 1.")
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, timeout=timedelta(hours=24))
            initialized = True
        else:
            initialized = True

    return DistributedInfo(rank=rank, local_rank=local_rank, world_size=world_size, initialized=initialized)


def barrier(info: DistributedInfo) -> None:
    if info.world_size > 1 and dist.is_initialized():
        dist.barrier()


def destroy_distributed(info: DistributedInfo) -> None:
    if info.world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def resolve_device(requested_device: str, info: DistributedInfo) -> torch.device:
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if requested_device in ("auto", "cuda") and torch.cuda.is_available():
        if info.world_size > 1:
            torch.cuda.set_device(info.local_rank)
            return torch.device("cuda", info.local_rank)
        return torch.device("cuda")
    if requested_device == "cuda":
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device("cpu")


def resolve_torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def dtype_to_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return "float32"


def load_base_dataset(args: argparse.Namespace, output_dir: Path) -> tuple[DatasetDict, str | None]:
    output_has_dataset = (output_dir / "dataset_dict.json").exists()
    if output_has_dataset and not args.overwrite_output_dir:
        logging.info("Loading existing output dataset from %s so new columns are appended.", output_dir)
        return load_hc3_dataset(dataset_dir=output_dir), str(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_output_dir and not output_has_dataset:
        raise FileExistsError(
            f"{output_dir} exists but does not look like a saved DatasetDict. "
            "Choose another --output-dir or pass --overwrite-output-dir."
        )

    dataset = load_hc3_dataset(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_revision=args.dataset_revision,
        cache_dir=args.cache_dir,
    )
    return dataset, args.dataset_dir


def select_prompt_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = load_prompt_specs(args.prompt_file)
    if not args.prompt_names:
        return specs

    requested = set(args.prompt_names)
    selected = [spec for spec in specs if spec["name"] in requested]
    missing = sorted(requested.difference({spec["name"] for spec in specs}))
    if missing:
        raise ValueError(f"Prompt name(s) not found in {args.prompt_file}: {', '.join(missing)}")
    return selected


def build_tasks(
    args: argparse.Namespace,
    prompt_specs: list[dict[str, Any]],
    dataset_dict: DatasetDict,
    split_names: list[str],
) -> list[PromptTask]:
    tasks: list[PromptTask] = []
    skipped: list[str] = []
    for spec in prompt_specs:
        column = build_generation_column_name(
            model_name=args.model_name,
            model_alias=args.model_alias,
            prompt_name=spec["name"],
            prefix=args.column_prefix,
        )
        prompt_column = f"prompt_{column}" if args.save_prompts else None
        conflicts = []
        for split_name in split_names:
            columns = set(dataset_dict[split_name].column_names)
            for candidate in (column, prompt_column):
                if candidate and candidate in columns:
                    conflicts.append(f"{split_name}.{candidate}")
        if conflicts and args.skip_existing_columns:
            skipped.append(column)
            continue
        if conflicts and not args.overwrite_columns:
            raise ValueError(
                f"Column conflict for {column}: {', '.join(conflicts)}. "
                "Use --overwrite-columns or --skip-existing-columns."
            )
        tasks.append(PromptTask(name=spec["name"], spec=spec, answer_column=column, prompt_column=prompt_column))

    if skipped:
        logging.info("Skipped existing generated columns: %s", ", ".join(skipped))
    return tasks


def validate_dataset_columns(dataset_dict: DatasetDict, args: argparse.Namespace, split_names: list[str]) -> None:
    for split_name in split_names:
        split_dataset = dataset_dict[split_name]
        required = {args.question_column}
        missing = sorted(required.difference(split_dataset.column_names))
        if missing:
            raise ValueError(f"Split '{split_name}' is missing required columns: {', '.join(missing)}")


def load_generation_components(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any, bool, torch.dtype]:
    dtype = resolve_torch_dtype(args.torch_dtype, device)
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    model_class = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
    model = model_class.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            model.resize_token_embeddings(len(tokenizer))

    if not config.is_encoder_decoder:
        tokenizer.padding_side = "left"
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()
    return model, tokenizer, config.is_encoder_decoder, dtype


def prompt_generation_value(spec: dict[str, Any], key: str, default: Any) -> Any:
    generation = spec.get("generation") or {}
    if not isinstance(generation, dict):
        raise ValueError(f"Prompt '{spec['name']}' has a non-object generation override.")
    return generation.get(key, default)


def build_generation_kwargs(args: argparse.Namespace, tokenizer: Any, spec: dict[str, Any]) -> dict[str, Any]:
    max_new_tokens = int(prompt_generation_value(spec, "max_new_tokens", args.max_new_tokens))
    do_sample = bool(prompt_generation_value(spec, "do_sample", args.do_sample))
    temperature = float(prompt_generation_value(spec, "temperature", args.temperature))
    top_p = float(prompt_generation_value(spec, "top_p", args.top_p))
    num_beams = int(prompt_generation_value(spec, "num_beams", args.num_beams))
    num_return_sequences = int(
        prompt_generation_value(spec, "num_return_sequences", args.num_return_sequences)
    )

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0.")
    if num_return_sequences <= 0:
        raise ValueError("num_return_sequences must be > 0.")

    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": num_return_sequences,
    }
    if tokenizer.pad_token_id is not None:
        kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id

    if do_sample:
        if temperature <= 0:
            raise ValueError("temperature must be > 0 when sampling.")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1].")
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    else:
        if num_beams <= 0:
            raise ValueError("num_beams must be > 0.")
        kwargs["num_beams"] = max(num_beams, num_return_sequences)
    return kwargs


def get_selected_human_answer(
    example: dict[str, Any],
    *,
    args: argparse.Namespace,
    split_name: str,
    row_index: int,
) -> str:
    if args.selected_human_column in example:
        text = safe_text(example.get(args.selected_human_column))
        if text:
            return text
    answer, _ = pick_deterministic_text(
        example.get(args.human_answers_column),
        seed=args.seed,
        split_name=split_name,
        row_index=row_index,
        salt=args.selected_human_column,
    )
    return answer


def build_human_answer_pool(split_dataset: Dataset, args: argparse.Namespace, split_name: str) -> list[tuple[int, str]]:
    pool: list[tuple[int, str]] = []
    for row_index in range(len(split_dataset)):
        example = split_dataset[row_index]
        answer = get_selected_human_answer(
            example,
            args=args,
            split_name=split_name,
            row_index=row_index,
        )
        if answer:
            pool.append((row_index, answer))
    return pool


def pick_random_human_example(
    pool: list[tuple[int, str]],
    *,
    seed: int,
    split_name: str,
    row_index: int,
    prompt_name: str,
) -> str:
    if not pool:
        return ""
    rng = random.Random(f"{seed}:{split_name}:{row_index}:{prompt_name}:random_human")
    if len(pool) == 1:
        return pool[0][1]
    candidates = pool
    for _ in range(10):
        candidate_index, answer = rng.choice(candidates)
        if candidate_index != row_index:
            return answer
    return rng.choice(candidates)[1]


def example_from_batch(batch: dict[str, list[Any]], offset: int) -> dict[str, Any]:
    return {column: values[offset] for column, values in batch.items()}


def count_words(text: str) -> int:
    return len(safe_text(text).split())


def target_word_bounds(*reference_answers: str) -> dict[str, int]:
    raw_counts = [count_words(answer) for answer in reference_answers if safe_text(answer)]
    raw_target = raw_counts[0] if raw_counts else 55
    target = min(95, max(12, raw_target))
    return {
        "target_word_count": target,
        "target_min_words": max(8, int(round(target * 0.75))),
        "target_max_words": min(110, max(14, int(round(target * 1.25)))),
    }


def build_prompt_context(
    example: dict[str, Any],
    *,
    args: argparse.Namespace,
    split_name: str,
    row_index: int,
    prompt_name: str,
    human_pool: list[tuple[int, str]],
) -> dict[str, Any]:
    selected_human = get_selected_human_answer(
        example,
        args=args,
        split_name=split_name,
        row_index=row_index,
    )
    random_human = pick_random_human_example(
        human_pool,
        seed=args.seed,
        split_name=split_name,
        row_index=row_index,
        prompt_name=prompt_name,
    )
    if "random_human" in prompt_name:
        word_bounds = target_word_bounds(random_human, selected_human)
    else:
        word_bounds = target_word_bounds(selected_human, random_human)
    return {
        "question": safe_text(example.get(args.question_column)),
        "source": safe_text(example.get(args.source_column)),
        "split": split_name,
        "row_index": row_index,
        "selected_human_answer": selected_human,
        "actual_human_answer": selected_human,
        "selected_human_word_count": count_words(selected_human),
        "random_human_answer": random_human,
        "random_human_word_count": count_words(random_human),
        **word_bounds,
    }


def should_use_chat_template(tokenizer: Any, args: argparse.Namespace) -> bool:
    return not args.disable_chat_template and bool(getattr(tokenizer, "chat_template", None))


def format_for_model(tokenizer: Any, *, user_prompt: str, system_prompt: str | None, use_chat: bool) -> str:
    if use_chat:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if system_prompt:
        return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"
    return user_prompt


def render_prompt(
    task: PromptTask,
    context: dict[str, Any],
    *,
    tokenizer: Any,
    use_chat: bool,
) -> str:
    user_prompt = render_template(task.spec["user"], context)
    system_template = task.spec.get("system")
    system_prompt = render_template(system_template, context) if system_template else None
    return format_for_model(tokenizer, user_prompt=user_prompt, system_prompt=system_prompt, use_chat=use_chat)


def decode_outputs(
    generated: torch.Tensor,
    *,
    tokenizer: Any,
    input_width: int,
    is_encoder_decoder: bool,
    num_return_sequences: int,
) -> list[str | list[str]]:
    if is_encoder_decoder:
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    else:
        decoded = tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=True)
    decoded = [text.strip() for text in decoded]
    grouped = [
        decoded[index : index + num_return_sequences]
        for index in range(0, len(decoded), num_return_sequences)
    ]
    if num_return_sequences == 1:
        return [items[0] if items else "" for items in grouped]
    return grouped


def generate_batch(
    prompts: list[str],
    *,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    is_encoder_decoder: bool,
    max_input_length: int,
    generation_kwargs: dict[str, Any],
) -> list[str | list[str]]:
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    with torch.inference_mode():
        generated = model.generate(**tokenized, **generation_kwargs)
    return decode_outputs(
        generated,
        tokenizer=tokenizer,
        input_width=tokenized["input_ids"].shape[1],
        is_encoder_decoder=is_encoder_decoder,
        num_return_sequences=generation_kwargs["num_return_sequences"],
    )


def zip_equal_lengths(*sequences: list[Any]) -> zip:
    lengths = [len(sequence) for sequence in sequences]
    if len(set(lengths)) != 1:
        raise RuntimeError(f"Expected equal sequence lengths, got: {lengths}")
    return zip(*sequences)


def shard_path(tmp_dir: Path, split_name: str, task_name: str, rank: int) -> Path:
    split_part = sanitize_identifier(split_name, fallback="split")
    task_part = sanitize_identifier(task_name, fallback="prompt")
    return tmp_dir / f"{split_part}__{task_part}__rank{rank}.jsonl"


def generate_shard(
    split_dataset: Dataset,
    *,
    split_name: str,
    task: PromptTask,
    args: argparse.Namespace,
    info: DistributedInfo,
    tmp_dir: Path,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    is_encoder_decoder: bool,
) -> None:
    human_pool = build_human_answer_pool(split_dataset, args, split_name)
    assigned_indices = list(range(info.rank, len(split_dataset), info.world_size))
    generation_kwargs = build_generation_kwargs(args, tokenizer, task.spec)
    use_chat = should_use_chat_template(tokenizer, args)
    output_path = shard_path(tmp_dir, split_name, task.name, info.rank)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Rank %d generating %s rows for split=%s prompt=%s column=%s",
        info.rank,
        f"{len(assigned_indices):,}",
        split_name,
        task.name,
        task.answer_column,
    )
    with output_path.open("w", encoding="utf-8") as file_handle:
        for start in range(0, len(assigned_indices), args.batch_size):
            batch_indices = assigned_indices[start : start + args.batch_size]
            batch = split_dataset[batch_indices] if batch_indices else {}
            prompts: list[str] = []

            for offset, row_index in enumerate(batch_indices):
                example = example_from_batch(batch, offset)
                context = build_prompt_context(
                    example,
                    args=args,
                    split_name=split_name,
                    row_index=row_index,
                    prompt_name=task.name,
                    human_pool=human_pool,
                )
                prompts.append(render_prompt(task, context, tokenizer=tokenizer, use_chat=use_chat))

            answers = generate_batch(
                prompts,
                tokenizer=tokenizer,
                model=model,
                device=device,
                is_encoder_decoder=is_encoder_decoder,
                max_input_length=args.max_input_length,
                generation_kwargs=generation_kwargs,
            )
            for row_index, answer, prompt in zip_equal_lengths(batch_indices, answers, prompts):
                payload: dict[str, Any] = {"index": row_index, "answer": answer}
                if task.prompt_column:
                    payload["prompt"] = prompt
                file_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

            logging.info(
                "Rank %d finished %s/%s rows for split=%s prompt=%s",
                info.rank,
                f"{min(start + args.batch_size, len(assigned_indices)):,}",
                f"{len(assigned_indices):,}",
                split_name,
                task.name,
            )


def read_task_shards(
    *,
    split_dataset: Dataset,
    split_name: str,
    task: PromptTask,
    info: DistributedInfo,
    tmp_dir: Path,
) -> tuple[list[Any], list[str] | None]:
    answers: list[Any] = [None] * len(split_dataset)
    prompts: list[str] | None = [None] * len(split_dataset) if task.prompt_column else None

    for rank in range(info.world_size):
        path = shard_path(tmp_dir, split_name, task.name, rank)
        if not path.exists():
            raise FileNotFoundError(f"Missing generation shard: {path}")
        with path.open("r", encoding="utf-8") as file_handle:
            for line_number, line in enumerate(file_handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                row_index = int(payload["index"])
                if row_index < 0 or row_index >= len(split_dataset):
                    raise ValueError(f"Invalid row index {row_index} in {path}:{line_number}")
                answers[row_index] = payload.get("answer", "")
                if prompts is not None:
                    prompts[row_index] = safe_text(payload.get("prompt"))

    missing_answers = [index for index, answer in enumerate(answers) if answer is None]
    if missing_answers:
        preview = ", ".join(str(index) for index in missing_answers[:10])
        raise RuntimeError(f"Missing generated answers for split={split_name} prompt={task.name}: {preview}")

    if prompts is not None:
        prompts = [safe_text(prompt) for prompt in prompts]
    return answers, prompts


def remove_columns_if_present(split_dataset: Dataset, columns: list[str | None]) -> Dataset:
    removable = [column for column in columns if column and column in split_dataset.column_names]
    return split_dataset.remove_columns(removable) if removable else split_dataset


def merge_shards_into_dataset(
    dataset_dict: DatasetDict,
    *,
    split_names: list[str],
    tasks: list[PromptTask],
    info: DistributedInfo,
    tmp_dir: Path,
) -> DatasetDict:
    updated_splits: dict[str, Dataset] = {}
    for split_name, split_dataset in dataset_dict.items():
        updated_split = split_dataset
        if split_name in split_names:
            for task in tasks:
                answers, prompts = read_task_shards(
                    split_dataset=split_dataset,
                    split_name=split_name,
                    task=task,
                    info=info,
                    tmp_dir=tmp_dir,
                )
                updated_split = remove_columns_if_present(
                    updated_split,
                    [task.answer_column, task.prompt_column],
                )
                updated_split = updated_split.add_column(task.answer_column, answers)
                if task.prompt_column and prompts is not None:
                    updated_split = updated_split.add_column(task.prompt_column, prompts)
        updated_splits[split_name] = updated_split
    return DatasetDict(updated_splits)


def read_existing_runs(output_dir: Path) -> list[dict[str, Any]]:
    runs_path = output_dir / "augmentation_runs.jsonl"
    if not runs_path.exists():
        return []
    runs: list[dict[str, Any]] = []
    with runs_path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if line.strip():
                runs.append(json.loads(line))
    return runs


def build_run_metadata(
    args: argparse.Namespace,
    *,
    source_dataset_dir: str | None,
    split_names: list[str],
    tasks: list[PromptTask],
    info: DistributedInfo,
    device: torch.device,
    dtype: torch.dtype,
    use_chat_template: bool,
) -> dict[str, Any]:
    return {
        "run_id": str(uuid.uuid4()),
        "source_dataset_dir": source_dataset_dir,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "splits": split_names,
        "prompt_file": args.prompt_file,
        "prompt_names": [task.name for task in tasks],
        "answer_columns": [task.answer_column for task in tasks],
        "prompt_columns": [task.prompt_column for task in tasks if task.prompt_column],
        "model_name": args.model_name,
        "model_alias": args.model_alias,
        "device": str(device),
        "torch_dtype": dtype_to_name(dtype),
        "world_size": info.world_size,
        "batch_size_per_rank": args.batch_size,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "seed": args.seed,
        "use_chat_template": use_chat_template,
    }


def save_dataset_atomic(
    dataset_dict: DatasetDict,
    *,
    output_dir: Path,
    run_metadata: dict[str, Any],
    previous_runs: list[dict[str, Any]],
) -> None:
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir.parent / f".{output_dir.name}.tmp.{uuid.uuid4().hex}"
    backup_dir = output_dir.parent / f".{output_dir.name}.backup.{uuid.uuid4().hex}"

    dataset_dict.save_to_disk(str(tmp_dir))
    with (tmp_dir / "augmentation_run_config.json").open("w", encoding="utf-8") as file_handle:
        json.dump(run_metadata, file_handle, indent=2)
    with (tmp_dir / "augmentation_runs.jsonl").open("w", encoding="utf-8") as file_handle:
        for run in [*previous_runs, run_metadata]:
            file_handle.write(json.dumps(run, ensure_ascii=False) + "\n")

    try:
        if output_dir.exists():
            output_dir.rename(backup_dir)
        tmp_dir.rename(output_dir)
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception:
        if output_dir.exists() and not (output_dir / "dataset_dict.json").exists():
            shutil.rmtree(output_dir)
        if backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def main() -> None:
    args = build_parser().parse_args()
    info = init_distributed()
    configure_logging(args.log_file, rank=info.rank)
    set_seed(args.seed + info.rank)

    try:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be > 0.")
        if args.max_input_length <= 0:
            raise ValueError("--max-input-length must be > 0.")

        output_dir = Path(args.output_dir)
        dataset_dict, source_dataset_dir = load_base_dataset(args, output_dir)
        split_names = resolve_split_names(dataset_dict, args.splits)
        validate_dataset_columns(dataset_dict, args, split_names)
        prompt_specs = select_prompt_specs(args)
        tasks = build_tasks(args, prompt_specs, dataset_dict, split_names)
        if not tasks:
            if info.is_main:
                logging.info("No prompt tasks to run.")
            return

        device = resolve_device(args.device, info)
        model, tokenizer, is_encoder_decoder, dtype = load_generation_components(args, device)
        use_chat_template = should_use_chat_template(tokenizer, args)

        if info.is_main:
            logging.info("Model: %s", args.model_name)
            logging.info("Device: %s | dtype: %s | world_size: %d", device, dtype_to_name(dtype), info.world_size)
            logging.info("Splits: %s", ", ".join(split_names))
            for task in tasks:
                logging.info("Task %s -> %s", task.name, task.answer_column)

        tmp_shard_dir = output_dir.parent / f".{output_dir.name}.generation_shards.{uuid.uuid4().hex}"
        if info.world_size > 1:
            object_list = [str(tmp_shard_dir)] if info.is_main else [None]
            dist.broadcast_object_list(object_list, src=0)
            tmp_shard_dir = Path(object_list[0])

        for split_name in split_names:
            for task in tasks:
                generate_shard(
                    dataset_dict[split_name],
                    split_name=split_name,
                    task=task,
                    args=args,
                    info=info,
                    tmp_dir=tmp_shard_dir,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    is_encoder_decoder=is_encoder_decoder,
                )
                barrier(info)

        if info.is_main:
            augmented = merge_shards_into_dataset(
                dataset_dict,
                split_names=split_names,
                tasks=tasks,
                info=info,
                tmp_dir=tmp_shard_dir,
            )
            previous_runs = [] if args.overwrite_output_dir else read_existing_runs(output_dir)
            run_metadata = build_run_metadata(
                args,
                source_dataset_dir=source_dataset_dir,
                split_names=split_names,
                tasks=tasks,
                info=info,
                device=device,
                dtype=dtype,
                use_chat_template=use_chat_template,
            )
            save_dataset_atomic(
                augmented,
                output_dir=output_dir,
                run_metadata=run_metadata,
                previous_runs=previous_runs,
            )
            logging.info("Saved augmented dataset to %s", output_dir)
            logging.info("Added columns: %s", ", ".join(task.answer_column for task in tasks))
            if not args.keep_temp_shards and tmp_shard_dir.exists():
                shutil.rmtree(tmp_shard_dir)

        barrier(info)
    finally:
        destroy_distributed(info)


if __name__ == "__main__":
    main()
