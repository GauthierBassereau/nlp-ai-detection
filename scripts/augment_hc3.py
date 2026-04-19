#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
)


DEFAULT_DATASET_REVISION = "refs/convert/parquet"
AVAILABLE_TEMPLATE_FIELDS = ("question", "source", "split", "row_index")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add a new AI-answer column to an HC3-style dataset by generating answers "
            "from a configurable Hugging Face model."
        )
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
        "--dataset-revision",
        default=DEFAULT_DATASET_REVISION,
        help="HF dataset revision used when --dataset-dir is not provided.",
    )
    dataset_group.add_argument(
        "--cache-dir",
        default=None,
        help="Optional HF cache directory.",
    )
    dataset_group.add_argument(
        "--question-column",
        default="question",
        help="Column that contains the question text.",
    )
    dataset_group.add_argument(
        "--source-column",
        default="source",
        help="Optional source/domain column exposed to templates as {source}.",
    )
    dataset_group.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Dataset splits to augment. Defaults to every split in the dataset.",
    )
    sampling_group = dataset_group.add_mutually_exclusive_group()
    sampling_group.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional cap on the first N rows of each selected split for quick experiments.",
    )
    sampling_group.add_argument(
        "--random-subset-size",
        type=int,
        default=None,
        help=(
            "Optional total number of rows to randomly sample across the selected "
            "splits before augmentation. The sampled subset is what gets saved."
        ),
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--column-name",
        required=True,
        help="Name of the new answer-list column to add to each row.",
    )
    output_group.add_argument(
        "--prompt-column-name",
        default=None,
        help="Optional extra column storing the exact rendered prompt sent to the model.",
    )
    output_group.add_argument(
        "--output-dir",
        required=True,
        help="Where to save the augmented dataset with datasets.save_to_disk().",
    )
    output_group.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Allow deleting an existing non-empty output directory.",
    )
    output_group.add_argument(
        "--overwrite-column",
        action="store_true",
        help="Replace the target column if it already exists in the input dataset.",
    )

    prompt_group = parser.add_argument_group("prompting")
    prompt_template_sources = prompt_group.add_mutually_exclusive_group()
    prompt_template_sources.add_argument(
        "--prompt-template",
        default=None,
        help=(
            "Optional prompt template. Available fields: "
            "{question}, {source}, {split}, {row_index}."
        ),
    )
    prompt_template_sources.add_argument(
        "--prompt-template-file",
        default=None,
        help="Path to a text file containing the prompt template.",
    )
    system_prompt_sources = prompt_group.add_mutually_exclusive_group()
    system_prompt_sources.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system instruction applied to chat models.",
    )
    system_prompt_sources.add_argument(
        "--system-prompt-file",
        default=None,
        help="Path to a text file containing the system prompt.",
    )
    prompt_group.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Use plain text prompts even if the tokenizer exposes a chat template.",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--model-name",
        required=True,
        help="HF model id or local path used to generate new answers.",
    )
    model_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer and model.",
    )
    model_group.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Execution device. Auto prefers cuda, then mps, then cpu.",
    )
    model_group.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Model dtype. Auto chooses bf16/fp16 on CUDA when possible.",
    )
    model_group.add_argument(
        "--max-input-length",
        type=int,
        default=1024,
        help="Tokenizer truncation length for the rendered prompt.",
    )

    generation_group = parser.add_argument_group("generation")
    generation_group.add_argument("--batch-size", type=int, default=4)
    generation_group.add_argument("--max-new-tokens", type=int, default=256)
    generation_group.add_argument(
        "--do-sample",
        action="store_true",
        help="Sample answers instead of using deterministic decoding.",
    )
    generation_group.add_argument("--temperature", type=float, default=0.8)
    generation_group.add_argument("--top-p", type=float, default=0.95)
    generation_group.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam count used when --do-sample is not set.",
    )
    generation_group.add_argument(
        "--num-return-sequences",
        type=int,
        default=1,
        help="How many answers to generate per question.",
    )
    generation_group.add_argument("--seed", type=int, default=42)

    return parser


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


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
            "revision": args.dataset_revision,
        }
        if args.dataset_config:
            load_kwargs["data_dir"] = args.dataset_config
        dataset_obj = load_dataset(**load_kwargs)
    return normalize_dataset_object(dataset_obj)


def ensure_local_dataset(
    local_dir: str | Path,
    *,
    dataset_name: str = "Hello-SimpleAI/HC3",
    dataset_config: str = "all",
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    cache_dir: str | None = None,
) -> DatasetDict:
    local_path = Path(local_dir)
    if local_path.exists():
        return normalize_dataset_object(load_from_disk(str(local_path)))

    load_kwargs = {
        "path": dataset_name,
        "cache_dir": cache_dir,
        "revision": dataset_revision,
    }
    if dataset_config:
        load_kwargs["data_dir"] = dataset_config

    dataset_obj = load_dataset(**load_kwargs)
    dataset_dict = normalize_dataset_object(dataset_obj)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(local_path))
    return dataset_dict


def resolve_split_names(dataset_dict: DatasetDict, requested_splits: list[str] | None) -> list[str]:
    available_splits = list(dataset_dict.keys())
    if not requested_splits:
        return available_splits

    missing = sorted(set(requested_splits).difference(available_splits))
    if missing:
        missing_text = ", ".join(missing)
        available_text = ", ".join(available_splits)
        raise ValueError(
            f"Unknown split(s): {missing_text}. Available splits: {available_text}"
        )
    return requested_splits


def maybe_random_subset_dataset(
    dataset_dict: DatasetDict,
    *,
    split_names: list[str],
    sample_size: int | None,
    seed: int,
) -> tuple[DatasetDict, dict[str, Any]]:
    ordered_split_names = [split_name for split_name in dataset_dict.keys() if split_name in split_names]
    input_rows_per_split = {split_name: len(dataset_dict[split_name]) for split_name in ordered_split_names}
    input_total_rows = sum(input_rows_per_split.values())
    metadata = {
        "requested_size": sample_size,
        "seed": seed,
        "input_total_rows": input_total_rows,
        "input_rows_per_split": input_rows_per_split,
        "output_total_rows": input_total_rows,
        "output_rows_per_split": dict(input_rows_per_split),
    }

    if sample_size is None:
        return dataset_dict, metadata
    if sample_size <= 0:
        raise ValueError("--random-subset-size must be > 0.")
    if sample_size >= input_total_rows:
        return dataset_dict, metadata

    sampled_global_indices = sorted(random.Random(seed).sample(range(input_total_rows), sample_size))
    sampled_splits: dict[str, Dataset] = {}
    sampled_rows_per_split: dict[str, int] = {}
    sample_cursor = 0
    global_offset = 0

    for split_name, split_dataset in dataset_dict.items():
        if split_name not in input_rows_per_split:
            sampled_splits[split_name] = split_dataset
            continue

        split_length = len(split_dataset)
        selected_indices: list[int] = []
        split_stop = global_offset + split_length

        while (
            sample_cursor < len(sampled_global_indices)
            and sampled_global_indices[sample_cursor] < split_stop
        ):
            selected_indices.append(sampled_global_indices[sample_cursor] - global_offset)
            sample_cursor += 1

        sampled_rows_per_split[split_name] = len(selected_indices)
        sampled_splits[split_name] = split_dataset.select(selected_indices)
        global_offset = split_stop

    metadata["output_total_rows"] = sample_size
    metadata["output_rows_per_split"] = sampled_rows_per_split
    return DatasetDict(sampled_splits), metadata


def read_optional_text_argument(
    inline_value: str | None,
    file_path: str | None,
    *,
    argument_name: str,
) -> str | None:
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"{argument_name} file not found: {path}")
        return path.read_text(encoding="utf-8").strip()
    if inline_value is None:
        return None
    return inline_value.strip()


def render_template(
    template: str | None,
    *,
    question: str,
    source: str,
    split_name: str,
    row_index: int,
    default_value: str,
) -> str:
    if template is None:
        return default_value

    context = {
        "question": question,
        "source": source,
        "split": split_name,
        "row_index": row_index,
    }

    try:
        return template.format_map(context).strip()
    except KeyError as exc:
        available_fields = ", ".join(AVAILABLE_TEMPLATE_FIELDS)
        raise ValueError(
            f"Unknown template field '{exc.args[0]}'. "
            f"Available fields: {available_fields}"
        ) from exc


def detect_backend(requested_backend: str) -> str:
    if requested_backend != "auto":
        return requested_backend
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_dtype(dtype_name: str, backend: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16

    if backend == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if backend == "mps":
        return torch.float16
    return torch.float32


def dtype_to_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return "float32"


def should_use_chat_template(tokenizer: Any, disable_chat_template: bool = False) -> bool:
    if disable_chat_template:
        return False
    return bool(getattr(tokenizer, "chat_template", None))


def format_model_input(
    tokenizer: Any,
    *,
    user_prompt: str,
    system_prompt: str | None,
    use_chat_template: bool,
) -> str:
    if use_chat_template:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if system_prompt:
        return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"
    return user_prompt


def render_prompt_for_row(
    *,
    question: str,
    tokenizer: Any,
    source: str = "",
    split_name: str = "train",
    row_index: int = 0,
    prompt_template: str | None = None,
    system_prompt_template: str | None = None,
    disable_chat_template: bool = False,
) -> dict[str, Any]:
    user_prompt = render_template(
        prompt_template,
        question=safe_text(question),
        source=safe_text(source),
        split_name=split_name,
        row_index=row_index,
        default_value=safe_text(question),
    )
    system_prompt = render_template(
        system_prompt_template,
        question=safe_text(question),
        source=safe_text(source),
        split_name=split_name,
        row_index=row_index,
        default_value="",
    )
    use_chat_template = should_use_chat_template(tokenizer, disable_chat_template)
    formatted_prompt = format_model_input(
        tokenizer,
        user_prompt=user_prompt,
        system_prompt=system_prompt or None,
        use_chat_template=use_chat_template,
    )
    return {
        "question": safe_text(question),
        "source": safe_text(source),
        "user_prompt": user_prompt,
        "system_prompt": system_prompt,
        "formatted_prompt": formatted_prompt,
        "use_chat_template": use_chat_template,
    }


def load_generation_components(
    args: argparse.Namespace,
    backend: str,
) -> tuple[Any, Any, bool, torch.dtype]:
    return load_generation_pipeline(
        model_name=args.model_name,
        backend=backend,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
    )


def load_generation_pipeline(
    *,
    model_name: str,
    backend: str,
    trust_remote_code: bool = False,
    torch_dtype: str = "auto",
) -> tuple[Any, Any, bool, torch.dtype]:
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    resolved_torch_dtype = resolve_torch_dtype(torch_dtype, backend)
    model_class = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
    model = model_class.from_pretrained(
        model_name,
        torch_dtype=resolved_torch_dtype,
        trust_remote_code=trust_remote_code,
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

    model.to(torch.device(backend))
    model.eval()

    return model, tokenizer, config.is_encoder_decoder, resolved_torch_dtype


def build_generation_kwargs(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    return build_generation_kwargs_from_values(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
    )


def build_generation_kwargs_from_values(
    *,
    tokenizer: Any,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    num_beams: int,
    num_return_sequences: int,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if max_input_length <= 0:
        raise ValueError("--max-input-length must be > 0.")
    if max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0.")
    if num_return_sequences <= 0:
        raise ValueError("--num-return-sequences must be > 0.")

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
            raise ValueError("--temperature must be > 0 when --do-sample is set.")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("--top-p must be in (0, 1] when --do-sample is set.")
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    else:
        if num_beams <= 0:
            raise ValueError("--num-beams must be > 0.")
        kwargs["num_beams"] = max(num_beams, num_return_sequences)

    return kwargs


def decode_generated_sequences(
    generated_sequences: torch.Tensor,
    *,
    tokenizer: Any,
    input_width: int,
    is_encoder_decoder: bool,
) -> list[str]:
    if is_encoder_decoder:
        decoded = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
    else:
        decoded = tokenizer.batch_decode(
            generated_sequences[:, input_width:],
            skip_special_tokens=True,
        )
    return [text.strip() for text in decoded]


def regroup_sequences(decoded_texts: list[str], num_return_sequences: int) -> list[list[str]]:
    return [
        decoded_texts[index : index + num_return_sequences]
        for index in range(0, len(decoded_texts), num_return_sequences)
    ]


def generate_from_prompts(
    prompts: list[str],
    *,
    tokenizer: Any,
    model: Any,
    is_encoder_decoder: bool,
    device: torch.device,
    max_input_length: int,
    generation_kwargs: dict[str, Any],
) -> list[list[str]]:
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    tokenized = {name: tensor.to(device) for name, tensor in tokenized.items()}

    with torch.inference_mode():
        generated = model.generate(**tokenized, **generation_kwargs)

    decoded = decode_generated_sequences(
        generated,
        tokenizer=tokenizer,
        input_width=tokenized["input_ids"].shape[1],
        is_encoder_decoder=is_encoder_decoder,
    )
    return regroup_sequences(decoded, generation_kwargs["num_return_sequences"])


def augment_split(
    split_dataset: Dataset,
    *,
    split_name: str,
    args: argparse.Namespace,
    tokenizer: Any,
    model: Any,
    is_encoder_decoder: bool,
    generation_kwargs: dict[str, Any],
    prompt_template: str | None,
    system_prompt_template: str | None,
    use_chat_template: bool,
    device: torch.device,
) -> Dataset:
    if args.question_column not in split_dataset.column_names:
        raise ValueError(
            f"Split '{split_name}' is missing question column '{args.question_column}'."
        )
    if args.prompt_column_name and args.prompt_column_name == args.column_name:
        raise ValueError("--prompt-column-name must be different from --column-name.")

    existing_conflicts = [
        column_name
        for column_name in (args.column_name, args.prompt_column_name)
        if column_name and column_name in split_dataset.column_names
    ]
    if existing_conflicts and not args.overwrite_column:
        conflict_list = ", ".join(existing_conflicts)
        raise ValueError(
            f"Split '{split_name}' already contains: {conflict_list}. "
            "Pass --overwrite-column to replace them."
        )

    if args.max_samples_per_split is not None:
        if args.max_samples_per_split <= 0:
            raise ValueError("--max-samples-per-split must be > 0.")
        limit = min(len(split_dataset), args.max_samples_per_split)
        split_dataset = split_dataset.select(range(limit))

    total_rows = len(split_dataset)
    generated_answers: list[list[str]] = []
    rendered_prompts: list[str] = []

    for start in range(0, total_rows, args.batch_size):
        end = min(start + args.batch_size, total_rows)
        batch = split_dataset[start:end]
        prompts: list[str] = []

        row_count = len(batch[args.question_column])
        for offset in range(row_count):
            row_index = start + offset
            question = safe_text(batch[args.question_column][offset])
            source = ""
            if args.source_column in batch:
                source = safe_text(batch[args.source_column][offset])

            prompt_preview = render_prompt_for_row(
                question=question,
                tokenizer=tokenizer,
                source=source,
                split_name=split_name,
                row_index=row_index,
                prompt_template=prompt_template,
                system_prompt_template=system_prompt_template,
                disable_chat_template=not use_chat_template,
            )
            prompts.append(prompt_preview["formatted_prompt"])
            if args.prompt_column_name:
                rendered_prompts.append(prompt_preview["formatted_prompt"])

        generated_answers.extend(
            generate_from_prompts(
                prompts,
                tokenizer=tokenizer,
                model=model,
                is_encoder_decoder=is_encoder_decoder,
                device=device,
                max_input_length=args.max_input_length,
                generation_kwargs=generation_kwargs,
            )
        )

        print(f"[{split_name}] Generated {end:,}/{total_rows:,} rows")

    updated_split = split_dataset

    removable_columns = []
    for column_name in (args.column_name, args.prompt_column_name):
        if column_name and column_name in updated_split.column_names:
            removable_columns.append(column_name)

    if removable_columns:
        updated_split = updated_split.remove_columns(removable_columns)

    updated_split = updated_split.add_column(args.column_name, generated_answers)
    if args.prompt_column_name:
        updated_split = updated_split.add_column(args.prompt_column_name, rendered_prompts)

    return updated_split


def prepare_output_dir(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    input_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else None

    if input_dir is not None and output_dir.resolve() == input_dir:
        raise ValueError("--output-dir must be different from --dataset-dir.")

    if not output_dir.exists():
        return output_dir
    if args.overwrite_output_dir:
        shutil.rmtree(output_dir)
        return output_dir
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} already exists and is not empty. "
            "Pass --overwrite-output-dir or choose a new --output-dir."
        )
    return output_dir


def augment_dataset_dict(
    dataset_dict: DatasetDict,
    *,
    tokenizer: Any,
    model: Any,
    is_encoder_decoder: bool,
    device: str | torch.device,
    column_name: str,
    prompt_column_name: str | None = None,
    question_column: str = "question",
    source_column: str = "source",
    splits: list[str] | None = None,
    seed: int = 42,
    random_subset_size: int | None = None,
    max_samples_per_split: int | None = None,
    prompt_template: str | None = None,
    system_prompt_template: str | None = None,
    disable_chat_template: bool = False,
    overwrite_column: bool = False,
    batch_size: int = 4,
    max_input_length: int = 1024,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.95,
    num_beams: int = 1,
    num_return_sequences: int = 1,
) -> DatasetDict:
    split_names = resolve_split_names(dataset_dict, splits)
    dataset_dict, _ = maybe_random_subset_dataset(
        dataset_dict,
        split_names=split_names,
        sample_size=random_subset_size,
        seed=seed,
    )
    runtime_args = argparse.Namespace(
        question_column=question_column,
        source_column=source_column,
        column_name=column_name,
        prompt_column_name=prompt_column_name,
        overwrite_column=overwrite_column,
        max_samples_per_split=max_samples_per_split,
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
    )
    generation_kwargs = build_generation_kwargs(runtime_args, tokenizer)
    use_chat_template = should_use_chat_template(tokenizer, disable_chat_template)
    resolved_device = device if isinstance(device, torch.device) else torch.device(device)

    augmented_splits: dict[str, Dataset] = {}
    for split_name, split_dataset in dataset_dict.items():
        if split_name not in split_names:
            augmented_splits[split_name] = split_dataset
            continue

        augmented_splits[split_name] = augment_split(
            split_dataset,
            split_name=split_name,
            args=runtime_args,
            tokenizer=tokenizer,
            model=model,
            is_encoder_decoder=is_encoder_decoder,
            generation_kwargs=generation_kwargs,
            prompt_template=prompt_template,
            system_prompt_template=system_prompt_template,
            use_chat_template=use_chat_template,
            device=resolved_device,
        )

    return DatasetDict(augmented_splits)


def save_run_config(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    backend: str,
    torch_dtype: torch.dtype,
    splits: list[str],
    prompt_template: str | None,
    system_prompt_template: str | None,
    use_chat_template: bool,
    subset_metadata: dict[str, Any],
) -> Path:
    config_path = output_dir / "generation_config.json"
    payload = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "dataset_dir": args.dataset_dir,
        "splits": splits,
        "question_column": args.question_column,
        "source_column": args.source_column,
        "column_name": args.column_name,
        "prompt_column_name": args.prompt_column_name,
        "model_name": args.model_name,
        "device": backend,
        "torch_dtype": dtype_to_name(torch_dtype),
        "max_input_length": args.max_input_length,
        "batch_size": args.batch_size,
        "max_samples_per_split": args.max_samples_per_split,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "seed": args.seed,
        "use_chat_template": use_chat_template,
        "prompt_template": prompt_template,
        "system_prompt": system_prompt_template,
        "random_subset_size": args.random_subset_size,
        "subset_input_total_rows": subset_metadata["input_total_rows"],
        "subset_output_total_rows": subset_metadata["output_total_rows"],
        "subset_input_rows_per_split": subset_metadata["input_rows_per_split"],
        "subset_output_rows_per_split": subset_metadata["output_rows_per_split"],
    }
    with config_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
    return config_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    prompt_template = read_optional_text_argument(
        args.prompt_template,
        args.prompt_template_file,
        argument_name="Prompt template",
    )
    system_prompt_template = read_optional_text_argument(
        args.system_prompt,
        args.system_prompt_file,
        argument_name="System prompt",
    )

    output_dir = prepare_output_dir(args)
    backend = detect_backend(args.device)
    device = torch.device(backend)
    source_dataset = load_source_dataset(args)
    splits_to_augment = resolve_split_names(source_dataset, args.splits)
    source_dataset, subset_metadata = maybe_random_subset_dataset(
        source_dataset,
        split_names=splits_to_augment,
        sample_size=args.random_subset_size,
        seed=args.seed,
    )

    if args.random_subset_size is not None:
        print(
            "Random subset applied: "
            f"{subset_metadata['output_total_rows']:,}/{subset_metadata['input_total_rows']:,} "
            "rows across selected splits"
        )
        for split_name in splits_to_augment:
            input_rows = subset_metadata["input_rows_per_split"][split_name]
            output_rows = subset_metadata["output_rows_per_split"].get(split_name, input_rows)
            print(f"  - {split_name}: {output_rows:,}/{input_rows:,} rows")

    model, tokenizer, is_encoder_decoder, torch_dtype = load_generation_components(args, backend)
    use_chat_template = should_use_chat_template(tokenizer, args.disable_chat_template)
    generation_kwargs = build_generation_kwargs(args, tokenizer)

    augmented_splits: dict[str, Dataset] = {}
    for split_name, split_dataset in source_dataset.items():
        if split_name in splits_to_augment:
            augmented_splits[split_name] = augment_split(
                split_dataset,
                split_name=split_name,
                args=args,
                tokenizer=tokenizer,
                model=model,
                is_encoder_decoder=is_encoder_decoder,
                generation_kwargs=generation_kwargs,
                prompt_template=prompt_template,
                system_prompt_template=system_prompt_template,
                use_chat_template=use_chat_template,
                device=device,
            )
        else:
            augmented_splits[split_name] = split_dataset

    augmented_dataset = DatasetDict(augmented_splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    augmented_dataset.save_to_disk(str(output_dir))
    config_path = save_run_config(
        output_dir,
        args=args,
        backend=backend,
        torch_dtype=torch_dtype,
        splits=splits_to_augment,
        prompt_template=prompt_template,
        system_prompt_template=system_prompt_template,
        use_chat_template=use_chat_template,
        subset_metadata=subset_metadata,
    )

    print(f"Saved augmented dataset to: {output_dir}")
    print(f"Saved generation config to: {config_path}")
    print(f"Model checkpoint: {args.model_name}")
    print(f"Decoder type: {'seq2seq' if is_encoder_decoder else 'causal'}")
    print(f"Backend: {backend}")
    print(f"Chat template enabled: {use_chat_template}")


if __name__ == "__main__":
    main()
