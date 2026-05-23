from __future__ import annotations

import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


DEFAULT_DATASET_NAME = "Hello-SimpleAI/HC3"
DEFAULT_DATASET_CONFIG = "all"
DEFAULT_DATASET_REVISION = "refs/convert/parquet"

DEFAULT_QUESTION_COLUMN = "question"
DEFAULT_HUMAN_ANSWERS_COLUMN = "human_answers"
DEFAULT_SOURCE_COLUMN = "source"
DEFAULT_SELECTED_HUMAN_COLUMN = "selected_human_answer"
DEFAULT_SELECTED_HUMAN_INDEX_COLUMN = "selected_human_answer_index"


def configure_logging(log_file: str | Path | None = None, *, rank: int = 0) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file and rank == 0:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def ensure_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        texts = [safe_text(item) for item in value]
        return [text for text in texts if text]
    text = safe_text(value)
    return [text] if text else []


def pick_deterministic_text(
    values: Any,
    *,
    seed: int,
    split_name: str,
    row_index: int,
    salt: str,
) -> tuple[str, int | None]:
    texts = ensure_text_list(values)
    if not texts:
        return "", None
    rng = random.Random(f"{seed}:{split_name}:{row_index}:{salt}")
    selected_index = rng.randrange(len(texts))
    return texts[selected_index], selected_index


def normalize_dataset_object(dataset_obj: Dataset | DatasetDict) -> DatasetDict:
    if isinstance(dataset_obj, DatasetDict):
        return dataset_obj
    return DatasetDict({"train": dataset_obj})


def load_hc3_dataset(
    *,
    dataset_dir: str | Path | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str | None = DEFAULT_DATASET_CONFIG,
    dataset_revision: str | None = DEFAULT_DATASET_REVISION,
    cache_dir: str | Path | None = None,
) -> DatasetDict:
    if dataset_dir:
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        return normalize_dataset_object(load_from_disk(str(dataset_path)))

    load_kwargs: dict[str, Any] = {
        "path": dataset_name,
        "cache_dir": str(cache_dir) if cache_dir else None,
    }
    if dataset_config:
        load_kwargs["data_dir"] = dataset_config
    if dataset_revision:
        load_kwargs["revision"] = dataset_revision

    load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
    return normalize_dataset_object(load_dataset(**load_kwargs))


def resolve_split_names(dataset_dict: DatasetDict, requested_splits: list[str] | None) -> list[str]:
    available_splits = list(dataset_dict.keys())
    if not requested_splits:
        return available_splits

    missing = sorted(set(requested_splits).difference(available_splits))
    if missing:
        raise ValueError(
            f"Unknown split(s): {', '.join(missing)}. "
            f"Available splits: {', '.join(available_splits)}"
        )
    return requested_splits


def sanitize_identifier(value: str, *, fallback: str = "value", max_length: int = 96) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    if not normalized:
        normalized = fallback
    if normalized[0].isdigit():
        normalized = f"{fallback}_{normalized}"
    return normalized[:max_length].rstrip("_") or fallback


def build_generation_column_name(
    *,
    model_name: str,
    prompt_name: str,
    model_alias: str | None = None,
    prefix: str = "ai",
) -> str:
    model_part = sanitize_identifier(model_alias or model_name, fallback="model")
    prompt_part = sanitize_identifier(prompt_name, fallback="prompt")
    prefix_part = sanitize_identifier(prefix, fallback="ai")
    return f"{prefix_part}_{model_part}_{prompt_part}"


def load_prompt_specs(prompt_file: str | Path) -> list[dict[str, Any]]:
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if isinstance(payload, dict) and "prompts" in payload:
        raw_prompts = payload["prompts"]
    else:
        raw_prompts = payload

    specs: list[dict[str, Any]] = []
    if isinstance(raw_prompts, dict):
        for name, value in raw_prompts.items():
            if not isinstance(value, dict):
                raise ValueError(f"Prompt '{name}' must be an object.")
            specs.append({"name": name, **value})
    elif isinstance(raw_prompts, list):
        for value in raw_prompts:
            if not isinstance(value, dict):
                raise ValueError("Every prompt entry must be an object.")
            specs.append(dict(value))
    else:
        raise ValueError("Prompt file must contain an object or a list of prompt objects.")

    names_seen: set[str] = set()
    for spec in specs:
        name = safe_text(spec.get("name"))
        if not name:
            raise ValueError("Every prompt must define a non-empty name.")
        if name in names_seen:
            raise ValueError(f"Duplicate prompt name: {name}")
        if not safe_text(spec.get("user")):
            raise ValueError(f"Prompt '{name}' must define a non-empty 'user' template.")
        spec["name"] = name
        names_seen.add(name)

    return specs


class StrictFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        available = ", ".join(sorted(self.keys()))
        raise KeyError(f"Unknown template field '{key}'. Available fields: {available}")


def render_template(template: str, context: dict[str, Any]) -> str:
    normalized_context = StrictFormatDict({key: safe_text(value) for key, value in context.items()})
    try:
        return template.format_map(normalized_context).strip()
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
