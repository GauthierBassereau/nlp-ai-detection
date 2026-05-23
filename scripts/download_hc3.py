#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

try:
    from scripts.hc3_utils import (
        DEFAULT_DATASET_CONFIG,
        DEFAULT_DATASET_NAME,
        DEFAULT_DATASET_REVISION,
        configure_logging,
        load_hc3_dataset,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from hc3_utils import (
        DEFAULT_DATASET_CONFIG,
        DEFAULT_DATASET_NAME,
        DEFAULT_DATASET_REVISION,
        configure_logging,
        load_hc3_dataset,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download/cache the raw HC3 dataset locally.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default="data/hc3_raw")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--log-file", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_file)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite_output_dir:
            raise FileExistsError(
                f"{output_dir} already exists and is not empty. Pass --overwrite-output-dir to replace it."
            )
        shutil.rmtree(output_dir)

    dataset_dict = load_hc3_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_revision=args.dataset_revision,
        cache_dir=args.cache_dir,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    logging.info("Saved raw dataset to %s", output_dir)
    for split_name, split_dataset in dataset_dict.items():
        logging.info("%s: %s rows", split_name, f"{len(split_dataset):,}")


if __name__ == "__main__":
    main()
