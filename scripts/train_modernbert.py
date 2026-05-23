#!/usr/bin/env python3
from __future__ import annotations

try:
    from scripts.train_classifier import main
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from train_classifier import main


if __name__ == "__main__":
    main()
