# NLP AI Text Detection

Minimal Hugging Face workflow for:

1. Downloading the HC3 dataset locally.
2. Fine-tuning BERT-base, RoBERTa-base, or ModernBERT-base for human-vs-AI text detection.

## Environment setup

Create and activate a virtual environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install PyTorch separately so you can choose the right build for your machine:

- macOS / CPU / Apple Silicon MPS: install the default PyTorch wheel.
- CUDA machine / HPC: install the CUDA-enabled PyTorch build that matches the cluster driver and CUDA runtime.

Then install the rest of the project dependencies:

```bash
pip install -r requirements.txt
```

## Download HC3

This downloads HC3 into the default Hugging Face cache directory.

```bash
python scripts/download_hc3.py
```

It uses the auto-converted Parquet revision of HC3, which avoids the legacy dataset-script error in current `datasets` versions.

## Train A Classifier

The training script:

- uses Hugging Face `datasets` for loading and preprocessing,
- uses `Trainer`/`TrainingArguments` for fine-tuning,
- auto-detects `cuda`, `mps`, or `cpu`,
- enables CUDA mixed precision automatically when available,
- logs to W&B by default,
- writes train/validation/test metrics,
- saves test-set predictions and misclassifications as JSONL.

Example:

```bash
python scripts/train_modernbert.py \
  --model-choice modernbert-base \
  --output-dir outputs/modernbert-hc3 \
  --run-name modernbert-hc3 \
  --max-length 256 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 16 \
  --num-train-epochs 3
```

Before your first tracked run:

```bash
wandb login
```

If you want to disable tracking for a run:

```bash
python scripts/train_modernbert.py \
  --output-dir outputs/no-tracking \
  --report-to none
```

Useful options:

- `--model-choice bert-base|roberta-base|modernbert-base`: choose one of the three baseline architectures.
- `--model-name some/custom-checkpoint`: override the preset choices with any HF checkpoint.
- `--text-mode answer`: classify from the answer text only.
- `--text-mode question_answer`: include both question and answer in the classifier input.
- `--run-name my-experiment`: name the W&B run explicitly.
- `--report-to wandb|tensorboard|none`: choose the tracker.
- `--train-subsample-ratio 0.25`: train on 25% of the training split for dataset-size experiments.
- `--max-train-samples 2000`: cap the train split directly.
- `--gradient-checkpointing`: reduce memory use for longer sequences.
- `--save-splits-dir data/processed/hc3_flat`: save the flattened train/validation/test dataset for reuse.

Common runs:

```bash
python scripts/train_modernbert.py --model-choice bert-base --output-dir outputs/bert-base
python scripts/train_modernbert.py --model-choice roberta-base --output-dir outputs/roberta-base
python scripts/train_modernbert.py --model-choice modernbert-base --output-dir outputs/modernbert-base
```

## Outputs

Training writes artifacts under the chosen `--output-dir`, including:

- model checkpoints,
- tokenizer files,
- `train_results.json`,
- `all_results.json`,
- `test_results.json`,
- `test_predictions.jsonl`,
- `test_misclassified.jsonl`.
