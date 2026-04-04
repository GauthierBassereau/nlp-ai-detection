# NLP AI Text Detection

Minimal Hugging Face workflow for:

1. Downloading the HC3 dataset locally.
2. Fine-tuning a ModernBERT classifier for human-vs-AI text detection.


## Train ModernBERT

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
    --output-dir outputs/roberta-base_test \
    --max-train-samples 2000 \
    --max-validation-samples 500 \
    --max-test-samples 500 \
    --per-device-train-batch-size 4 \
    --per-device-eval-batch-size 8 \
    --num-train-epochs 1 \
    --model-choice roberta-base
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