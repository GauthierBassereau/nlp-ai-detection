# NLP AI Text Detection

Minimal Hugging Face workflow for:

1. Downloading the HC3 dataset locally.
2. Augmenting HC3 with answers from a newer LLM such as Llama 3.
3. Fine-tuning a ModernBERT classifier for human-vs-AI text detection.

## Augment HC3 With New LLM Answers

The augmentation script adds a new answer column to the raw HC3 dataset so you can:

- swap the Hugging Face generation model with `--model-name`,
- run with or without a conditioning prompt,
- save the rendered prompt in a debug column if you want to inspect what was sent,
- keep generating new columns for different models or prompt variants.

Example without extra prompting:

```bash
python scripts/augment_hc3.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --column-name llama31_answers \
    --output-dir data/hc3_llama31 \
    --batch-size 2 \
    --max-new-tokens 192
```

Example with a conditioning prompt:

```bash
python scripts/augment_hc3.py \
    --dataset-dir data/hc3_llama31 \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --column-name llama31_conditioned_answers \
    --prompt-template "Answer the following question like a real student. Question: {question}" \
    --prompt-column-name llama31_conditioned_prompt \
    --output-dir data/hc3_llama31_conditioned \
    --batch-size 2 \
    --max-new-tokens 192
```

Prompt templates support these fields:

- `{question}`
- `{source}`
- `{split}`
- `{row_index}`

Each generated column is stored as a list of answers per HC3 row, so it stays compatible with the training pipeline.


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
    --dataset-dir data/hc3_llama31_conditioned \
    --ai-answers-columns chatgpt_answers llama31_answers llama31_conditioned_answers \
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
