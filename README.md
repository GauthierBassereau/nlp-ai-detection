# NLP AI Text Detection

Script-based workflow for HC3 human-vs-AI detection experiments.

The intended dataset flow is:

1. Create a reproducible `train`/`eval` subset from `Hello-SimpleAI/HC3`.
2. Append generated answer columns with one or more prompt styles and one generation model.
3. Train one encoder classifier per generated answer column.

## 1. Create A Subset

This creates a saved Hugging Face `DatasetDict` with `train` and `eval` splits. It also freezes one random human answer per row in `selected_human_answer`, while keeping the original HC3 columns.

```bash
python -m scripts.create_hc3_subset \
  --train-size 5000 \
  --eval-size 1000 \
  --seed 42 \
  --output-dir data/hc3_subset_5k_1k
```

## 2. Generate New AI Answers

Prompts live in [configs/hc3_generation_prompts.json](configs/hc3_generation_prompts.json). The default file includes:

- `basic`
- `human_imitator`
- `random_human_example`
- `actual_human_reference`

Single-process example:

```bash
python -m scripts.augment_hc3 \
  --dataset-dir data/hc3_subset_5k_1k \
  --output-dir data/hc3_augmented \
  --model-name meta-llama/Llama-3.2-3B-Instruct \
  --model-alias llama32_3b \
  --batch-size 8
```

Running the command again with the same `--output-dir` appends new columns to the existing dataset. Existing generated columns are never removed unless you pass `--overwrite-columns`.

HPC launch through Slurm:

```bash
ENV_SETUP_CMD='source .venv/bin/activate' \
./submit_multi_node.sh -s gen 1 4 h200-141g scripts/augment_hc3.py \
  --dataset-dir data/hc3_subset_5k_1k \
  --output-dir data/hc3_augmented \
  --model-name meta-llama/Llama-3.2-3B-Instruct \
  --model-alias llama32_3b \
  --batch-size 8
```

Under `torchrun`, generation is data-parallel: each rank generates a different row shard, then rank 0 merges the shards into the saved dataset.

## 3. Train Classifiers

The trainer auto-detects generated columns beginning with `ai_` and runs one classifier training job per column. Each run writes model artifacts, Trainer metrics, source-level accuracy, and failed eval examples locally.

```bash
python -m scripts.train_classifier \
  --dataset-dir data/hc3_augmented \
  --model-name answerdotai/ModernBERT-base \
  --output-dir outputs/modernbert_hc3_augmented \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 32 \
  --num-train-epochs 3 \
  --wandb-project nlp-ai-detection
```

HPC launch:

```bash
ENV_SETUP_CMD='source .venv/bin/activate' \
./submit_multi_node.sh -s train 1 4 h200-141g scripts/train_classifier.py \
  --dataset-dir data/hc3_augmented \
  --model-name answerdotai/ModernBERT-base \
  --output-dir outputs/modernbert_hc3_augmented \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 32 \
  --num-train-epochs 3 \
  --wandb-project nlp-ai-detection
```

To reduce answer-length shortcuts, train on deterministic random answer windows:

```bash
ENV_SETUP_CMD='source .venv/bin/activate' \
./submit_multi_node.sh -s train-w50 1 4 h200-141g scripts/train_classifier.py \
  --dataset-dir data/hc3_augmented \
  --model-name answerdotai/ModernBERT-base \
  --output-dir outputs/modernbert_hc3_augmented_w50 \
  --answer-window-words 50 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 32 \
  --num-train-epochs 3 \
  --wandb-project nlp-ai-detection
```

For each generated answer column, local eval logs are written under:

```text
outputs/<run>/<ai_column>/eval_logs/
```

Important files:

- `metrics.json`
- `classification_report.json`
- `confusion_matrix.json`
- `source_metrics.csv`
- `source_metrics.png`
- `source_accuracy.csv`
- `bad_classifications.jsonl`

`source_metrics.csv` and `source_metrics.png` include per-source accuracy, AI precision, AI recall, AI F1, and the corresponding human-class metrics.

## Sweep Training Set Size

Run the same classifier setup on increasing numbers of train rows. Each train row gives one human example and one AI example for the selected AI column.

```bash
ENV_SETUP_CMD='source .venv/bin/activate' \
./submit_multi_node.sh -s sweep-roberta-qwen 1 4 h200-141g scripts/sweep_train_sizes.py \
  --dataset-dir data/hc3_augmented \
  --ai-answer-columns ai_qwen25_3b_actual_human_reference \
  --model-choice roberta-base \
  --output-dir outputs/roberta_qwen25_3b_actual_size_sweep \
  --train-row-sizes 100 500 1000 2500 5000 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 32 \
  --num-train-epochs 3 \
  --wandb-project nlp-ai-detection
```

If you are using answer windows in the rest of the experiments, add the same flag here:

```bash
  --answer-window-words 50
```

Outputs are separated by size:

```text
outputs/roberta_qwen25_3b_actual_size_sweep/train_rows_100/ai_qwen25_3b_actual_human_reference/
outputs/roberta_qwen25_3b_actual_size_sweep/train_rows_500/ai_qwen25_3b_actual_human_reference/
...
```

The sweep also writes `all_size_metrics.json`, `size_metrics.csv`, and `size_metrics.png` in the sweep output directory.

## Evaluate Checkpoint Transfer

Evaluate one trained classifier checkpoint on a different generated-answer column without retraining:

```bash
python -m scripts.evaluate_classifier_checkpoint \
  --dataset-dir data/hc3_augmented \
  --checkpoint-dir outputs/modernbert_hc3_augmented/ai_qwen25_3b_actual_human_reference \
  --ai-answer-columns ai_llama32_3b_v3_actual_human_reference \
  --output-dir outputs/transfer_eval/qwen25_actual_on_llama32_v3_actual
```

HPC launch:

```bash
ENV_SETUP_CMD='source .venv/bin/activate' \
./submit_multi_node.sh -s transfer-eval 1 1 h200-141g scripts/evaluate_classifier_checkpoint.py \
  --dataset-dir data/hc3_augmented \
  --checkpoint-dir outputs/modernbert_hc3_augmented/ai_qwen25_3b_actual_human_reference \
  --ai-answer-columns ai_llama32_3b_v3_actual_human_reference \
  --output-dir outputs/transfer_eval/qwen25_actual_on_llama32_v3_actual
```

If the checkpoint was trained with answer windows, pass the same setting during transfer eval:

```bash
  --answer-window-words 50
```

The transfer eval writes the same local files as training under `outputs/transfer_eval/<run>/<ai_column>/eval_logs/`.

## Plot Answer Lengths

Plot one combined length-distribution graph for all answer columns:

```bash
python -m scripts.plot_answer_lengths \
  --dataset-dir data/hc3_augmented \
  --output-file outputs/answer_length_distribution.png
```

Use model-token counts instead of simple word counts:

```bash
python -m scripts.plot_answer_lengths \
  --dataset-dir data/hc3_augmented \
  --length-unit hf_tokens \
  --tokenizer-name answerdotai/ModernBERT-base \
  --output-file outputs/answer_length_distribution_tokens.png
```

## Export Random Examples

Write a plain text file with sampled questions, one human answer, and each AI answer column:

```bash
python -m scripts.export_random_examples \
  --dataset-dir data/hc3_augmented \
  --split eval \
  --num-examples 10 \
  --output-file outputs/random_augmented_examples.txt
```

Inspect the exact classifier text and token IDs for paired human/AI examples:

```bash
python -m scripts.export_classifier_inputs \
  --dataset-dir data/hc3_augmented \
  --split eval \
  --ai-answer-columns ai_llama32_3b_v2_basic \
  --model-name answerdotai/ModernBERT-base \
  --answer-window-words 50 \
  --num-rows 10 \
  --output-file outputs/classifier_input_examples.txt
```

## Notes

- `scripts/train_modernbert.py` is kept as a compatibility wrapper around `scripts/train_classifier.py`.
- Pass `--ai-answer-columns col_a col_b` to train only selected generated columns.
- Pass `--prompt-names basic human_imitator` to generate only selected prompt types.
- Pass `--report-to none` to disable W&B while keeping all local logs.
