#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./submit_multi_node.sh [-s SUFFIX] NUM_NODES NUM_GPUS_PER_NODE GPU_TYPE path/to/script.py [args...]
#
# Optional environment overrides:
#   REPO=/path/to/repo
#   ENV_SETUP_CMD='source .venv/bin/activate'
#   SBATCH_PARTITION=gpu
#   SBATCH_MEM=200G
#   SBATCH_TIME=1-23:59:00
#   CPUS_PER_GPU=8

usage() {
  echo "Usage: ./submit_multi_node.sh [-s SUFFIX] NUM_NODES NUM_GPUS_PER_NODE GPU_TYPE path/to/script.py [args...]"
}

SUFFIX=""
if [[ "${1:-}" == "-s" ]] || [[ "${1:-}" == "--suffix" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "Error: --suffix requires an argument." >&2
    usage >&2
    exit 1
  fi
  SUFFIX="${2}"
  shift 2
fi

if [[ "$#" -lt 4 ]]; then
  usage >&2
  exit 1
fi

NUM_NODES="${1}"
NUM_GPUS_PER_NODE="${2}"
GPU_TYPE="${3}"
shift 3
PYFILE="${1}"
shift

REPO="${REPO:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
REPO="$(cd "$REPO" && pwd)"

if [[ "$PYFILE" != /* ]]; then
  PYFILE="$(pwd)/$PYFILE"
fi
PYFILE="$(cd "$(dirname "$PYFILE")" && pwd)/$(basename "$PYFILE")"

if [[ "$PYFILE" != "$REPO/"* ]]; then
  echo "Error: script must be inside repo: $REPO" >&2
  echo "Got: $PYFILE" >&2
  exit 1
fi

rel="${PYFILE#$REPO/}"
rel="${rel%.py}"
mod="${rel//\//.}"
name="$(basename "$PYFILE" .py)"

mkdir -p "$REPO/logs"

JOB_SUFFIX="${SUFFIX:+-$SUFFIX}"
JOB_NAME="${name}-${NUM_NODES}n${NUM_GPUS_PER_NODE}g${JOB_SUFFIX}"
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
CPUS_PER_TASK=$((NUM_GPUS_PER_NODE * CPUS_PER_GPU))
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu}"
SBATCH_MEM="${SBATCH_MEM:-200G}"
SBATCH_TIME="${SBATCH_TIME:-1-23:59:00}"
ENV_SETUP_CMD="${ENV_SETUP_CMD:-}"

args=""
for arg in "$@"; do
  args+=" $(printf '%q' "$arg")"
done

sbatch \
  --job-name="${JOB_NAME}" \
  --output="${REPO}/logs/%x.out" \
  --partition="${SBATCH_PARTITION}" \
  --gres="gpu:${GPU_TYPE}:${NUM_GPUS_PER_NODE}" \
  --nodes="${NUM_NODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${SBATCH_MEM}" \
  --time="${SBATCH_TIME}" \
  --export=ALL \
  --wrap "bash -lc '
    set -euo pipefail
    source ~/.bashrc || true
    shopt -s expand_aliases

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false

    if [[ -n \"${ENV_SETUP_CMD}\" ]]; then
      eval \"${ENV_SETUP_CMD}\"
    fi

    nodes=(\$(scontrol show hostnames \$SLURM_JOB_NODELIST))
    head_node=\${nodes[0]}
    export MASTER_ADDR=\$head_node
    export MASTER_PORT=\$(shuf -i 29500-65000 -n 1)
    export RDZV_ID=\${SLURM_JOB_ID:-\$RANDOM}

    echo
    echo \"=============================\"
    echo \"Job: ${JOB_NAME}\"
    echo \"Module: ${mod}\"
    echo \"Config: ${NUM_NODES} node(s), ${NUM_GPUS_PER_NODE} GPU(s)/node, ${TOTAL_GPUS} total GPU(s), type ${GPU_TYPE}\"
    echo \"Nodes: \${nodes[*]}\"
    echo \"Master: \$MASTER_ADDR:\$MASTER_PORT\"
    echo \"Date: \$(date)\"
    echo \"Repo: ${REPO}\"
    echo \"=============================\"
    echo

    cd \"${REPO}\"

    srun bash -lc \"torchrun \\
      --nnodes=${NUM_NODES} \\
      --nproc_per_node=${NUM_GPUS_PER_NODE} \\
      --rdzv_id=\$RDZV_ID \\
      --rdzv_backend=c10d \\
      --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \\
      -m ${mod}${args}\"
  '"
