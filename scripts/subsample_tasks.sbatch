#!/bin/bash
#SBATCH --job-name=eq-subsample
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/eq_subsample_%j.out
#
# Subsample TaskQuery parquets down to PER_CLASS positives + PER_CLASS negatives per
# (query, duration_days) task.
#
#   mkdir -p logs    # SLURM opens --output BEFORE this script runs, so it must already
#                    # exist, and it resolves relative to the SUBMISSION directory.
#   sbatch scripts/subsample_tasks.sbatch
#
# Reads the whole input into memory (polars, single pass) and writes ONE parquet file.
# Feed OUTPUT_FILE's parent directory to EQ_llm_predict as tasks_dir.

set -euo pipefail

# ─── EDIT ME ────────────────────────────────────────────────────────────────────────────

# Repo checkout. The job may start in any directory, so we cd here explicitly.
export REPO_DIR="$HOME/EveryQuery_text"

# Input: either a single TaskQuerySchema parquet or a directory of shards (read recursively).
export INPUT="/users/gbk2114/nwicu-eq/eval-new-sampler/eval/held_out"

# Output: a single parquet FILE (its parent directory is created below).
export OUTPUT_FILE="/users/gbk2114/nwicu-eq/eval_tasks_500/eval/held_out/tasks.parquet"

# Second output: same sampling, cut down to N_TASKS random (query, duration_days) tasks.
# At the same SEED this is a strict row-subset of OUTPUT_FILE. Set N_TASKS=0 to skip it.
#export OUTPUT_FILE_SMALL="/users/gbk2114/mimic-eq/eval_tasks_500_20tasks/eval/held_out/tasks.parquet"
export N_TASKS=20

# Rows kept per label per task — 250 + 250 = 500 patients per task.
export PER_CLASS=250

# Drop tasks that can't field this many in EITHER class. 1 drops the degenerate
# single-class tasks (zero positives or zero negatives — no AUROC, nothing to learn);
# 0 keeps every task however thin. Raise it for a real floor per class.
export MIN_PER_CLASS=1

# Change to re-draw a different random sample.
export SEED=0

# ─── END EDIT ME ────────────────────────────────────────────────────────────────────────

cd "$REPO_DIR"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is)"
echo "in=$INPUT  out=$OUTPUT_FILE  per_class=$PER_CLASS  min_per_class=$MIN_PER_CLASS  seed=$SEED"

mkdir -p "$(dirname "$OUTPUT_FILE")"

uv run scripts/subsample_tasks.py \
    "$INPUT" \
    "$OUTPUT_FILE" \
    --per-class "$PER_CLASS" \
    --min-per-class "$MIN_PER_CLASS" \
    --seed "$SEED"

if [[ "$N_TASKS" -gt 0 ]]; then
    echo "--- ${N_TASKS}-task subset ---"
    mkdir -p "$(dirname "$OUTPUT_FILE_SMALL")"
    uv run scripts/subsample_tasks.py \
        "$INPUT" \
        "$OUTPUT_FILE_SMALL" \
        --per-class "$PER_CLASS" \
        --min-per-class "$MIN_PER_CLASS" \
        --seed "$SEED" \
        --n-tasks "$N_TASKS"
fi

echo "finished=$(date -Is)"
