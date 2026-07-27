#!/bin/bash
#SBATCH --job-name=eq-evaluate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eq_evaluate_%j.out
#
# EQ_evaluate on the predictions.parquet written by eq_llm_predict_openai.sbatch.
#
#   mkdir -p logs    # SLURM opens --output BEFORE this script runs.
#   sbatch scripts/eq_evaluate.sbatch                    # uses OUTPUT_DIR below
#   sbatch scripts/eq_evaluate.sbatch /some/other/out    # or override it

set -euo pipefail

export REPO_DIR="$HOME/EveryQuery_text"
OUTPUT_DIR="${1:-/users/gbk2114/eq-neurips-rebuttle/llm_baseline_out_gpt}"

cd "$REPO_DIR"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} started=$(date -Is) out=$OUTPUT_DIR"

uv run EQ_evaluate \
    predictions_parquet="$OUTPUT_DIR/predictions.parquet" \
    metrics_parquet="$OUTPUT_DIR/metrics.parquet"

echo "finished=$(date -Is)"
echo "metrics: $OUTPUT_DIR/metrics.parquet"
