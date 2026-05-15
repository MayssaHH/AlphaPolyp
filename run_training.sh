#!/usr/bin/env bash
# run_training.sh  —  Launch AlphaPolyp training on RunPod.
#
# Prerequisites: setup_runpod.sh must have been run successfully.
#
# Usage:
#   bash run_training.sh               # full run (train.py)
#   bash run_training.sh --cache ram   # enable RAM cache for faster epochs

set -euo pipefail

WORKSPACE=/workspace
REPO_DIR=${WORKSPACE}/AlphaPolyp
DATA_DIR=${WORKSPACE}/data
LOG_DIR=${WORKSPACE}/logs      # on the Network Volume → persists

mkdir -p "${LOG_DIR}"

cd "${REPO_DIR}"

# Symlink logs into repo so scripts find them at ./logs
if [ ! -L "${REPO_DIR}/logs" ]; then
    ln -s "${LOG_DIR}" "${REPO_DIR}/logs"
fi

# Copy regression stats from a previous run if available
# (skip if this is the first run)
if [ -f "${LOG_DIR}/regression_stats.pkl" ] && \
   [ ! -f "${REPO_DIR}/regression_stats.pkl" ]; then
    cp "${LOG_DIR}/regression_stats.pkl" "${REPO_DIR}/regression_stats.pkl"
    echo "Restored regression_stats.pkl from previous run."
fi

echo "============================================================"
echo "  GPU info"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null || echo "  (no nvidia-smi)"

echo ""
echo "============================================================"
echo "  Starting training"
echo "  Data : ${DATA_DIR}"
echo "  Logs : ${LOG_DIR}"
echo "============================================================"
echo ""

# Pipe stdout+stderr to both terminal and a persistent log file
LOGFILE="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

python -u train.py \
    --root "${DATA_DIR}" \
    "$@" \
    2>&1 | tee "${LOGFILE}"

# Copy regression_stats.pkl to the persistent log dir
cp -f "${REPO_DIR}/regression_stats.pkl" "${LOG_DIR}/regression_stats.pkl" 2>/dev/null || true

echo ""
echo "Full training log: ${LOGFILE}"
echo "Reports + checkpoints: ${LOG_DIR}/"
