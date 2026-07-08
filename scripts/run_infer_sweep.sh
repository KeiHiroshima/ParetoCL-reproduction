#!/bin/bash
# Reproduce Figure 3(left): Pareto front approximated by ParetoCL at each
# training stage. Sweeps preference[0] (α_stability) from 0 to 1 and
# evaluates every per-task checkpoint saved by scripts/run_table1.sh.
#
# Run scripts/run_table1.sh first so that seed${SEED}_model_aftertask*.pt
# checkpoints exist under LOG_DIR.

set -e

BUFFER="${BUFFER:-32}"
LOG_DIR="${LOG_DIR:-results/table1_buffer${BUFFER}}"
DATASETS="${DATASETS:-cifar10 cifar100 tinyimagenet}"
SEEDS="${SEEDS:-0 1 2 3 4}"

declare -A NUM_TASKS=(
    ["cifar10"]=5
    ["cifar100"]=10
    ["tinyimagenet"]=10
)

run_exp() {
    local DATASET=$1
    local SETTING=$2   # "online" or "offline"
    local SEED=$3

    local BASE="$LOG_DIR/${DATASET}/${SETTING}"

    for TASK_ID in $(seq 1 "${NUM_TASKS[$DATASET]}"); do
        local CKPT="${BASE}/seed${SEED}_model_aftertask${TASK_ID}.pt"
        if [ ! -f "$CKPT" ]; then
            echo "Missing checkpoint: $CKPT (skipping)"
            continue
        fi
        echo "Sweeping: dataset=$DATASET setting=$SETTING seed=$SEED task=$TASK_ID"
        uv run python infer.py \
            --model-pt "$CKPT" \
            --model-config "${BASE}/seed${SEED}_model_config.json" \
            --dataset "$DATASET" \
            --batch_size 256 \
            --sweep \
            --seed "$SEED" \
            --gpu-id "$SEED" \
            2>&1 | tee "${BASE}/seed${SEED}_aftertask${TASK_ID}_infer.log"
    done
}

for SETTING in online offline; do
    for DATASET in $DATASETS; do
        for SEED in $SEEDS; do
            run_exp "$DATASET" "$SETTING" "$SEED"
        done
    done
done

uv run python visualise.py --path_shared "$LOG_DIR" --seeds $SEEDS --sweep
