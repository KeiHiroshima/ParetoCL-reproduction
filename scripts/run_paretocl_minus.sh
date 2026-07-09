#!/bin/bash
# Reproduce Figure 3(right): ParetoCL vs ParetoCL-- incremental-accuracy
# comparison on Seq-CIFAR100, online setting.
#
# ParetoCL-- uses a static preference α=(0.5, 0.5) at inference instead of
# the dynamic entropy-based selection (Algorithm 2).

set -e

BUFFER="${BUFFER:-1000}"
LOG_DIR="${LOG_DIR:-results/buffer${BUFFER}}"
SEEDS="${SEEDS:-0 1 2}"
read -ra SEEDS <<< "$SEEDS"

DATASET="cifar100"
SETTING="online"
NUM_TASKS=10
BASE="$LOG_DIR/${DATASET}/${SETTING}"

run_exp() {
    local SEED=$1

    for TASK_ID in $(seq 1 "$NUM_TASKS"); do
        local CKPT="${BASE}/seed${SEED}_model_aftertask${TASK_ID}.pt"
        local OUT="${BASE}/seed${SEED}_inference_fixed_aftertask${TASK_ID}.json"
        if [ ! -f "$CKPT" ]; then
            echo "Missing checkpoint: $CKPT (skipping)"
            continue
        fi
        if [ -f "$OUT" ]; then
            echo "Already exists, skipping (delete it to regenerate): $OUT"
            continue
        fi
        echo "ParetoCL--: dataset=$DATASET setting=$SETTING seed=$SEED task=$TASK_ID"
        uv run python infer.py \
            --model-pt "$CKPT" \
            --model-config "${BASE}/seed${SEED}_model_config.json" \
            --dataset "$DATASET" \
            --batch_size 256 \
            --preference 0.5 0.5 \
            --seed "$SEED" \
            --gpu-id "$SEED" \
            2>&1 | tee "${BASE}/seed${SEED}_aftertask${TASK_ID}_paretocl_minus.log"
    done
}

run_exp "${SEEDS[0]}" &\
run_exp "${SEEDS[1]}" &\
run_exp "${SEEDS[2]}"
wait

uv run python visualise.py --path_shared "$LOG_DIR" --seeds "${SEEDS[@]}" --fig3_right
