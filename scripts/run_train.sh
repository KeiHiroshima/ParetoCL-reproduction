#!/bin/bash

set -e

BUFFER="${BUFFER:-1000}"
LR="${LR:-0.05}"
LOG_DIR="${LOG_DIR:-results/buffer${BUFFER}}"
DATASETS="${DATASETS:-cifar10 cifar100 tinyimagenet}"
SEEDS="${SEEDS:-0 1 2}"
read -ra SEEDS <<< "$SEEDS"

run_exp() {
    local DATASET=$1
    local SETTING=$2   # "online" or "offline"
    local SEED=$3

    local ONLINE_FLAG=""
    if [ "$SETTING" = "online" ]; then
        ONLINE_FLAG="--online"
    fi

    local BASE="$LOG_DIR/${DATASET}/${SETTING}"
    mkdir -p "$BASE"

    echo "Running: dataset=$DATASET setting=$SETTING seed=$SEED -> $BASE/seed${SEED}.log"
    uv run python train.py \
        --dataset "$DATASET" \
        --buffer_size "$BUFFER" \
        --lr "$LR" \
        --seed "$SEED" \
        --gpu_id "$SEED" \
        $ONLINE_FLAG \
        --log_dir "$LOG_DIR" \
        2>&1 | tee "$BASE/seed${SEED}.log"
}


for SETTING in online offline; do
    for DATASET in $DATASETS; do
        run_exp "$DATASET" "$SETTING" "${SEEDS[0]}" &\
        run_exp "$DATASET" "$SETTING" "${SEEDS[1]}" &\
        run_exp "$DATASET" "$SETTING" "${SEEDS[2]}"
    done
done

uv run python visualise.py --path_shared "$LOG_DIR" --seeds "${SEEDS[@]}"
