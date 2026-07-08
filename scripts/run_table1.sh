#!/bin/bash
# Reproduce Table 1 (ParetoCL row): Seq-CIFAR10, Seq-CIFAR100, Seq-TinyImageNet
# Online (1 epoch/task) and Offline (5 epochs/task), 5 seeds each.
# Paper defaults: buffer_size=32, lr=0.05.
#
# For Table 2 (buffer-size ablation on Seq-CIFAR100, online setting),
# override BUFFER, e.g.:
#   BUFFER=600  DATASETS=cifar100 SEEDS="0 1 2" ./scripts/run_table1.sh
#   BUFFER=1000 DATASETS=cifar100 SEEDS="0 1 2" ./scripts/run_table1.sh
#   BUFFER=1400 DATASETS=cifar100 SEEDS="0 1 2" ./scripts/run_table1.sh

set -e

BUFFER="${BUFFER:-32}"
LR="${LR:-0.05}"
LOG_DIR="${LOG_DIR:-results/table1_buffer${BUFFER}}"
DATASETS="${DATASETS:-cifar10 cifar100 tinyimagenet}"
SEEDS="${SEEDS:-0 1 2 3 4}"

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
        for SEED in $SEEDS; do
            run_exp "$DATASET" "$SETTING" "$SEED"
        done
    done
done

uv run python visualise.py --path_shared "$LOG_DIR" --seeds $SEEDS
