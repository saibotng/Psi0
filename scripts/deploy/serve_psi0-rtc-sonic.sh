#!/bin/bash

source .venv-psi/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1   # log markers must reach tee/pipe-pane immediately
echo "Training with $nprocs GPUs, which is/are $CUDA_VISIBLE_DEVICES"

python src/psi/deploy/psi_serve_rtc_token-sonic.py \
    --host 0.0.0.0 \
    --port ${PSI0_PORT:-8014} \
    --action_exec_horizon 30 \
    --policy psi \
    --rtc \
    --ctrl-hz=${CTRL_HZ:-30} \
    --run-dir=${CHECKPOINT_DIR} \
    --ckpt-step=${CHECKPOINT_STEP}
