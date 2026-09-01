#!/bin/bash
# Single-camera (egocentric only) kick-football finetune.
# Checkpoints land in $OUT_DIR/sonic/$RUN_NAME.<...>.<timestamp>/checkpoints/
# train.name must stay "sonic": it selects the trainer module (psi.trainers.<name>).

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

source .venv-psi/bin/activate

NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
ulimit -n 65535
echo "Training with $NPROC_PER_NODE GPUs"

PSI_HOME=${PSI_HOME:-/home/innovation-hacking/luebbet/g1_vla_workspace/external/Psi0/psi_home}
OUT_DIR=${OUT_DIR:-/home/innovation-hacking/luebbet/g1_vla_workspace/models}
RUN_NAME=${RUN_NAME:-psi0_g1_wbc_kick_football_singlecam}

args="
finetune_real_psi0_config \
--seed=292285 \
--exp=$RUN_NAME \
--train.output_dir=$OUT_DIR \
--train.name=sonic \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=16 \
--train.max_checkpoints_to_keep=10 \
--train.gradient_accumulation_steps=8 \
--train.learning_rate=1e-4 \
--train.max_training_steps=32000 \
--train.warmup_ratio=None \
--train.warmup_steps=1000 \
--train.checkpointing_steps=2000 \
--train.validation_steps=1000 \
--train.val_num_batches=20 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=cosine \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--log.report_to=wandb \
--data.root_dir=$PSI_HOME/data/sonic/lerobot \
--data.train_repo_ids=lerobot_g1_kick_football_SINGLE_CAM_1 \
--data.transform.field.stat-path=meta/stats_psi0.json \
--data.transform.field.stat-action-key=action \
--data.transform.field.stat-state-key=states \
--data.transform.field.action_norm_type=bounds \
--data.transform.field.no-use-norm-mask \
--data.transform.field.normalize-state \
--data.transform.model.img-aug \
--data.transform.model.resize.size 240 320 \
--data.transform.model.center_crop.size 240 320 \
--model.model_name_or_path=$PSI_HOME/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
--model.pretrained-action-header-path=$PSI_HOME/cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k \
--model.noise-scheduler=flow \
--model.train-diffusion-steps=1000 \
--model.n_conditions=0 \
--model.action-chunk-size=30 \
--model.action-dim=78 \
--model.action-exec-horizon=30 \
--model.observation-horizon=1 \
--model.odim=43 \
--model.view_feature_dim=2048 \
--model.no-tune-vlm \
--model.no-use_film \
--model.no-combined_temb \
--model.rtc \
--model.max-delay=8
"

find_free_port() {
    start_port=${1:-29500}
    port=${start_port}
    while true; do
        CHECK_PORT=${port} python - <<'PY'
import os,sys,socket
port = int(os.environ.get('CHECK_PORT','0'))
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(('0.0.0.0', port))
    sock.close()
    sys.exit(0)
except OSError:
    sys.exit(1)
PY
        if [ $? -eq 0 ]; then
            echo ${port}
            return 0
        fi
        port=$((port+1))
        if [ ${port} -gt $((start_port+1000)) ]; then
            echo "Failed to find free port after 1000 attempts" >&2
            return 1
        fi
    done
}

MAIN_PORT=$(find_free_port 29500)
if [ -z "${MAIN_PORT}" ]; then
    echo "Could not find free main process port, aborting." >&2
    exit 1
fi

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=${MAIN_PORT} scripts/train.py \
    ${args}
