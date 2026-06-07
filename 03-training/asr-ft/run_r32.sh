#!/bin/bash
set -e
cd /awshesh/lfm2.5/awshesh
source .venv_train/bin/activate

HF_TOKEN="${HF_TOKEN:?export HF_TOKEN before running}"
WANDB_KEY="${WANDB_KEY:?export WANDB_KEY before running}"

echo "[1/4] Downloading FLEURS (EN 500 + JA 500 + CS 500 target + EN/JA val)..."
python -u download_fleurs_500.py \
  --data-root /awshesh/lfm2.5/awshesh \
  --hf-token $HF_TOKEN \
  --n 500 \
  > /awshesh/lfm2.5/awshesh/download_fleurs_extra.log 2>&1

echo "[2/4] Building balanced JSONL (all CS synthetic + 300 EN/JA synthetic)..."
python -u combine_balanced.py \
  --data-root /awshesh/lfm2.5/awshesh \
  --n-synthetic 300 \
  >> /awshesh/lfm2.5/awshesh/download_fleurs_extra.log 2>&1

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/awshesh/lfm2.5/awshesh/.venv_train/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NO_PROXY='*'
export no_proxy='*'

echo "[3/4] Preprocessing..."
python -u preprocess_asr_data.py \
  --train_jsonl data/training/balanced_train.jsonl \
  --eval_jsonl data/training/balanced_eval.jsonl \
  --output_dir data/preprocessed_balanced_r32 \
  > /awshesh/lfm2.5/awshesh/preprocess_balanced_r32.log 2>&1

echo "[4/4] Training rank-32 conformer LoRA..."
python -u train_lora.py \
  --data data/preprocessed_balanced_r32/train \
  --val_data data/preprocessed_balanced_r32/eval \
  --output_dir checkpoints/lfm_lora_balanced_r32 \
  --max_steps 2000 --warmup_steps 100 \
  --lr 3e-5 --encoder_lr 6e-6 \
  --lora_r 8 --encoder_lora_r 32 \
  --batch_size 8 --save_interval 200 \
  --val_interval 100 --log_interval 10 \
  --wandb_key $WANDB_KEY \
  --wandb_project lfm25-asr-lora-jp \
  > /awshesh/lfm2.5/awshesh/train_balanced_r32.log 2>&1

echo "All done."
