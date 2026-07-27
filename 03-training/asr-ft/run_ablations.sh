#!/usr/bin/env bash
# Ablation study: fine-tune whisper-large-v3 (LoRA) under different corpus /
# hyperparameter configs, evaluate each on REAL CS-FLEURS and the synthetic
# (dissertation) test set. Run on mactrn01 from /awshesh/code-switching.
#   bash scripts/03-training/asr-ft/run_ablations.sh 0
set -u
cd /awshesh/code-switching
export HF_HOME=/awshesh/code-switching/hf_cache CUDA_VISIBLE_DEVICES="${1:-0}"
T=scripts/03-training/asr-ft/train_whisper.py
RH=scripts/04-evaluation/eval-ft/eval/run_whisper_hf.py
S=scripts/04-evaluation/eval-ft/eval/score.py
DEC="--repetition-penalty 1.3 --no-repeat-ngram-size 3"
BASE=openai/whisper-large-v3
CS_M=data/eval-real/csfleurs/read_test/manifest.jsonl; CS_R=data/eval-real/csfleurs
SY_M=data/eval-diss-synth/manifest.jsonl; SY_R=.
PRED=data/eval-preds/ablation; mkdir -p "$PRED" models/ablation

CLEAN="data/synth/edge_tts/manifest.jsonl data/synth/kokoro/manifest.jsonl data/synth/melo/manifest.jsonl"
FULL="$CLEAN data/augmented/manifest.jsonl"
SINGLE="data/synth/edge_tts/manifest.jsonl data/ablation/aug_edge.jsonl"

evalset () {  # $1=adapter dir  $2=name
  .venv/bin/python $RH --model-id $BASE --lora-adapter "$1" $DEC --manifest $CS_M --audio-root $CS_R --out "$PRED/$2_csfleurs.jsonl" >/dev/null 2>&1
  .venv/bin/python $RH --model-id $BASE --lora-adapter "$1" $DEC --manifest $SY_M --audio-root $SY_R --out "$PRED/$2_synth.jsonl"     >/dev/null 2>&1
}

train_eval () {  # $1=name  $2=rank  $3...=manifests
  local name=$1 rank=$2; shift 2
  echo "[$(date +%T)] TRAIN $name (r=$rank)"
  .venv/bin/python $T --model-id $BASE --manifests "$@" --out "models/ablation/$name" \
     --epochs 3 --batch-size 16 --lr 1e-4 --lora-r "$rank" >"data/logs/abl_${name}.log" 2>&1
  echo "[$(date +%T)] EVAL  $name"
  evalset "models/ablation/$name/final" "$name"
}

# A0 full r8 already trained — just evaluate it on both sets under this harness
echo "[$(date +%T)] EVAL full_r8 (already trained)"
evalset models/trained/whisper-large-v3/final full_r8

train_eval clean_only  8  $CLEAN
train_eval single_edge 8  $SINGLE
train_eval rank16      16 $FULL
train_eval rank32      32 $FULL

echo "########## ABLATIONS — REAL CS-FLEURS (n=196) ##########"
.venv/bin/python $S \
  full_r8:$PRED/full_r8_csfleurs.jsonl clean_only:$PRED/clean_only_csfleurs.jsonl \
  single_edge:$PRED/single_edge_csfleurs.jsonl rank16:$PRED/rank16_csfleurs.jsonl \
  rank32:$PRED/rank32_csfleurs.jsonl --json $PRED/scores_csfleurs.json
echo; echo "########## ABLATIONS — SYNTHETIC in-domain (n=30) ##########"
.venv/bin/python $S \
  full_r8:$PRED/full_r8_synth.jsonl clean_only:$PRED/clean_only_synth.jsonl \
  single_edge:$PRED/single_edge_synth.jsonl rank16:$PRED/rank16_synth.jsonl \
  rank32:$PRED/rank32_synth.jsonl --json $PRED/scores_synth.json
echo ABLATIONS_DONE
