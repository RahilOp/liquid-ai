#!/usr/bin/env bash
# Wait for run_whisper_all.sh to finish, then evaluate all 4 trained Whisper
# models (base vs LoRA-fine-tuned) on both CS eval sets and score.
# Launch in background; produces the final comparison tables autonomously.
set -u
cd /awshesh/code-switching
export HF_HOME=/awshesh/code-switching/hf_cache CUDA_VISIBLE_DEVICES="${1:-0}"
RH=scripts/04-evaluation/eval-ft/eval/run_whisper_hf.py
S=scripts/04-evaluation/eval-ft/eval/score.py
DEC="--repetition-penalty 1.3 --no-repeat-ngram-size 3"
CS_M=data/eval-real/csfleurs/read_test/manifest.jsonl; CS_R=data/eval-real/csfleurs
AR_M=data/eval-synth/csartificial_manifest.jsonl; AR_R=.
OUT=data/eval-preds/trained; mkdir -p "$OUT"

echo "[$(date +%T)] waiting for TRAIN_ALL_DONE ..."
for i in $(seq 1 1080); do grep -q TRAIN_ALL_DONE data/logs/train_all.log 2>/dev/null && break; sleep 20; done
echo "[$(date +%T)] training done — evaluating trained models"

for size in tiny base small large-v3; do
  B="openai/whisper-$size"; A="models/trained/whisper-$size/final"
  echo "[$(date +%T)] eval whisper-$size (base + trained)"
  .venv/bin/python $RH --model-id $B $DEC --manifest $CS_M --audio-root $CS_R --out "$OUT/base_${size}_csfleurs.jsonl"  >/dev/null 2>&1
  .venv/bin/python $RH --model-id $B $DEC --manifest $AR_M --audio-root $AR_R --out "$OUT/base_${size}_artificial.jsonl" >/dev/null 2>&1
  .venv/bin/python $RH --model-id $B --lora-adapter $A $DEC --manifest $CS_M --audio-root $CS_R --out "$OUT/ft_${size}_csfleurs.jsonl"  >/dev/null 2>&1
  .venv/bin/python $RH --model-id $B --lora-adapter $A $DEC --manifest $AR_M --audio-root $AR_R --out "$OUT/ft_${size}_artificial.jsonl" >/dev/null 2>&1
done

echo "########## CS-FLEURS (real JA-EN, n=196) ##########"
.venv/bin/python $S \
  base-tiny:$OUT/base_tiny_csfleurs.jsonl ft-tiny:$OUT/ft_tiny_csfleurs.jsonl \
  base-base:$OUT/base_base_csfleurs.jsonl ft-base:$OUT/ft_base_csfleurs.jsonl \
  base-small:$OUT/base_small_csfleurs.jsonl ft-small:$OUT/ft_small_csfleurs.jsonl \
  base-lgv3:$OUT/base_large-v3_csfleurs.jsonl ft-lgv3:$OUT/ft_large-v3_csfleurs.jsonl \
  --json $OUT/scores_csfleurs.json
echo; echo "########## ARTIFICIAL held-out synthetic (n=200) ##########"
.venv/bin/python $S \
  base-tiny:$OUT/base_tiny_artificial.jsonl ft-tiny:$OUT/ft_tiny_artificial.jsonl \
  base-base:$OUT/base_base_artificial.jsonl ft-base:$OUT/ft_base_artificial.jsonl \
  base-small:$OUT/base_small_artificial.jsonl ft-small:$OUT/ft_small_artificial.jsonl \
  base-lgv3:$OUT/base_large-v3_artificial.jsonl ft-lgv3:$OUT/ft_large-v3_artificial.jsonl \
  --json $OUT/scores_artificial.json
echo TRAINED_EVAL_DONE
