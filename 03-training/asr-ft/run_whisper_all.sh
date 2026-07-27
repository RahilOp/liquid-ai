#!/usr/bin/env bash
# Train Whisper tiny/base/small/large-v3 (LoRA) on the synthetic CS corpus.
# Run on mactrn01 from /awshesh/code-switching. Adapters land in models/trained/.
#   bash scripts/03-training/asr-ft/run_whisper_all.sh 0
set -u
cd /awshesh/code-switching
GPU="${1:-0}"
export HF_HOME=/awshesh/code-switching/hf_cache CUDA_VISIBLE_DEVICES="$GPU"
T=scripts/03-training/asr-ft/train_whisper.py
MAN="data/synth/edge_tts/manifest.jsonl data/synth/kokoro/manifest.jsonl data/synth/melo/manifest.jsonl data/augmented/manifest.jsonl"
LOG=data/logs

# size -> per-device batch (H100 94GB; large-v3 smaller batch)
declare -A BS=( [tiny]=64 [base]=48 [small]=32 [large-v3]=16 )

for size in tiny base small large-v3; do
  echo "[$(date +%T)] === training whisper-$size (batch ${BS[$size]}) ==="
  .venv/bin/python "$T" \
     --model-id "openai/whisper-$size" \
     --manifests $MAN \
     --out "models/trained/whisper-$size" \
     --epochs 3 --batch-size "${BS[$size]}" --lr 1e-4 \
     > "$LOG/train_whisper_$size.log" 2>&1
  echo "[$(date +%T)] whisper-$size done -> $(ls models/trained/whisper-$size/final/*.safetensors 2>/dev/null | wc -l) adapter file(s)"
done
echo "[$(date +%T)] TRAIN_ALL_DONE"
