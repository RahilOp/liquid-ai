#!/usr/bin/env bash
# Fine-tune Whisper on CS corpus + monolingual JA/EN replay (anti-forgetting),
# then run the forgetting comparison: base vs CS-only vs CS+mono on FLEURS mono
# controls and CS-FLEURS. Run on mactrn01 from /awshesh/code-switching.
#   bash scripts/03-training/asr-ft/run_cs_mono.sh 1
set -u
cd /awshesh/code-switching
export HF_HOME=/awshesh/code-switching/hf_cache CUDA_VISIBLE_DEVICES="${1:-1}"
T=scripts/03-training/asr-ft/train_whisper.py
RH=scripts/04-evaluation/eval-ft/eval/run_whisper_hf.py
S=scripts/04-evaluation/eval-ft/eval/score.py
DEC="--repetition-penalty 1.3 --no-repeat-ngram-size 3"
CLEAN="data/synth/edge_tts/manifest.jsonl data/synth/kokoro/manifest.jsonl data/synth/melo/manifest.jsonl"
CS_MONO="$CLEAN data/augmented/manifest.jsonl data/mono/mono_train.jsonl"
declare -A BS=( [tiny]=64 [base]=48 [small]=32 [large-v3]=16 )

CS_M=data/eval-real/csfleurs/read_test/manifest.jsonl; CS_R=data/eval-real/csfleurs
JA_M=data/eval-mono/ja_jp_test/manifest.jsonl; EN_M=data/eval-mono/en_us_test/manifest.jsonl; MONO_R=data/eval-mono
P=data/eval-preds/csmono; mkdir -p "$P" models/trained_mono

echo "[$(date +%T)] waiting for mono data ..."
for i in $(seq 1 180); do grep -q MONO_DL_DONE data/logs/mono_dl.log 2>/dev/null && break; sleep 15; done
.venv/bin/python scripts/02-preprocessing/asr-ft/build_mono_train.py

run () {  # $1=model-id  $2=adapterOrEmpty  $3=manifest  $4=root  $5=outfile
  local ad=""; [ -n "$2" ] && ad="--lora-adapter $2"
  .venv/bin/python $RH --model-id "$1" $ad $DEC --manifest "$3" --audio-root "$4" --out "$5" >/dev/null 2>&1
}

for size in tiny base small large-v3; do
  B="openai/whisper-$size"
  echo "[$(date +%T)] TRAIN whisper-$size on CS+mono"
  .venv/bin/python $T --model-id $B --manifests $CS_MONO --out "models/trained_mono/whisper-$size" \
     --epochs 3 --batch-size "${BS[$size]}" --lr 1e-4 > "data/logs/csmono_$size.log" 2>&1
  M="models/trained_mono/whisper-$size/final"
  C="models/trained/whisper-$size/final"          # CS-only (previous run)
  echo "[$(date +%T)] EVAL whisper-$size (base / cs-only / cs+mono)"
  # CS-FLEURS: cs+mono only (base + cs-only already scored earlier)
  run $B "$M" $CS_M $CS_R "$P/csmono_${size}_csfleurs.jsonl"
  # mono forgetting controls: all three variants
  run $B ""   $JA_M $MONO_R "$P/base_${size}_ja.jsonl";     run $B ""   $EN_M $MONO_R "$P/base_${size}_en.jsonl"
  run $B "$C" $JA_M $MONO_R "$P/csonly_${size}_ja.jsonl";   run $B "$C" $EN_M $MONO_R "$P/csonly_${size}_en.jsonl"
  run $B "$M" $JA_M $MONO_R "$P/csmono_${size}_ja.jsonl";   run $B "$M" $EN_M $MONO_R "$P/csmono_${size}_en.jsonl"
done

for size in tiny base small large-v3; do
  echo; echo "########## whisper-$size — JA-only (FLEURS ja_jp test, JA-CER matters) ##########"
  .venv/bin/python $S base:$P/base_${size}_ja.jsonl csonly:$P/csonly_${size}_ja.jsonl csmono:$P/csmono_${size}_ja.jsonl
  echo "########## whisper-$size — EN-only (FLEURS en_us test, EN-WER matters) ##########"
  .venv/bin/python $S base:$P/base_${size}_en.jsonl csonly:$P/csonly_${size}_en.jsonl csmono:$P/csmono_${size}_en.jsonl
  echo "########## whisper-$size — CS-FLEURS (cs+mono) ##########"
  .venv/bin/python $S csmono:$P/csmono_${size}_csfleurs.jsonl
done
echo CS_MONO_DONE
