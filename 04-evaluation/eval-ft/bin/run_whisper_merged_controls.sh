#!/usr/bin/env bash
# Score the merged Whisper checkpoint on the two FLEURS monolingual controls.
# This fills the two empty cells for "Whisper large-v3 + LoRA (synthetic-trained)"
# in Table 5 (tab:real).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 ./bin/run_whisper_merged_controls.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(cd ../.. && pwd)"
PYTHON="${PYTHON:-"$([ -x "$ROOT/.venv/bin/python" ] && echo "$ROOT/.venv/bin/python" || echo python3)"}"

if [[ ! -f data/eval/fleurs_ja_jp_test200.jsonl || ! -f data/eval/fleurs_en_us_test200.jsonl ]]; then
  echo "[merged-controls] monolingual manifests missing; run the loaders in data/eval/README.md first" >&2
  exit 1
fi

MODEL="Awshesh12/whisper-large-v3-ja-en-cs-merged"
DEC="--repetition-penalty 1.3 --no-repeat-ngram-size 3"
PY="${PYTHON:-python3}"

echo "[merged-controls] JA control"
$PY eval/run_whisper_hf.py --model-id "$MODEL" $DEC \
  --manifest data/eval/fleurs_ja_jp_test200.jsonl \
  --audio-root data/fleurs \
  --out results/preds/whisper_ft_merged_ja.jsonl

echo "[merged-controls] EN control"
$PY eval/run_whisper_hf.py --model-id "$MODEL" $DEC \
  --manifest data/eval/fleurs_en_us_test200.jsonl \
  --audio-root data/fleurs \
  --out results/preds/whisper_ft_merged_en.jsonl

# If the CS prediction already exists, score all three subsets together.
if [[ -f results/preds/whisper_ft_merged_cs.jsonl ]]; then
  $PY eval/score.py \
    whisper-ft-merged-ja:results/preds/whisper_ft_merged_ja.jsonl \
    whisper-ft-merged-en:results/preds/whisper_ft_merged_en.jsonl \
    whisper-ft-merged-cs:results/preds/whisper_ft_merged_cs.jsonl
fi

echo "[merged-controls] done. Now run 'make benchmark' to update Table 5."
