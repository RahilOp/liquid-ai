#!/usr/bin/env bash
# Regenerate Table 5 from checked-in prediction JSONLs in results/preds/.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(cd ../.. && pwd)"
PYTHON="${PYTHON:-"$([ -x "$ROOT/.venv/bin/python" ] && echo "$ROOT/.venv/bin/python" || echo python3)"}"

declare -A rows=(
  ["LFM2.5-Audio-1.5B-JP (zero-shot)"]="lfm_base"
  ["Whisper large-v3 (zero-shot)"]="whisper_base"
  ["Whisper large-v3 + LoRA (synthetic-trained)"]="whisper_ft_merged"
  ["LFM + LoRA, augmentation only"]="lfm_lora_aug"
  ["LFM + LoRA, + Conformer encoder LoRA"]="lfm_lora_encoder"
  ["LFM + LoRA, r=32 encoder + FLEURS mix"]="lfm_lora_r32"
)

mkdir -p results/scores

run_subset() {
  local subset=$1 manifest=$2 out=$3
  shift 3
  # Build scorer argument list from the row stems.
  local args=()
  for name in "${!rows[@]}"; do
    local stem=${rows[$name]}
    local pred="results/preds/${stem}_${subset}.jsonl"
    if [[ -f "$pred" ]]; then
      args+=("$name:$pred")
    fi
  done
  if [[ ${#args[@]} -eq 0 ]]; then
    echo "[run_table5] no predictions found for subset=$subset; skipping" >&2
    return 0
  fi
  echo "[run_table5] scoring $subset (${#args[@]} systems)"
  "${PYTHON:-python3}" eval/score.py "${args[@]}" --json "$out"
}

run_subset cs data/eval/csfleurs_jaen_read196.jsonl results/scores/table5_cs.json
run_subset ja data/eval/fleurs_ja_jp_test200.jsonl results/scores/table5_ja.json
run_subset en data/eval/fleurs_en_us_test200.jsonl results/scores/table5_en.json

# Merge the three subset score JSONs into a single Table 5 object.
"${PYTHON:-python3}" - "$@" <<'PY'
import json, os, sys

def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)

cs = load("results/scores/table5_cs.json")
ja = load("results/scores/table5_ja.json")
en = load("results/scores/table5_en.json")

rows = []
for c in cs:
    name = c["model"]
    j = next((x for x in ja if x["model"] == name), None)
    e = next((x for x in en if x["model"] == name), None)
    rows.append({
        "system": name,
        "ScriptAcc": c.get("ScriptAcc"),
        "MER": c.get("MER"),
        "JA_CER": j.get("JA_CER") if j else None,
        "JA_WER": j.get("JA_WER") if j else None,
        "EN_WER": e.get("EN_WER") if e else None,
    })

with open("results/table5.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

# Pretty-print a table similar to the paper's Table 5.
header = f"{'System':45s} {'ScriptAcc':>9s} {'MER':>7s} {'JA-CER':>7s} {'JA-WER':>7s} {'EN-WER':>7s}"
print(header)
print("-" * len(header))
for r in rows:
    fmt = lambda v: f"{v*100:6.1f}%" if v is not None else "   —  "
    print(f"{r['system']:45s} {fmt(r['ScriptAcc'])} {fmt(r['MER'])} {fmt(r['JA_CER'])} {fmt(r['JA_WER'])} {fmt(r['EN_WER'])}")
PY

printf '\nTable 5 written to results/table5.json and results/table5.txt\n'
# Save the printed table as well.
"${PYTHON:-python3}" - <<'PY' > results/table5.txt
import json
rows = json.load(open("results/table5.json", encoding="utf-8"))
header = f"{'System':45s} {'ScriptAcc':>9s} {'MER':>7s} {'JA-CER':>7s} {'JA-WER':>7s} {'EN-WER':>7s}"
out = [header, "-" * len(header)]
for r in rows:
    fmt = lambda v: f"{v*100:6.1f}%" if v is not None else "   —  "
    out.append(f"{r['system']:45s} {fmt(r['ScriptAcc'])} {fmt(r['MER'])} {fmt(r['JA_CER'])} {fmt(r['JA_WER'])} {fmt(r['EN_WER'])}")
print("\n".join(out))
PY
