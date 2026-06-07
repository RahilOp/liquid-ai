# Final evaluation benchmark (frozen)

The **final, fixed eval set** for every model — base **LFM2.5-Audio-1.5B-JP**,
**Whisper large-v3**, the **fine-tuned Whisper**, and **all future fine-tuned
LFMs**. Three frozen subsets, scored independently so we see both the target task
*and* monolingual forgetting:

| Subset | File | N | Purpose |
|--------|------|---|---------|
| **Code-switching** | `csfleurs_jaen_read196.jsonl` | 196 | the target task (EN-JP CS ASR) |
| **Japanese-only** | `fleurs_ja_jp_test200.jsonl` | 200 | JA forgetting control (JA-CER) |
| **English-only** | `fleurs_en_us_test200.jsonl` | 200 | EN forgetting control (EN-WER) |

## Why FLEURS monolingual is the *correct* control
CS-FLEURS is built by **align-then-swap over the same FLEURS/FLoRes sentences**,
in the **same read-speech style and speaker pool**. So FLEURS `ja_jp` / `en_us`
are a **matched control**: same sentence domain, audio conditions, and speakers as
the CS set — the *only* thing that changes is the code-switching itself. Any
monolingual regression is therefore attributable to the fine-tune, not a domain
shift between datasets. (You can see the match directly: e.g. the "Alloys are
basically a mixture of two or more metals…" sentence appears in EN-only, JA-only,
*and* the CS set.)

## Subset details
- **CS** (`csfleurs_jaen_read196.jsonl`): entire JA-EN slice of CS-FLEURS
  `read/test`, 39.5 min, ~59% EN / 41% JA chars, ~7.5 switches/utt.
  ⚠️ single speaker (`SS`). Headline metric: **ScriptAcc** (EN kept in Latin).
- **JA-only** (`fleurs_ja_jp_test200.jsonl`): 200 distinct FLEURS `ja_jp` test
  sentences, ~45 min, both genders / multiple speakers. Metric: **JA-CER**.
- **EN-only** (`fleurs_en_us_test200.jsonl`): 200 distinct FLEURS `en_us` test
  sentences, ~35 min, both genders / multiple speakers. Metric: **EN-WER**.

Manifests are committed (ids + gold refs + metadata); WAVs are git-ignored and
regenerated deterministically from HF.

## Regenerate audio (any box)
```bash
.venv/bin/python data/load_csfleurs.py --method read --split test --limit 196   # -> data/csfleurs/
.venv/bin/python data/load_fleurs.py --config ja_jp --split test --limit 200    # -> data/fleurs/
.venv/bin/python data/load_fleurs.py --config en_us --split test --limit 200    # -> data/fleurs/
```

## Eval protocol (STANDARD: one prompt for every model)
- **All LFM models** (base + every fine-tune) use the system prompt **`"Perform
  ASR."`** — this is the default of `eval/run_baseline.py`, and the training data
  now uses it too, so just run **without** `--system-prompt`. No language forced;
  `--repeat-guard` on (stops greedy repetition loops).
  - ⚠️ **Legacy exception:** the **step_1000 LoRA** was trained on `"Transcribe the
    audio."`, so its benchmark row was produced with `--system-prompt "Transcribe
    the audio."`. **Do NOT re-run it with `"Perform ASR."`** (off-distribution for
    it). All *future* fine-tunes are trained on `"Perform ASR."` → use the default.
- **Whisper** (base + ft) runs via `eval/run_whisper_hf.py` (transformers/GPU;
  faster-whisper's GPU path is blocked by a CTranslate2/driver PTX mismatch),
  auto-language, `--repetition-penalty 1.3 --no-repeat-ngram-size 3`.
- Pin `CUDA_VISIBLE_DEVICES=2` on this shared box.

```bash
# per subset (audio-root = data/csfleurs for CS, data/fleurs for mono).
# Standard prompt "Perform ASR." is the default — no --system-prompt needed:
CUDA_VISIBLE_DEVICES=2 .venv/bin/python eval/run_baseline.py --model lfm \
  [--lora-adapter <ckpt>] \
  --manifest <subset.jsonl> --audio-root <root> --out artifacts/preds/<name>.jsonl

# score one subset (corpus-pooled MER / JA-CER / EN-WER / ScriptAcc)
.venv/bin/python eval/score.py base:<preds>.jsonl ft:<preds>.jsonl
```

## Leaderboard UI (Streamlit)
View the leaderboard and evaluate a new LoRA checkpoint from a dropdown:
```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/streamlit run eval/leaderboard_app.py
```
- Top: the leaderboard (best-in-column highlighted), seeded with the 4 reference
  models from `data/eval/leaderboard.json`.
- Bottom: pick any LoRA checkpoint under `/awshesh/lfm2.5/awshesh/checkpoints/`
  (auto-discovered by `adapter_config.json`), evaluates it on all 3 subsets with
  the standard `"Perform ASR."` prompt (override for legacy ckpts), and adds the
  row. "Quick test (20 utts)" for a fast sanity check. Backend: `eval/leaderboard.py`.
