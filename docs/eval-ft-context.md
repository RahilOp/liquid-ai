# CONTEXT — EN-JP Code-Switching ASR (Liquid AI Hackathon, Track 2)

> **Single source of truth for resuming this project on another machine.**
> Read this top-to-bottom before continuing. Last updated after Day-1 Track-A
> (eval harness + CS-FLEURS loader + first LFM baseline verified end-to-end).

## 1. What this project is

Fine-tune **LiquidAI/LFM2.5-Audio-1.5B-JP** for **English-Japanese code-switching
ASR** — transcribing speech that mixes EN and JA, keeping each language in its
correct script (English in Latin, not force-converted to katakana). Pitch:
**on-device** model that beats cloud ASR on private bilingual transcription.

Deliverables: (1) an eval harness + datasets benchmarking ASR on code-switching;
(2) a fine-tune that beats baselines on a real-speech eval set.

**Scope is ASR only.** TTS / speech-out is out of scope (ignore the Mimi
detokenizer / audio-out path entirely).

## 2. Current status (✅ done / ⏳ next)

- ✅ Repo scaffolded, Python 3.12 venv, all deps installed (`requirements.txt`).
- ✅ **Scoring harness** `eval/score.py` (MER, JA-CER, EN-WER, **Script Accuracy**) + unit tests (`tests/test_score.py`, 6/6 pass).
- ✅ **CS-FLEURS loader** `data/load_csfleurs.py` — pulls the JA-EN slice directly from HF (bypasses the broken `datasets` auto-loader; see §6).
- ✅ **Baseline runner** `eval/run_baseline.py` — runs LFM or Whisper over a manifest.
- ✅ Verified the **fine-tuning toolchain** API (§7) — incl. the finding that **LoRA is NOT built in**.
- ✅ Resumed on the **4× H100 box**: rebuilt venv (uv py3.12.8) + cu126 torch (§3),
  full LFM baseline re-verified on GPU. **Always pin `CUDA_VISIBLE_DEVICES=2`** —
  GPUs 0/1 are teammates'.
- ✅ **Frozen eval benchmark**: `data/eval/csfleurs_jaen_read196.jsonl` (all 196
  JA-EN `read/test` real-speech utts; see §5 + `data/eval/README.md`). Shared,
  identical test set for LFM / Whisper / fine-tuned-LFM.
- ✅ **Benchmarked on n=196** (full table in §5): base LFM-JP, Whisper large-v3
  (faster-whisper + transformers), and a **fine-tuned Whisper-v3** (`Awshesh12/
  whisper-large-v3-ja-en-cs-merged`, via `eval/run_whisper_hf.py`). FT-Whisper
  wins all metrics (MER 36.3 / ScriptAcc 73.0) — the bar the FT-LFM must beat.
- ⏳ **Test the fine-tuned LFM(s)** on the frozen set; run ablations / data
  updates / analysis to improve performance (training data prepared separately,
  outside CS-FLEURS — so the whole slice stays a clean held-out test).
- ⏳ Optional: cloud-API baseline (Google/Azure).
- ⏳ Record the **real human gold eval set** (see `docs/human-eval-protocol.md`).

## 3. Environment (what's installed, what to reinstall)

- **Python**: 3.12.x. On *this* box use **uv's managed CPython 3.12.8**
  (`~/.local/share/uv/python/cpython-3.12.8-*/bin/python3.12`) — the system has
  only 3.9/3.11 and the `miniconda3` here is broken (no `python` binary).
- **GPU**: 4× NVIDIA H100 NVL (95 GB), driver 565.x → max **CUDA 12.7**.
- Venv lives in `.venv/` (git-ignored — **recreate on the new box**).
- Key pinned-ish versions actually installed & working:
  `torch 2.12.0+cu126` (**CUDA 12.6 wheels** — plain `torch` pulls the +cu130 /
  CUDA-13 build which fails on this driver; cu126 is forward-compatible. See
  `requirements.txt` header), `transformers 5.10.2`, `liquid-audio 1.3.0`,
  `datasets 5.0.0`, `faster-whisper 1.2.1`, `jiwer 4.0.0`, `fugashi 1.5.2`,
  `unidic-lite`, `soundfile`, `librosa`, `regex`.
- **`torchcodec` is deliberately NOT installed** — not needed (we bypass the
  `datasets` audio path). Don't add it unless you switch loaders.

### Resume on a new GPU machine
```bash
git clone <your-repo> liquid-ai && cd liquid-ai
# Use a real 3.12 interpreter. On this box, uv has one managed already:
PY312=$(uv python find 3.12)        # or: ~/.local/share/uv/python/cpython-3.12.8-*/bin/python3.12
uv venv --python "$PY312" .venv
uv pip install --python .venv/bin/python -r requirements.txt
# (requirements.txt pins torch/torchaudio to cu126 — correct for driver<=CUDA 12.7)

# sanity: scoring harness
.venv/bin/python tests/test_score.py            # expect 6/6 passed

# re-pull the JA-EN eval data (≈196 real read-speech utts available)
.venv/bin/python data/load_csfleurs.py --method read --split test --limit 196

# re-run the LFM baseline + score
.venv/bin/python eval/run_baseline.py --model lfm \
  --manifest data/csfleurs/read_test/manifest.jsonl --audio-root data/csfleurs \
  --out artifacts/preds/lfm_read.jsonl
.venv/bin/python eval/score.py lfm:artifacts/preds/lfm_read.jsonl
```
Optional: set `HF_TOKEN` env var for faster/higher-rate HF downloads.

## 4. Repo layout
```
eval/score.py            # metrics: MER, JA-CER, EN-WER, ScriptAcc (+ unit-tested)
eval/run_baseline.py     # run ONE model (lfm | whisper) over a manifest -> preds.jsonl
data/load_csfleurs.py    # JA-EN CS-FLEURS loader (HF-direct, torchcodec-free)
data/human_eval/         # real gold-set protocol + manifest template (see docs/)
docs/human-eval-protocol.md   # how to record + transcribe the real eval set
tests/test_score.py      # hand-made unit tests for the scorer
requirements.txt
CONTEXT.md               # <- this file
artifacts/               # run outputs & predictions (git-ignored)
```

### Data formats (the contract everything shares)
- **Manifest / prediction JSONL** line: `{"id", "reference", "hypothesis", ...}`.
  Loader fills `reference` (+ metadata, `hypothesis:""`); baseline fills `hypothesis`.
- `eval/score.py` consumes `{id, reference, hypothesis}` and pools metrics corpus-wide.

## 5. FINAL benchmark — 3 frozen subsets (CS + JA-only + EN-only)

The official eval set for **every** model now has three frozen subsets (see
`data/eval/README.md`). FLEURS `ja_jp`/`en_us` are a **matched control** for
forgetting: CS-FLEURS is align-then-swap over the *same* FLEURS sentences/style/
speakers, so any monolingual regression is attributable to the fine-tune, not a
domain shift.
- `csfleurs_jaen_read196.jsonl` — CS, 196 utts (single speaker `SS`; headline = ScriptAcc)
- `fleurs_ja_jp_test200.jsonl` — JA-only, 200 distinct sentences (headline = JA-CER)
- `fleurs_en_us_test200.jsonl` — EN-only, 200 distinct sentences (headline = EN-WER)

**Eval protocol (STANDARD):** every model uses the **`"Perform ASR."`** system
prompt (the base LFM is strongly prompt-sensitive, and the training data uses it
too — just run without `--system-prompt`; `--repeat-guard` on). Whisper runs via
`eval/run_whisper_hf.py` (transformers/GPU), auto-lang + `repetition_penalty=1.3,
no_repeat_ngram_size=3`.

| Model | CS ScriptAcc↑ | CS MER↓ | JA-CER↓ | JA-WER↓ | EN-WER↓ |
|-------|------|------|------|------|------|
| LFM2.5-Audio-1.5B-JP (base) | 37.4% | 67.3% | 6.6% | 7.6% | 68.5% |
| Whisper large-v3 (base) | 64.6% | 40.9% | 6.0% | 7.9% | **4.2%** |
| LFM + LoRA — data-aug, no encoder LoRA | 43.6% | 54.3% | 8.2% | 8.7% | 80.2% |
| LFM + LoRA — + Conformer encoder LoRA | 61.8% | 54.1% | 9.4% | 10.2% | 73.1% |
| **LFM + LoRA — r32 encoder LoRA + FLEURS mix (best)** | **80.8%** | **27.0%** | **6.4%** | **7.2%** | 37.1% |

**Reading it:** the best LoRA (rank-32 Conformer-encoder LoRA + a FLEURS-balanced
data mix) lifts CS Script Accuracy 37% → 81% and cuts MER 67% → 27% — surpassing
Whisper large-v3 on code-switching — while keeping Japanese intact (CER 6.6 → 6.4,
**no forgetting**) and roughly halving the base model's English error (68.5% →
37.1% WER). The ablation rows isolate the gains: data augmentation (+6 ScriptAcc),
Conformer-encoder LoRA (+24), then rank-32 + FLEURS mix (+43). English-mono WER is
the remaining gap vs Whisper's 4.2%.

**Caveat:** CS-FLEURS switches are *artificially dense* (align-then-swap), so CS
error rates run high; the real human gold set is still the headline differentiator.

## 6. CS-FLEURS gotcha (already solved — don't re-debug)

`load_dataset('byan/cs-fleurs')` fails: the dataset is raw WAVs + per-method
`metadata.jsonl`, and `datasets` 5.0 tries to re-encode decoded audio arrays via
`torchcodec` (ImportError without it + system FFmpeg). **Solution implemented:**
`data/load_csfleurs.py` reads the metadata JSONL directly and pulls only the
`language == "jpn-eng"` WAVs via `hf_hub_download`. Structure:
- `read/test/metadata.jsonl` → **real read speech**, 196 JA-EN rows → our frozen
  eval set (`data/eval/csfleurs_jaen_read196.jsonl`).
- `xtts/{train,test1}` → synthetic TTS (2097 / 650 JA-EN); `mms/test` &
  `xtts/test2` → **0 JA-EN rows**. Not used (training data comes from elsewhere).
- Row fields: `{id, file_name, text, duration, fluency, language, speaker}`.

## 7. Fine-tuning toolchain — VERIFIED facts (this was the #1 risk)

From inspecting installed `liquid_audio 1.3.0`:
- **ASR inference recipe** (what `run_baseline.py` uses):
  ```python
  proc  = LFM2AudioProcessor.from_pretrained("LiquidAI/LFM2.5-Audio-1.5B-JP")
  model = LFM2AudioModel.from_pretrained(...).eval()
  chat = ChatState(proc)
  chat.new_turn("system"); chat.add_text("Perform ASR."); chat.end_turn()
  chat.new_turn("user");   chat.add_audio(wave_1xT_float, sr); chat.end_turn()
  chat.new_turn("assistant")
  toks = [int(t) for t in model.generate_sequential(**chat, max_new_tokens=256)
          if t.numel()==1]            # text path; token 7 = <|im_end|> ends it
  text = proc.text.decode(toks, skip_special_tokens=True)
  ```
  `add_audio` resamples to 16 kHz internally. Output is already cased + punctuated.
- **Training** = `liquid_audio.trainer.Trainer` — **FULL fine-tune** via `accelerate`,
  bf16, defaults: lr 3e-5, max_steps 1000, batch_size 16.
- ⚠️ **NO LoRA / PEFT in the package.** The brief assumed LoRA-via-trainer; it
  doesn't exist. Options on the new GPU:
  (a) **Full FT** of the 1.5B backbone — fits comfortably on one 80 GB H100; or
  (b) add **PEFT/LoRA manually** by wrapping `model.lfm` with `peft.get_peft_model`
      before constructing `Trainer` (freeze the FastConformer encoder either way).
- **Training-data contract** (`liquid_audio.data.preprocess.preprocess_dataset`):
  input is `Iterable[list[ChatMessage]]`; each example a chat:
  `system: TextSegment("Perform ASR.")` → `user: AudioSegment(audio=<wav bytes>)`
  → `assistant: TextSegment(<reference transcript>)`. An `LFM2AudioChatMapper`
  packs each chat to tensors (`text, audio_in, audio_in_lens, audio_out,
  modality_flag, supervision_mask`); `preprocess_dataset` writes a HF dataset via
  `save_to_disk`. `LFM2DataLoader(dataset_path)` (`load_from_disk`) feeds `Trainer`.
  Loss is supervised on the assistant text only (`supervision_mask`).
  > TODO when building data: confirm `AudioSegment.audio` byte format the mapper
  > expects (see `liquid_audio/data/mapper.py`) before generating at scale, and
  > smoke-test the full data→~50-steps→eval loop on ~10 examples first.

## 8. Methodology reference (metrics + conventions)

- **MER** over a mixed token stream (1 token per JA char, 1 per EN word).
- **JA-CER** over the Japanese-only character stream; **EN-WER** over the Latin
  word stream. **Script Accuracy** = of EN reference words, fraction the model
  rendered in Latin (vs katakana/other/dropped) — the headline metric.
- Identical normalization (NFKC, lowercase, punctuation→space, English fillers
  removed) applied to references AND every hypothesis. Details in `eval/score.py` docstring.
- **Bootstrap CIs** and **paired comparisons** via `--bootstrap` / `--paired`.
- **Switch-point analysis** (`--switch-report`) stratifies token accuracy by
  distance to EN/JP boundaries and ScriptAcc by switch type / embedded-span length.
- A **script audit** (`--audit`) lists every English reference word and its
  hypothesis rendering, separating dropped words from transliterations.
- Annotation/script-policy conventions for the human gold set: `docs/human-eval-protocol.md`.

## 9. Strategy / decisions (agreed)

- Validate the gap on public data FIRST (done — gap is real). Demo > metrics for
  judging: build a side-by-side live transcription demo (base/Whisper/cloud
  katakana-forcing vs ours). Cut the academic 2-annotator/IAA protocol.
- **Eval vs train split**: training data is **prepared separately** (NOT from
  CS-FLEURS). CS-FLEURS is therefore used **only as a frozen held-out benchmark**
  (`data/eval/csfleurs_jaen_read196.jsonl`) — same fixed 196-utt set scores base
  LFM, Whisper large-v3, and the fine-tuned LFM. No leakage to worry about.
- CS-FLEURS is the *credibility/secondary* benchmark (single speaker, artificial
  switching). The **real human gold set is the headline** for any "beats cloud"
  claim — don't stake the win on CS-FLEURS alone.
- Fallback narrative if FT doesn't beat baselines: on-device **script-policy /
  prompt-steering control** ("cloud can't do this privately").
