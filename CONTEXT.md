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
- ✅ **First baseline verified**: LFM2.5-Audio-1.5B-JP zero-shot on 20 CS-FLEURS read JA-EN utterances (table in §5). Pipeline works end-to-end.
- ✅ Verified the **fine-tuning toolchain** API (§7) — incl. the finding that **LoRA is NOT built in**.
- ⏳ Run remaining baselines: **Whisper large-v3** and a **cloud API** (Google/Azure).
- ⏳ Build **training data** (splicing trick) and **fine-tune** (needs a dedicated GPU — the reason for the machine transfer).
- ⏳ Record the **real human gold eval set** (see `docs/human-eval-protocol.md`).

## 3. Environment (what's installed, what to reinstall)

- **Python**: 3.12.11 (miniforge). **GPU**: trained/run on NVIDIA H100.
- Venv lives in `.venv/` (git-ignored — **recreate on the new box**).
- Key pinned-ish versions actually installed & working:
  `torch 2.12.0` (CUDA 13 wheels), `transformers 5.10.2`, `liquid-audio 1.3.0`,
  `datasets 5.0.0`, `faster-whisper 1.2.1`, `jiwer 4.0.0`, `fugashi 1.5.2`,
  `unidic-lite`, `soundfile`, `librosa`, `regex`.
- **`torchcodec` is deliberately NOT installed** — not needed (we bypass the
  `datasets` audio path). Don't add it unless you switch loaders.

### Resume on a new GPU machine
```bash
git clone <your-repo> liquid-ai && cd liquid-ai
python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip wheel
.venv/bin/pip install -r requirements.txt

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

## 5. First baseline result (n=20 smoke test — NOT the final benchmark)

LFM2.5-Audio-1.5B-JP, zero-shot, CS-FLEURS `read/test` JA-EN, system prompt `Perform ASR.`:

| Model | N | MER↓ | JA-CER↓ | EN-WER↓ | **ScriptAcc↑** |
|-------|---|------|---------|---------|------------|
| lfm-jp (zero-shot) | 20 | 72.8% | 92.7% | 89.1% | **16.7%** |

Script breakdown of 192 English reference words: **latin 32 / katakana 107 /
other-JP 50 / dropped 3**. i.e. the base model **forces ~56% of English words
into katakana** and only keeps 17% in Latin — the exact gap the fine-tune targets.

**Caveats (be honest about these):** n=20 only; CS-FLEURS switches are
*artificially dense* (align-then-swap) and harder/less natural than real speech,
so error rates run high; JA-CER is character-level and brutal when content words
are misheard. One example (`jpn_1928`) is transcribed near-perfectly, confirming
the metric isn't pathologically penalizing. Real spontaneous gold set is the
differentiator — CS-FLEURS is the credibility leg only.

## 6. CS-FLEURS gotcha (already solved — don't re-debug)

`load_dataset('byan/cs-fleurs')` fails: the dataset is raw WAVs + per-method
`metadata.jsonl`, and `datasets` 5.0 tries to re-encode decoded audio arrays via
`torchcodec` (ImportError without it + system FFmpeg). **Solution implemented:**
`data/load_csfleurs.py` reads the metadata JSONL directly and pulls only the
`language == "jpn-eng"` WAVs via `hf_hub_download`. Structure:
- `read/test/metadata.jsonl` → **real read speech** (best eval slice; ~196 JA-EN rows)
- `mms/test`, `xtts/{train,test1,test2}` → synthetic (bulk of training data)
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
  dropped) applied to references AND every hypothesis. Details in `eval/score.py` docstring.
- Annotation/script-policy conventions for the human gold set: `docs/human-eval-protocol.md`.

## 9. Strategy / decisions (agreed)

- Validate the gap on public data FIRST (done — gap is real). Demo > metrics for
  judging: build a side-by-side live transcription demo (base/Whisper/cloud
  katakana-forcing vs ours). Cut the academic 2-annotator/IAA protocol.
- Fallback narrative if FT doesn't beat baselines: on-device **script-policy /
  prompt-steering control** ("cloud can't do this privately").
- Splicing-trick caveat: concatenated monolingual clips leave acoustic seams /
  unnatural prosody at switch points — gains may be modest; evaluate on real audio only.
