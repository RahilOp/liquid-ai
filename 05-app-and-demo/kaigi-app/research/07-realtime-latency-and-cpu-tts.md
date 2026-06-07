# 07 — Real-time latency & the CPU-TTS bottleneck (improvement brief)

> Compiled 2026-06-06. Scope: **which parts of the running app are / are not real-time, why, and how to fix the
> non-real-time part.** Grounded first in this repo's own measured numbers (`documentation/system-resources.md`,
> `src/csmeeting/live/`), then in vendor + research sources. Treat vendor RTF/latency claims as targets to validate on
> *our* hardware, not guarantees.

---

## TL;DR

- **On the H100 (dev box) everything is already real-time.** End-to-end speaker-pause → translated audio is **≈0.9 s**;
  TTS runs at **RTF 0.42** (generates ~2.4× faster than playback). Nothing to fix on GPU.
- **The only genuinely non-real-time component is TTS on the CPU/ONNX backend** — the *on-device deploy target*:
  **RTF ≈ 2.6 (EN) / 3.1 (JA)** → a 5 s reply takes ~13 s. ASR, MT and minutes on CPU are fine.
- The "streaming" CPU TTS is **fake streaming**: `OnnxAudioEngine.synthesize_stream` synthesizes the *whole* clip,
  then slices it into 0.5 s chunks (`src/csmeeting/live/onnx_engine.py:101`). It can't truly stream because CPU
  generation is slower than playback (RTF > 1), so there are no early frames to emit.
- A second, *structural* latency axis: the app is **turn-based, not simultaneous** — VAD translates on the pause
  (~0.9 s end-of-utterance latency, by design; Japanese is verb-final). Improvable but a bigger research effort.
- **Highest-leverage fix:** route ONNX TTS to the **Ryzen iGPU via the DirectML execution provider** (AMD reports
  DirectML ≈ **4.7× over CPU**) → comfortably crosses **RTF < 1**, the requirement for gap-free playback.
- **Second:** LFM2-Audio's **interleaved generation** minimizes *time-to-first-audio* and total tokens (Liquid's
  documented mode for real-time speech). Caveat: it is designed for **speech-to-speech**, not pure text→speech TTS.

---

## A. What is and isn't real-time (grounded in our numbers)

"Real-time" for spoken output means **RTF < 1** — the model must generate audio at least as fast as it plays, or
playback stutters. Source numbers: `documentation/system-resources.md` §A (H100, measured) and §B (CPU, measured on a
Xeon 6442Y as a *generous* proxy — a Ryzen laptop has fewer cores → slower).

### GPU path (PyTorch / `liquid_audio`, `AudioEngine`) — H100, measured

| Stage | Value | Real-time? |
|---|---|---|
| ASR | 0.3 s / utterance | ✅ |
| MT | 0.1 s / utterance | ✅ |
| TTS — time-to-first-audio | **0.46 s** | ✅ |
| TTS — real-time factor | **0.42** | ✅ (faster than playback) |
| Minutes | ~1.9 s (768 tok) | ⚠️ on-demand batch, latency-uncritical |
| **End-to-end** (pause → audio) | **≈ 0.9 s** | ✅ |

### CPU / on-device path (ONNX via `liquidonnx`, `OnnxAudioEngine`) — Xeon proxy, measured

| Stage | Value | Real-time? |
|---|---|---|
| ASR (JA & EN) | ~1.6 s / 3 s clip | ✅ |
| MT / minutes (text) | a few s / utterance | ✅ fine |
| **TTS** | **RTF ≈ 2.6 (EN) / 3.1 (JA)** | ❌ **the bottleneck** — turn-based (5 s reply ≈ 13 s) |

**Conclusion: the single non-real-time component is CPU TTS.** Everything else meets real-time on both paths
(minutes is batch, not a streaming concern).

### Two code facts that confirm the diagnosis

1. **CPU streaming is synthetic.** `OnnxAudioEngine.synthesize_stream` (`onnx_engine.py:101-106`) calls
   `synthesize()` (full clip) then yields 0.5 s slices. The real frame-by-frame streamer is the torch path
   (`engine.py:103 synthesize_stream`, which decodes a sliding window of Mimi frames as they are generated). The ONNX
   engine cannot stream because RTF > 1 leaves no head-room.
2. **Turn-based by design.** The cascade translates on the VAD pause (`src/csmeeting/live/vad.py`,
   `pipeline.py`), so even on GPU there is ~0.9 s *end-of-utterance* latency. `docs/translation.md` argues this is
   *linguistic* (Japanese verb-final) and that the cascade is deliberate (reuses the ASR fine-tune; emits the text the
   transcript + minutes features need). This is a defensible product choice, not a defect.

---

## B. Fix #1 (highest leverage) — run ONNX TTS on the Ryzen iGPU via DirectML

**Why it's the right lever.** Our own `system-resources.md` §B already names it: *"The Ryzen Radeon iGPU via
onnxruntime DirectML EP is the way to push TTS toward real-time."* The research backs the magnitude: AMD reports the
**DirectML execution provider averages ≈ 4.7× over CPU**. Applied to RTF ≈ 2.6 that lands at **RTF ≈ 0.55** —
under 1, i.e. gap-free. This is the change most likely to make spoken output real-time on-device.

**What to change (this repo).**
- `OnnxAudioEngine` currently lets `liquidonnx` create its onnxruntime sessions with default (CPU) providers. Add an
  **execution-provider selector** (`LiveConfig.onnx_ep` = `cpu | dml | cuda | auto`) and inject the provider list into
  every session `liquidonnx` builds. Because `liquidonnx`'s constructor signature isn't guaranteed to accept a
  `providers=` kwarg, the robust mechanism is to (a) pass `providers=`/`execution_providers=` if the signature
  supports it, and (b) fall back to a scoped patch of `onnxruntime.InferenceSession.__init__` during model
  construction that injects our providers (and, for DML, `SessionOptions(enable_mem_pattern=False,
  execution_mode=ORT_SEQUENTIAL)` — DirectML requires mem-pattern off). Restore the original afterwards.
- Expose `--onnx-ep dml` on `scripts/demo_app.py`.

**Operational notes / landmines.**
- DirectML needs the **`onnxruntime-directml`** wheel (Windows), *not* stock `onnxruntime`; the two conflict — install
  one. It is a **Windows** EP (or WSL2 caveats apply).
- DML favors **fp16**; if RTF is still marginal at q4, also benchmark the fp16 ONNX export.
- **NPU will not save TTS.** The FastConformer encoder + audio detokenizer have **no NPU kernels** (already noted in
  §B); only the LM backbone offloads. AMD's own guidance is **hybrid NPU/iGPU** — route TTS to the **iGPU**, and (if
  desired) the text LM to the NPU. Do not promise "NPU speech-to-speech."
- **Validate on the real laptop.** Xeon ≠ Ryzen; the 4.7× is an average across AMD's models, not a TTS guarantee. Use
  `scripts/bench_tts.py` (this work) to measure RTF + TTFA for `cpu` vs `dml`.

---

## C. Fix #2 — LFM2-Audio interleaved generation for the spoken path

**What it is.** `liquid_audio.model.lfm2_audio.LFM2AudioModel` exposes **two** generators with identical signatures:
- `generate_sequential` — model stays in one modality and switches via special tokens (`<|audio_start|>`=128,
  `<|im_end|>`=7). This is what `AudioEngine` uses today, and is **Liquid's documented mode for ASR/TTS**.
- `generate_interleaved` — emits text and audio in a **fixed interleaved pattern** (`conf.interleaved_n_text` text
  tokens, then `conf.interleaved_n_audio` audio frames, …; `<|text_end|>`=130 flips it to audio). Liquid's blog: this
  *"minimizes time to first audio output and the number of tokens generated … ideal for naturally flowing real-time
  speech-to-speech on resource-constrained devices."* Liquid reports **< 100 ms** end-to-end in interleaved S2S.

**The honest caveat (important).** Interleaved's win is for **speech-to-speech**, where the model *reasons in text and
speaks at the same time* — first audio arrives after only `interleaved_n_text` tokens instead of after the whole
response. Our cascade's spoken leg is **pure TTS**: the translation text is already produced by the *text* model and
handed to the audio model, which only needs to emit audio. In that setting:
- Sequential already streams audio frame-by-frame, so TTFA is already ~0.46 s on GPU — interleaved may not beat it.
- Interleaved on a fixed-text TTS turn can emit **spurious interleaved text tokens**, which we'd have to discard and
  which can perturb prosody/timing.

So interleaved is implemented here as a **selectable, benchmarkable mode** (`LiveConfig.tts_generation_mode`), not the
default. Where it genuinely pays off is the **end-to-end speech-to-speech variant**: feed source audio to the audio
model and let it emit target text+audio interleaved (fusing MT+TTS, or even ASR+MT+TTS). That trades against the
cascade's design rationale (the cascade reuses the ASR fine-tune and emits text for transcript/minutes), so treat
S2S-interleaved as an **experiment**, measured against the cascade on latency *and* quality (chrF) before adopting.

**What to change (this repo).**
- `LiveConfig.tts_generation_mode` = `sequential | interleaved`; `AudioEngine.synthesize` / `synthesize_stream`
  dispatch to `generate_interleaved` when selected (audio-frame extraction is identical: `numel == codebooks` and all
  codes `< 2048`; terminal frame is all `== 2048`; text tokens have `numel == 1` and are skipped).
- `--tts-mode interleaved` on `scripts/demo_app.py`; compare TTFA sequential vs interleaved via `scripts/bench_tts.py`.

---

## D. Fix #3 (polish) — quantization & model shape

Already at q4. Remaining levers, in rough order:
- **fp16 on iGPU/DML** — often faster than q4 on GPU-class EPs (q4 dequant overhead); measure both.
- **One-audio-model variant** (§A) — fine-tune ASR from the English base so a single checkpoint does ASR + EN TTS →
  ~5.4 GB weights, fewer sessions to place on the iGPU.
- **q8 vs q4** RTF/quality A/B on the laptop.
- Fewer Mimi codebook frames per decode step trades quality for speed (last resort).

---

## E. Fix #4 (separate, bigger) — turn-based → simultaneous

If we want to beat the ~0.9 s *end-of-utterance* latency (not just CPU throughput), the simultaneous-translation
literature applies directly — but Japanese verb-final word order bounds the gains (you often must wait for the verb;
wait-k policies formalize this). Options, increasing effort:
- **Stable partial hypotheses + run-on decoding** — cut word-level latency **18.1 s → 1.1 s** with no WER loss (KIT,
  arXiv 2003.09891).
- **Translate PARTIAL + FINAL hypotheses** (incremental / re-translation MT) instead of waiting for the full pause.
- **Alignment-based streaming MT for on-device cascades** (arXiv 2508.13358) — almost exactly our architecture.

Recommendation: **defer.** Diminishing returns for JA specifically; the turn-based design is defensible for the demo.

---

## F. Recommended order of work

1. ✅ **DirectML EP** for `OnnxAudioEngine` (+ `--onnx-ep`, benchmark) — the actual real-time unlock on-device.
2. ✅ **Interleaved generation** mode for `AudioEngine` (+ `--tts-mode`, benchmark TTFA) — selectable, measured.
3. ⬜ Quantization A/B (fp16 vs q4, one-audio-model) on the real Ryzen laptop.
4. ⬜ (Stretch) simultaneous MT — only if turn latency becomes the headline complaint.

> **Validation status:** items 1–2 are implemented in code in this commit but **not yet executed** — `liquidonnx`, the
> model weights, and a GPU/iGPU are only on the host (`the GPU host`) / the Ryzen laptop, not the dev workstation. Run
> `scripts/bench_tts.py` on the target hardware to fill in the real RTF/TTFA numbers and update
> `documentation/system-resources.md` §B.

---

## Sources

- Repo (measured): `documentation/system-resources.md` §A/§B; `src/csmeeting/live/{engine,onnx_engine,config,vad,pipeline}.py`; `docs/translation.md`.
- AMD — *Model Pipelining on NPU and GPU using Ryzen AI Software*: https://www.amd.com/en/developer/resources/technical-articles/model-pipelining-on-npu-and-gpu-using-ryzen-ai-software.html
- AMD — *Hybrid NPU/iGPU Optimized Agent on Ryzen AI*: https://www.amd.com/en/developer/resources/technical-articles/2025/hybrid-npu-igpu-optimized-agent-on-amd-ryzen-ai-powered-pc-.html
- onnxruntime — *DirectML Execution Provider*: https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- Ryzen AI Software — *DirectML Flow*: https://ryzenai.docs.amd.com/en/latest/gpu/ryzenai_gpu.html
- Liquid AI — *LFM2-Audio: An End-to-End Audio Foundation Model* (interleaved vs sequential; <100 ms S2S): https://www.liquid.ai/blog/lfm2-audio-an-end-to-end-audio-foundation-model
- LiquidAI/LFM2.5-Audio-1.5B model card: https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B
- `liquid_audio` source (generation API): https://github.com/Liquid4All/liquid-audio/blob/main/src/liquid_audio/model/lfm2_audio.py
- KIT — *Low Latency ASR for Simultaneous Speech Translation* (arXiv:2003.09891): https://arxiv.org/abs/2003.09891
- *Overcoming Latency Bottlenecks in On-Device Speech Translation: Alignment-Based Streaming MT* (arXiv:2508.13358): https://arxiv.org/html/2508.13358v1
