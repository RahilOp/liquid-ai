# 08 — Improving the translating-TTS LoRA (techniques + datasets)

Deep-research synthesis (2024–2026 sources, adversarially verified: 25 claims → 24 confirmed, 1 killed) on how to
raise speech quality + translation accuracy for our LFM2.5-Audio translating-TTS LoRA (JA↔EN + code-switch).
See also [`finetuning-translate-tts.md`](../documentation/finetuning-translate-tts.md).

## TL;DR — highest-leverage moves
1. **Fix the data to match Mimi's 24 kHz / 12.5 Hz operating point.** Our v4 regression (real 16 kHz FLEURS HURT) is
   corroborated: a 24 kHz Mimi decoder shouldn't be trained on noisy 16 kHz targets. Use **24 kHz, clean** audio;
   bandpass **80 Hz–7 kHz** + amplitude-normalize any real audio.
2. **Swap synthetic Kokoro English for real studio-grade 24 kHz English** (the weak side): **LibriTTS-R** (24 kHz,
   exact Mimi match) and **HiFi-TTS** (44.1 kHz → 24 kHz), both **CC BY 4.0**. More synthetic voices = diminishing
   returns (we already saw this).
3. **Add inference-time CFG (classifier-free guidance)** — no retraining; on Koel-TTS it cut CER 2.56% → 0.69%.
   Later, **DPO** with ASR-CER + speaker-verification rewards (CER → 0.58%). These target intelligibility directly.
4. **Code-switch:** synthesize JA-EN with the **UniCoM/SWORDS** recipe (LLM POS word-substitution + MMS-FA forced
   alignment + kNN-VC one-voice unification) instead of relying on bilingual speakers; or use **JECS** (real JA-EN).

## Datasets (verified, with license)
| Dataset | Lang | Hours | SR | License | Use |
|---|---|---|---|---|---|
| **LibriTTS-R** (OpenSLR #141) | EN | ~585 (2456 spk) | **24 kHz** | CC BY 4.0 | real clean English targets — **the** fix for the weak EN voice (exact Mimi SR) |
| **HiFi-TTS** (OpenSLR #109) | EN | ~292 (10 spk, ≥17 h/spk) | 44.1 kHz→24 kHz | CC BY 4.0 | additional very-clean English (≥13 kHz BW, ≥32 dB SNR) |
| **FLEURS-R** (arXiv 2408.06227) | 102 langs (ja, en) | ~1300 | **24 kHz** | CC BY 4.0 | real JA↔EN parallel speech, Miipher-restored — the **right** FLEURS (not raw 16 kHz) |
| **JVS** | JA | ~30 (100 spk) | 24 kHz | free (research; check terms) | clean Japanese multi-speaker |
| **JECS** (Takamichi) | JA-EN code-switch | small | — | check terms | real code-switch JA-EN |
| **CoVoST2 / covost2NativeJa** | JA↔EN | — | — | CC0/CC BY | parallel translation text/speech |
| **Emilia** (amphion) | multi | 100k+ | 24 kHz | CC BY-NC (non-commercial!) | large in-the-wild; NC license — avoid for commercial |

Process all real audio: resample/keep **24 kHz**, **bandpass 80 Hz–7 kHz**, **amplitude-normalize** (UniCoM recipe).
LibriTTS-R/HiFi-TTS derive from LibriVox (high-SNR amateur, not literal studio) — fine, standard restored corpora.

## Techniques (verified)
- **CFG at inference** (Koel-TTS, arXiv 2502.05236): γ≈2.5, doubles inference batch, CER 2.56→0.69%, no retraining.
- **DPO preference alignment** (Koel-TTS): rewards = ASR-CER + speaker-verification → CER 0.62%; **DPO+CFG = 0.58%**.
- **Dual CFG** (DualSpeech, arXiv 2408.14423): separate text-intelligibility vs speaker weights — push EN clarity
  without flattening prosody. (Selective CFG arXiv 2509.19668 corroborates.)
- **Mimi internals** (Moshi paper): 24 kHz→12.5 Hz, Q=8×2048=1.1 kbps; **RVQ-1 = WavLM semantic token** (acoustic-only
  tokens "cannot produce intelligible speech") → intelligibility lives in the semantic token; ASR-guided DPO acts there.
- **DualCodec** (arXiv 2505.13000): semantic RVQ-1 + larger codebook sharply raises low-bitrate intelligibility —
  guidance for any future codec/decoder tuning (caveat: codec-reconstruction, not LM-generation, results).
- **Code-switch synthesis** (UniCoM/SWORDS, arXiv 2508.15244): POS-mapped substitution (nouns/verbs/interjections,
  ≤3 embedded words) + MMS-FA + kNN-VC. (Caveat: released CS-FLEURS is European pairs; JA-EN is an extrapolation.)

## Caveats
- CFG/DPO results are from encoder-decoder/diffusion codec-LM TTS, **not** our exact Mimi/RQ-Transformer stack —
  techniques are architecture-general but magnitudes will differ; CFG needs a guidance formulation for the depthformer.
- DualCodec numbers are codec **reconstruction** at fixed bitrate, not end-to-end generation difficulty.
- JA-EN code-switch via UniCoM is an **extrapolation** (no published JA-EN result).
- Emilia is **CC BY-NC** — non-commercial; don't use if commercial.

## Prioritized action plan (our iteration order, GPU 3)
1. **LibriTTS-R real 24 kHz English** targets for ja→en pairs (translate EN text→JA for input) — fix the weak EN voice.
2. **+ HiFi-TTS** (→24 kHz) for more clean English diversity.
3. **FLEURS-R** (24 kHz, bandpass+normalize) for real JA↔EN parallel — redo the failed v4 correctly.
4. **CFG at inference** in `translate_speak` (no retrain) — try on the best adapter.
5. **Code-switch** synthesis (UniCoM-style / JECS) for the code-switch path.
6. **Mixing-ratio / curriculum** tuning of {real EN, FLEURS-R, Kokoro, code-switch}; find the real-vs-synthetic crossover.
7. (Later/heavy) **DPO** with ASR-CER + speaker-verification rewards.

Open questions: does CFG/DPO reproduce on the LFM2.5 depthformer? Is the weak EN voice mostly data (likely) vs
decoder? Right multi-task mixing ratio to avoid the noisy-real regression?
