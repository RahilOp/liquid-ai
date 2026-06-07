# Dataset research — index & data strategy

Six primary-source-verified briefs on the best datasets for every task in the Kaigi pipeline, ranked for
**commercial deployability** (the product ships under the permissive LFM Open License, so dataset license clarity
matters). This README is the **synthesis**: the cross-cutting finding, a master license table, the recommended
commercial-safe recipe per task, and the decisions/landmines that change our current code.

> Compiled 2026-06-06 by a fleet of research agents. Each brief quotes licenses from the primary source (HF dataset
> card / paper / project page). Treat every "commercial?" verdict as *engineering due diligence, not legal advice* —
> the flagged items warrant a real legal read before a production ship.

## The briefs

| # | File | Covers |
|---|---|---|
| 01 | [`01-codeswitch-datasets.md`](01-codeswitch-datasets.md) | JP↔EN code-switch speech + text; canonical CS corpora (transfer); CS text for generation |
| 02 | [`02-japanese-monolingual-asr.md`](02-japanese-monolingual-asr.md) | Monolingual JA ASR (the ~55% anchor bucket) |
| 03 | [`03-english-monolingual-asr.md`](03-english-monolingual-asr.md) | Monolingual EN ASR replay (anti-forgetting) + meeting-domain EN |
| 04 | [`04-speech-translation-parallel-text.md`](04-speech-translation-parallel-text.md) | JA↔EN speech translation + parallel text (for the TTS-translation feature & synthetic gen) |
| 05 | [`05-tts-models-and-datasets.md`](05-tts-models-and-datasets.md) | TTS models to *generate* synthetic CS audio + TTS corpora |
| 06 | [`06-meeting-summarization-minutes.md`](06-meeting-summarization-minutes.md) | Meeting/minutes summarization text SFT (for the local minutes model) |
| 07 | [`07-realtime-latency-and-cpu-tts.md`](07-realtime-latency-and-cpu-tts.md) | **Real-time latency**: which stages aren't real-time + the CPU-TTS fix (DirectML iGPU, interleaved gen, quantization) |
| 08 | [`08-improving-translating-tts.md`](08-improving-translating-tts.md) | **Improving the translating-TTS LoRA** (2024–26): CFG/DPO, 24 kHz data-match, LibriTTS-R/HiFi-TTS/FLEURS-R, code-switch synth — drove the v6 win |

---

## The one finding that shapes everything

**There is no commercially-licensed, real-voice JP↔EN code-switch speech corpus.** Every real JP↔EN CS speech
resource is non-commercial/no-redistribute (JECS 2.5h, CS-FLEURS, SwitchLingua), synthetic, or not-English. This
**confirms the synthetic-first strategy** — and it's literature-validated: an ablation found *"differences between
synthetic and authentic speech have negligible impact on [speech-translation] quality"* (arXiv 2602.21646).

So: **real CS data = eval gold only** (JECS is our primary held-out test). **Bulk training = our synthetic pipeline**,
driven by **commercially-clean parallel text** and rendered by **permissively-licensed TTS**.

---

## Master license posture (consolidated across all six briefs)

### 🟢 Ship-safe — commercial + redistribute (build the shipped training mix from these)
| Dataset | Task | License | Note |
|---|---|---|---|
| **Common Voice** JA/EN | mono ASR | **CC0** | cleanest; use CC0 mirror `fsicoli/...` (official repo gated since Oct 2025) |
| **FLEURS** (ja_jp, en_us) | mono ASR + **ST anchor** | **CC-BY-4.0** | only commercial *real* JA→EN ST signal (FLoRes id-join) |
| **LibriSpeech** | EN replay | **CC-BY-4.0** | clean read-English anchor |
| **AMI**, **ICSI** | EN replay + **meeting domain** | **CC-BY-4.0** | ⭐ replay *and* domain adaptation in one |
| **VoxPopuli** (en) | EN ASR | **CC0** (transcripts) | free GigaSpeech alternative |
| **People's Speech** `clean` | EN ASR | **CC-BY-4.0** | use only `clean`/`dirty`, not `*_sa` |
| **ASCEND** | CS acoustics (zh-en transfer) | **CC-BY-SA-4.0** | copyleft; teaches the *act* of switching |
| **JESC** | parallel text (gen) | **CC-BY-SA-4.0**¹ | ⭐ 2.8M colloquial JA-EN; primary gen fuel |
| **Tatoeba** ja-en | parallel text | **CC-BY-2.0** | short everyday sentences |
| **KFTT / WikiMatrix** | parallel text | **CC-BY-SA-3.0/4.0** | copyleft; formal domain |
| **Aya** (jpn) | minutes/instruction SFT | **Apache-2.0** | human-written JA |
| **databricks-dolly-15k-ja** | minutes/instruction SFT | **CC-BY-SA-3.0** | summarization slice |
| **QMSum** | meeting summ SFT | **MIT / Apache-2.0** | query-based → action items |
| **TTS models** Kokoro, MeloTTS, CosyVoice2, Piper, Parler | synth generation | **Apache / MIT** | outputs commercial-safe |
| **TTS data** LJSpeech (PD), VCTK, LibriTTS-R | TTS ref | PD / CC-BY-4.0 | |

¹ JESC: Stanford says CC-BY-SA-4.0, HF card says CC-BY-4.0 → treat as **CC-BY-SA** (comply with share-alike).

### 🟡 Use with care — weights-only / paid / copyleft / contact-required
| Dataset | Issue | Safe use |
|---|---|---|
| **ReazonSpeech** (35k h JA) | gated to JP Copyright Act **Art. 30-4** (TDM) | **train weights** (the ecosystem norm — Kotoba-Whisper ships Apache); **do NOT redistribute the audio** |
| **JSUT / JVS** (JA TTS/read) | free incl. commercial *by contact*, **no redistribution** | train/replay only; never ship the audio |
| **HTH "Kataro"** (120h JA conversational) | **paid** commercial license | best domain match *if* there's data budget |
| **YODAS** (JA/EN) | CC-BY-3.0 repo but per-video YouTube risk | weights-only; filter hard |
| **Earnings-22 / People's-Speech `*_sa` / KFTT / WikiMatrix / JESC** | **copyleft** (share-alike) | OK commercially; a redistributed *derived dataset* must stay share-alike |
| **ELITR Minuting Corpus** | CC-BY-SA-4.0 *per LINDAT — verify badge* | real minutes (EN/CS, no JA) |
| **Jenny** (kit's EN TTS example) | permissive but no-resale-of-data | train a voice, don't redistribute data |

### 🔴 Research / NC / blocker — eval & ablation ONLY, never in shipped weights
JECS (our eval gold), CS-FLEURS, SwitchLingua, CS-Dialogue, SEAME (paid), **BSD** (NC — but best business template),
**CoVoST2** (NC — ST eval), JParaCrawl (bans commercial), OpenSubtitles/OPUS-100 (unknown), TED-LIUM3 (NC-ND),
GigaSpeech (restricted), SPGISpeech (no-redistribute), MediaSum (research), SAMSum (NC-ND), DialogSum (NC),
MeetingBank (NC), XLSum (NC), ichikara free tier (NC; commercial=paid), **XTTS-v2 / F5-TTS / fish-speech** (NC TTS),
**edge-tts output** (Microsoft service ToS — see landmines).

---

## Recommended commercial-safe recipe (per task)

| Task | Anchor (real, commercial) | Bulk (synthetic / commercial) | Eval (may use NC) |
|---|---|---|---|
| **CS ASR** (core) | — (none exists) | **our synthetic** CS audio, generated from JESC/Tatoeba text, optionally SWORDS-style; + ASCEND as zh-en CS-acoustic transfer | **JECS** (gold) + held-out synthetic + CS-FLEURS (romaji-normalized) |
| **JA mono** (~55%) | Common Voice JA (CC0) + FLEURS ja (CC-BY) | ReazonSpeech `small/medium` CER-filtered (weights-only) + YODAS ja | FLEURS/CV ja test, ReazonSpeech test |
| **EN replay** (~12%) | **LibriSpeech** clean + **AMI ihm** (CC-BY) + CV-en (CC0) | — | LibriSpeech test.clean + AMI test |
| **JA→EN ST** | **FLEURS** ja→en (FLoRes join) | **JA-TTS over JESC + Tatoeba** (~50–150k) | **CoVoST2 ja_en test** (NC, eval only) |
| **TTS engine** | — | **Kokoro** (Apache, 24kHz splice) + **CosyVoice2** (Apache, native JP+EN one-utterance) | A/B by ear |
| **Minutes** | AMI + QMSum + Aya-ja + dolly-ja | **synthetic** transcript→minutes via a **local open-weight teacher** (confidential, license-clean) | held-out synthetic; MeetingBank/DialogSum for ablation only |

**Refined corpus mix** (audio fine-tune; scale to compute): **~30% synthetic CS** · **~55% JA mono** (CV+FLEURS ship-safe
core, ReazonSpeech weights-only accelerant) · **~12% EN replay** (AMI-led for domain match) · plus a separate
**ST bucket** if we train the translation task in.

---

## Decisions & landmines that change our current code

1. **🔴 Swap edge-tts → Kokoro + CosyVoice2.** Our `src/csmeeting/tts_backends.py` defaults to **edge-tts**, whose audio
   is **not commercial-clean** (governed by Microsoft Edge's read-aloud *service* ToS, not the LGPL wrapper; the
   maintainer frames it as personal-use). For shippable synthetic data, use **Kokoro-82M** (Apache-2.0, native 24 kHz —
   matches our target, drop-in splice replacement) and **CosyVoice2-0.5B** (Apache-2.0) for **native JP+EN-in-one-
   utterance** (better switch-point prosody → less reliance on our span-splicing). → add `KokoroBackend` /
   `CosyVoice2Backend` to `tts_backends.py`; keep edge-tts only for offline throwaway prototyping.
2. **🟡 CosyVoice2 reduces the splicing problem.** It renders mixed-language utterances natively, so the realistic,
   high-CS-density lines get natural prosody; reserve splicing (Kokoro) for bulk monolingual-span volume.
3. **🚩 Verify the speech-translation system prompt empirically.** No public LFM doc names an ST task string. Test
   candidates (`"Perform speech translation to english."`, `"Perform ST."`, `"Translate the audio to English."`) against
   the model, or just *define and train* our own prompt. (Only matters when we add the ST task.)
4. **🟡 ReazonSpeech = weights-only.** Train on it, document the Art. 30-4 basis, never redistribute the audio. If a
   customer needs bulletproof indemnification, lean on CV+FLEURS+YODAS for the public-safe portion.
5. **🚩 Confirm the `-JP` base checkpoint.** Two agents couldn't independently locate a public `LFM2.5-Audio-1.5B-JP`
   on the Hub (the round-1 audio agent *did* find and quote its card, incl. 4.42 CER). Re-confirm the exact repo id /
   provenance before training. *(Our round-1 research treated it as real; this is a cross-check flag.)*
6. **🟢 Add AMI to the replay mix.** It's the single best English add — CC-BY-4.0, one-line load, *and* meeting-domain,
   so it prevents English forgetting **and** adapts to our actual product domain.

## Open verification items (consolidated)
- ReazonSpeech audio-redistribution legal status (Art. 30-4) — conservative read = weights-only.
- ELITR Minuting Corpus license badge (CC-BY-SA-4.0 per LINDAT — verify on download).
- TALCS license (paper says "permissive", names none → check 100TAL portal ToS) before any commercial use.
- CosyVoice2 weights repo — confirm no separate model-agreement file beyond Apache-2.0; A/B JP↔EN prosody.
- LFM speech-translation prompt string (above).
- `-JP` base checkpoint id/provenance (above).
- Common Voice post-v17 access now routes via Mozilla Data Collective; CC0 grant per-release.
