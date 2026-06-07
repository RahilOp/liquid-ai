# Datasets — Monolingual Japanese ASR

*The monolingual-Japanese anchor bucket (~50–70% of the mix) to teach/strengthen Japanese and anchor the model while adding the code-switch skill. Verified against HF cards, papers, project pages (2026-06-06).*

> **Licensing implication:** the binding constraint on "commercially deployable" is the **training-data** license, not the model license. The single biggest nuance is **ReazonSpeech's Article 30-4 gating** (see §4).

---

## 1. Recommended shortlist

### (a) Bulk — teach/strengthen Japanese (~100–500h *used*)
| Priority | Dataset | Use | Why |
|---|---|---|---|
| **#1 anchor** | **ReazonSpeech** (`reazon-research/reazonspeech`, `small`=100h / `medium`=1,000h) | **200–400h** CER-filtered | Only large free **spontaneous broadcast** JA corpus → closest free proxy to conversational/business register. Art. 30-4 gating → §4. CER-filter (Kotoba-Whisper kept WER≤10). |
| **#1b license-clean alt** | **YODAS / YODAS2 ja** (`espnet/yodas`, `espnet/yodas2`) | **50–150h** (filter hard) | CC-BY-3.0, CC-licensed YouTube → spontaneous, varied. Per-video license risk + noisy auto-captions. |
| **#2 fully-open backbone** | **Common Voice JA** (`mozilla-foundation/common_voice_17_0`; CC0 mirror `fsicoli/...`) | **100–300h** validated | **CC0** — safest. CV25 ≈ 372h validated. Read speech but license-perfect; `sentence_domain` tags enable business weighting. |

**Strategy:** lead with **CV (CC0)** as the safe backbone, layer **ReazonSpeech-filtered** for spontaneity/scale, sprinkle **YODAS** for diversity. For bulletproof redistribution, demote ReazonSpeech to weights-only or drop for CV+YODAS+FLEURS.

### (b) Small clean replay slice (~5–20h)
- **FLEURS ja** (`google/fleurs`, `ja_jp`) — ~10h, CC-BY-4.0, pristine, one-line load. Replay + free eval.
- **Common Voice JA test/validated** — CC0 held-out clean read speech.
- *(optional)* **JSUT basic5000** — ~5h single-speaker studio; pronunciation anchor. **Train-only (audio not redistributable).**

**Eval (not training):** `japanese-asr/ja_asr.reazonspeech_test`, FLEURS ja test, CV ja test, TEDxJP-10K.

---

## 2. Comparison table

| Dataset | Hours (JA) | Domain | License — commercial? redistribute? | HF ID / Download | Notes |
|---|---|---|---|---|---|
| **ReazonSpeech v2** | ~35,000h (tiny 8.5h / small 100h / medium 1,000h / large 5,000h / all 35,000h) | **Spontaneous broadcast TV** | Corpus **CDLA-Sharing-1.0** but **gated**: agree "use SOLELY for Copyright Act **Art. 30-4**." Commercial *model* use is the norm; **redistributing raw audio = grey** (§4) | `reazon-research/reazonspeech` (🔒 gated, `trust_remote_code=True`); parquet+EN: `japanese-asr/ja_asr.reazon_speech_all` | Best domain match for spontaneous JP. |
| **Common Voice JA (CV25)** | 725h total / **372h validated** | Read (crowd) | **CC0-1.0** ✅✅ | `mozilla-foundation/common_voice_17_0`; CV18–25 via Mozilla Data Collective | Domain-weak; license-perfect; `sentence_domain` tags. |
| **YODAS / YODAS2 (ja)** | ja = portion of 369,510h total | **Spontaneous YouTube** | **CC-BY-3.0**, ✅ in principle, **per-video risk + takedown** | `espnet/yodas`, `espnet/yodas2` (`trust_remote_code=True`) | `ja000`=manual, `ja100`=auto. Filter. |
| **FLEURS ja** | ~10h | Read (Wikipedia register) | **CC-BY-4.0** ✅✅ | `google/fleurs` (`ja_jp`) | Tiny, pristine. Replay + eval. |
| **JTubeSpeech** | ~1,300h ASR | Spontaneous YouTube | Repo Apache-2.0, **ID lists only** (you scrape YouTube) | github.com/sarulab-speech/jtubespeech | Per-video copyright burden. |
| **LaboroTVSpeech v1/v2** | ~2,036h train | **Broadcast TV** | Corpus **NOT open** — application-gated; commercial unclear | github.com/laboroai/LaboroTVSpeech | Good domain, access-gated. |
| **CSJ** | ~650h (~90% monologue) | **Spontaneous monologue** (academic) | **Paid / NINJAL**. ❌ free ❌ redistribute | clrd.ninjal.ac.jp/csj | Gold spontaneous corpus but license-incompatible; lecture-monologue not meeting-dialogue. |
| **JSUT** | 10h | Read (1 female, studio) | Labels CC-BY-SA-4.0; **audio: free incl. commercial via contact, redistribution NOT permitted** | takamichi site; mirror `japanese-asr/ja_asr.jsut_basic5000` | Pronunciation anchor. **Train-only.** |
| **JVS** | 30h (100 spk) | Read/TTS (studio) | Same as JSUT (non-redistributable) | takamichi site | Speaker diversity. Train-only. |
| **HTH "Kataro"** | 120h | **Casual conversational** | **Commercial — paid** (info@hth-inc.com); preview free | `HTH-inc/japanese-casual-conversational-speech-golden-dataset-preview` | **Closest to conversational/meeting domain with a clean commercial license** if budget allows. |
| **joujiboi/japanese-anime-speech (v1)** | ~10K–100K clips | Anime/VN dialogue | **CC0-1.0** ✅✅ (v2 GPL — avoid) | `joujiboi/japanese-anime-speech` | CC0 emotive dialogue, but domain mismatch. |

---

## 3. Per-dataset details (highlights)

### ReazonSpeech (top anchor)
- **Access:** gated — agree to *"USE THE DATASET SOLELY FOR ... ARTICLE 30-4"*, then `huggingface-cli login`.
- **Load (pick a subset, not `all`=2.3TB):**
  ```python
  ds = load_dataset("reazon-research/reazonspeech", "small", trust_remote_code=True, split="train")  # 100h/6GB
  # configs: tiny(8.5h)|small(100h)|medium(1000h)|large(5000h)|all(35000h); cols: audio, transcription
  # parquet alt w/ EN translation col (useful for CS task):
  ds = load_dataset("japanese-asr/ja_asr.reazon_speech_all", "subset_0", split="train")
  ```
- **CER-filter:** transcribe candidates with a strong JP model, keep CER ≤ ~10–15% + 1–20s. (Kotoba-Whisper-v2 kept ~7.2M clips at WER≤10.)
- **Gotchas:** TV music/SFX/overlap; subtitle drift; no speaker labels; ~5s clips.

### Common Voice JA
- CV17 = 477h/124h validated; **CV25 = 725h/372h validated**. Since Oct 2025 CV distributed via **Mozilla Data Collective**; `common_voice_17_0` is the last easy HF load (or CC0 mirror `fsicoli/common_voice_17_0`).
  ```python
  ds = load_dataset("mozilla-foundation/common_voice_17_0", "ja", split="validated", trust_remote_code=True)
  ```
- Up-weight business `sentence_domain` (finance/news/technology/service). Read speech → pair with ReazonSpeech/YODAS.

### YODAS ja
- `load_dataset("espnet/yodas", "ja000", split="train", trust_remote_code=True)`. `ja000`=manual (still CER-filter), `ja100`=noisier. YODAS2 = 24 kHz long-form (segment to ≤20s).

### FLEURS ja
- `load_dataset("google/fleurs", "ja_jp", split="train")`. ~10h, CC-BY-4.0.

---

## 4. Licensing posture

**🟢 Commercial + redistributable:** Common Voice JA (CC0), FLEURS ja (CC-BY-4.0), YODAS (CC-BY-3.0, per-video caveat), joujiboi v1 (CC0).

**🟡 Paid / contact / train-only:** HTH Kataro (paid commercial — best domain match); JSUT/JVS (free incl. commercial via contact, **no redistribution** → train-only).

**🔴 Research-only / paid:** CSJ (paid, non-redistributable); LaboroTVSpeech (application-gated, commercial unclear).

### The ReazonSpeech redistribution question (the crux)
- Corpus license = **CDLA-Sharing-1.0**, but the HF download is gated to **Art. 30-4** ("information analysis" TDM exception).
- **Art. 30-4 explicitly permits commercial AI training in Japan** (unlike EU/UK). The ecosystem reflects this — **Kotoba-Whisper / ReazonSpeech models ship Apache-2.0**.
- The grey zone is **re-publishing the raw audio** (third-party TV copyright).
  - ✅ **Safe:** train on ReazonSpeech, **release fine-tuned weights** commercially (established norm).
  - ⚠️ **Avoid:** redistributing ReazonSpeech audio clips inside your product/dataset.
  - **Recommendation:** use as **training input only**, keep audio internal, document the Art. 30-4 basis. For low risk-tolerance (enterprise indemnification), lean on **CV+FLEURS+YODAS** for the public-safe portion, or consult JP IP counsel.

---

## 5. How to use — recommended monolingual-JA mix (~300–400h effective)

| Source | Hours | Rationale |
|---|---|---|
| **ReazonSpeech** (small/medium, CER≤~12) | **180h** | Spontaneous broadcast register; workhorse. |
| **Common Voice JA** (validated, business-weighted) | **120h** | CC0 safety net + diversity. |
| **YODAS ja000** (CER-filtered) | **60h** | Real-world acoustic/topical diversity. |
| **FLEURS ja + JSUT basic5000** | **~15h** | Clean replay / pronunciation anchor (JSUT train-only). |
| *(optional, paid)* **HTH Kataro** | **+60–120h** | Direct conversational/meeting-domain lift. |

**Domain weighting toward business:** up-weight CV25 business `sentence_domain`; prefer ReazonSpeech news/talk + YODAS interview/lecture over variety; buy HTH if budget. Reserve genuine meeting/CS realism for the synthetic CS bucket — the mono bulk's job is to anchor JP.

**Preprocessing:** resample 16 kHz mono (`cast_column("audio", Audio(sampling_rate=16000))`); length filter 1–20s; CER/WER filter (ReazonSpeech/YODAS); consistent NFKC/punctuation normalization shared with the CS bucket; wrap `(audio, transcript)` into `liquid_audio` ChatMessage; de-dup + keep eval sets disjoint.

---

## 6. Sources & uncertainties

**Sources:** ReazonSpeech https://huggingface.co/datasets/reazon-research/reazonspeech , https://research.reazon.jp/projects/ReazonSpeech/ , https://github.com/reazon-research/ReazonSpeech · Kotoba-Whisper https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0 · CV stats https://github.com/common-voice/cv-dataset · Mozilla Data Collective https://mozilladatacollective.com/ · YODAS https://huggingface.co/datasets/espnet/yodas , https://arxiv.org/abs/2406.00899 · FLEURS https://huggingface.co/datasets/google/fleurs · JTubeSpeech https://github.com/sarulab-speech/jtubespeech , https://arxiv.org/abs/2112.09323 · LaboroTV https://github.com/laboroai/LaboroTVSpeech · CSJ https://clrd.ninjal.ac.jp/csj/en/ · JSUT/JVS https://sites.google.com/site/shinnosuketakamichi/publication/jsut · HTH https://huggingface.co/datasets/HTH-inc/japanese-casual-conversational-speech-golden-dataset-preview · Art. 30-4 https://www.japaneselawtranslation.go.jp/en/laws/view/3379

**Uncertainties:** `-JP` base checkpoint unverified (public LFM2.5-Audio card lists English only — confirm provenance). YODAS ja exact hours unpublished (estimate). ReazonSpeech audio-redistribution legal status genuinely grey (weights-only = conservative read, not legal advice). CV post-v17 audio now behind Mozilla Data Collective. LaboroTV v1 vs v2 hours conflated in sources.
