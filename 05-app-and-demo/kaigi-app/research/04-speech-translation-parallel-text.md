# Datasets — JA↔EN Speech Translation + Parallel Text

*For the **live JA→EN speech-translation + spoken (TTS) output** feature. We may fine-tune an ST task (JA audio in → EN text out) and/or pipeline ASR→MT→TTS. Verified against HF cards/papers/project pages (2026-06-06).*

> **Reality:** real JA-source ST data barely exists. **CoVoST2 ja_en** is the exact task direction but **tiny (~1.1k) and non-commercial**; **FLEURS ja** is the only commercially-licensed real signal (~2.3k). → **Manufacture the bulk synthetically** (JA-TTS over commercial-safe parallel text). Validated: *"differences between synthetic and authentic speech have negligible impact on [ST] quality"* (arXiv 2602.21646).

---

## 1. Recommended shortlist

**A. ST corpus to fine-tune JA→EN:**
1. **CoVoST2 `ja_en`** (`fixie-ai/covost2` audio-embedded) — exact task, but **~1.1k train / 635 val / 684 test**, **CC-BY-NC-4.0** → **eval/anchor only**.
2. **FLEURS `ja_jp`** (`google/fleurs`) — **CC-BY-4.0**, ~2.3k train (~10h); derive JA→EN by FLoRes-id join with `en_us`. **Only commercial real JA→EN ST signal.**

**B. Business/meeting parallel text:** **BSD — Business Scene Dialogue** (`ryo0634/bsd_ja_en`) — 24,171 sentences / 808 dialogues with speaker names + scene tags (face-to-face, phone, **meeting**, presentation, training) + original-language flags. **CC-BY-NC-SA-4.0** → demo + **template/few-shot source** for the CS generator; not shipped weights.

**C. Commercial-safe bulk parallel text:** **JESC** (`nntsuzu/JESC`) — **2.8M** conversational subtitle pairs, **CC-BY-SA-4.0** (treat as SA despite HF `cc-by-4.0` tag). Workhorse for synthetic ST + CS text + MT.

**Volumes:** real anchor = FLEURS ja_jp (~2.3k) + CoVoST2 ja_en (~1.1k, eval); synthetic JA→EN ST = JA-TTS of JESC+Tatoeba ≈ **50k–150k**; business flavor = BSD 20k (demo) or LLM-synthesized (commercial); reverse EN→JA = CoVoST2 en_ja (289k, NC, demo) or EN-TTS of parallel (commercial).

---

## 2. Comparison table

| Dataset | Type | Direction | Size | Domain | License (commercial? redistribute?) | HF ID | Notes |
|---|---|---|---|---|---|---|---|
| **CoVoST2 ja_en** | ST | JA audio→EN text | ~1.1k/635/684 | CV read | **CC-BY-NC-4.0** ❌/✅(NC) | `fixie-ai/covost2` (`ja_en`); `facebook/covost2` | Only real JA→EN ST; **tiny**. Eval only. |
| **CoVoST2 en_ja** | ST | EN audio→JA text | 289k/15.5k/15.5k | CV read | **CC-BY-NC-4.0** ❌ | `facebook/covost2` (`en_ja`) | Big, reverse direction, NC. |
| **FLEURS ja_jp** | ASR+ST (n-way) | JA audio (+EN via FLoRes) | ~2.3k/266/650 (~10h) | Read (FLoRes) | **CC-BY-4.0** ✅✅ | `google/fleurs` (`ja_jp`) | **Only commercial real JA→EN signal.** |
| **GigaST** | ST | EN→Zh/De only | 10,000h | multi | CC-BY-NC-4.0 | — | **No Japanese.** |
| **MuST-C** | ST | EN audio→JA text | hundreds h | TED | CC-BY-NC-ND ❌ | — | Only EN→JA; NC-ND. |
| **BSD** | text | JA↔EN | 24,171 sent / 808 dlg | **Business: meetings, calls, presentations** | **CC-BY-NC-SA-4.0** ❌/✅(NC,SA) | `ryo0634/bsd_ja_en`; GH `tsuruoka-lab/BSD` | **Best domain match.** Speaker+scene+original_language. Demo/template only. |
| **JESC** | text | JA↔EN | **2.8M** | **Conversational** (subtitles) | **CC-BY-SA-4.0** ✅✅(SA) | `nntsuzu/JESC` | **Best commercial-safe bulk.** HF mislabels cc-by-4.0; treat as SA. |
| **JParaCrawl v3.0** | text | JA↔EN | ~22M | Web | **Research-only — ❌ commercial** (bans selling translators trained on it) | NTT site; `Verah/JParaCrawl-Filtered-...` | **Known gotcha.** Avoid for deployment. |
| **KFTT** | text | JA↔EN | ~440k | **Formal** Wikipedia | **CC-BY-SA-3.0** ✅✅(SA) | OPUS `KFTT` | Commercial OK; formal domain (low meeting relevance). |
| **OPUS-100 en-ja** | text | JA↔EN | ~1.0M | mixed | **"unknown"** ⚠️ | `Hoshikuzu/opus-100-en-ja` | License-murky; not commercial-safe by default. |
| **OpenSubtitles ja-en** | text | JA↔EN | ~1M+ | subtitles | **"unknown" / not commercial** ❌ | `Helsinki-NLP/open_subtitles` | Subtitle copyright unclear → avoid; JESC covers same domain legally. |
| **Tatoeba ja-en** | text | JA↔EN | ~200k+ | everyday sentences | **CC-BY-2.0** ✅✅ | `Helsinki-NLP/tatoeba` | Clean conversational seed. |
| **WikiMatrix en-ja** | text | JA↔EN | ~100k–400k | Wikipedia (mined) | **CC-BY-SA-4.0** ✅✅(SA) | `BackpropBuff/WikiMatrix.en-ja` (`mit` tag wrong) | Noisier alignment. |
| **TED2020 ja-en** | text | JA↔EN | hundreds k | TED subtitles | CC-BY-NC-ND ❌ | OPUS `TED2020` | NC-ND. |

---

## 3. Per-dataset details (highlights)

### CoVoST2 — `fixie-ai/covost2` (audio embedded; easiest)
```python
ds = load_dataset("fixie-ai/covost2", "ja_en", split="test")  # JA audio -> EN translation; cols: audio, sentence(JA), translation(EN)
```
- ja_en train **1.1k** / val 635 / test 684 — the headline gotcha (Japanese is among the smallest source langs). `facebook/covost2` does NOT bundle audio (needs Common Voice JA on disk); `fixie-ai` embeds it. CC-BY-NC-4.0.

### FLEURS — `google/fleurs` (commercial real JA speech)
```python
fl_ja = load_dataset("google/fleurs", "ja_jp", split="train")  # JA audio + JA text
fl_en = load_dataset("google/fleurs", "en_us", split="train")  # EN text for same FLoRes ids
# build {flores_id -> en_text}, attach to fl_ja by id  → (JA audio, EN text)
```
- The FLEURS↔FLoRes id join is the only fiddly part — verify the mapping.

### BSD — `ryo0634/bsd_ja_en` (business domain, NC)
```python
ds = load_dataset("ryo0634/bsd_ja_en", split="train")  # en_sentence, ja_sentence, tag, en_speaker, ja_speaker, original_language
```
- License: **CC-BY-NC-SA-4.0** (README: "NonCommercial-ShareAlike"). 20k/2.05k/2.12k. Scene tags map directly to meeting scenarios; `original_language` tells which side is human-authored (use as source to avoid translationese). **Use as few-shot exemplars/templates** even on the commercial path (learning *patterns* ≠ redistributing rows — get a legal read).

### JESC — `nntsuzu/JESC` (commercial-safe bulk)
```python
ds = load_dataset("nntsuzu/JESC", split="train")  # ds[0]["translation"] -> {"en":..., "ja":...}
```
- Stanford says **CC-BY-SA-4.0**; HF tag says cc-by-4.0 → **assume SA**. 2.8M colloquial pairs; filter by length ratio + language-id (noisy tail).

### JParaCrawl v3.0 — the gotcha
- NTT terms: *"not available for commercial use, including the sale of translators trained using this data."* ~22M pairs. **Do not use for commercial weights.**

### Tatoeba — `Helsinki-NLP/tatoeba` (`lang1="en", lang2="ja"`), CC-BY-2.0, short everyday sentences — good synthetic-ST seed.

---

## 4. Licensing posture

**🟢 Ship-safe:** FLEURS (CC-BY-4.0, only commercial *real* JA speech), JESC (CC-BY-SA*), Tatoeba (CC-BY-2.0), KFTT (CC-BY-SA-3.0, formal), WikiMatrix (CC-BY-SA-4.0, noisy). *ShareAlike attaches to redistributed *datasets*, not (by consensus) to model weights — but license a released derived dataset SA-compatibly + attribute.

**🔴 Don't ship weights:** CoVoST2 ja_en/en_ja (NC — eval/demo), BSD (NC — template/demo), JParaCrawl (bans commercial), OpenSubtitles/OPUS-100 (unknown), TED2020/MuST-C (NC-ND), GigaST (no JA).

**Model license:** LFM Open License v1.0 caps free commercial use at **<$10M revenue**; above → contact Liquid AI.

---

## 5. How to use — recipe to build JA→EN ST training data

The model's task is set by a **system message** (`"Perform ASR."`, `"Perform TTS..."`). An ST task would be invoked by a system prompt like `"Perform speech translation to english."` — **🚩 UNVERIFIED** (no public LFM doc names an ST prompt; test candidates or *define+train* your own).

1. **Real anchor:** FLEURS ja_jp (commercial, FLoRes-join) ~2.3k; CoVoST2 ja_en (NC) → hold **test (684)** as real JA→EN ST eval.
2. **Synthetic bulk (commercial):** take JESC + Tatoeba (+ a little KFTT/WikiMatrix), filter (length ratio, language-id, ≤25s); **synthesize JA audio** (multi-speaker TTS, varied prosody, light noise/RIR); emit `system=ST prompt / user=synthetic JA audio / assistant=EN text`. Target 50k–150k; oversample conversational (JESC/Tatoeba) over formal ~3:1. (Reverse EN→JA symmetric.)
3. **Business domain:** demo = add BSD (TTS the source side); **commercial = don't train on BSD** — use its scene tags as **few-shot exemplars** to have an LLM generate *original* business dialogues → translate (commercial MT) → TTS → owned, clean ST data (~5–20k).
4. **Feed the CS generator:** BSD `conversation` arrays (speaker, scene, original_language) = ideal turn-taking/who-speaks-which-language **templates**; JESC/Tatoeba = commercial **content**; keep aligned EN/JA per turn so the same data trains ASR, ST, and MT consistently.

| Bucket | Source | Volume | License path |
|---|---|---|---|
| Real ST anchor | FLEURS ja_jp (+CoVoST2 eval) | ~2–3k (+~1k eval) | Commercial / NC eval |
| Synthetic ST JA→EN | JESC+Tatoeba via JA-TTS | 50k–150k | Commercial |
| Business ST | BSD (demo) / LLM-gen+TTS (commercial) | 5k–20k | NC / Commercial |
| Reverse EN→JA | CoVoST2 en_ja (demo) / EN-TTS (commercial) | 20k–50k | NC / Commercial |

---

## 6. Sources & uncertainties

**Sources:** CoVoST2 https://huggingface.co/datasets/facebook/covost2 , https://huggingface.co/datasets/fixie-ai/covost2 , https://arxiv.org/abs/2007.10310 · FLEURS https://huggingface.co/datasets/google/fleurs , https://arxiv.org/abs/2205.12446 · BSD https://huggingface.co/datasets/ryo0634/bsd_ja_en , https://github.com/tsuruoka-lab/BSD , https://arxiv.org/pdf/2008.01940 · JESC https://nlp.stanford.edu/projects/jesc/ , https://huggingface.co/datasets/nntsuzu/JESC , https://arxiv.org/abs/1710.10639 · JParaCrawl https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/ · KFTT https://www.phontron.com/kftt/ · Tatoeba https://huggingface.co/datasets/Helsinki-NLP/tatoeba · WikiMatrix https://huggingface.co/datasets/BackpropBuff/WikiMatrix.en-ja , https://arxiv.org/abs/1907.05791 · GigaST https://st-benchmark.github.io/resources/GigaST.html · synthetic-ST validity arXiv 2602.21646 · SpeechLLM ST arXiv 2402.12025

**Uncertainties:** 🚩 LFM ST system-prompt string UNVERIFIED (test or define+train). JESC license discrepancy (treat as CC-BY-SA-4.0). WikiMatrix re-host `mit` tag wrong (upstream CC-BY-SA-4.0). Tatoeba ja-en exact count verify at load. CoVoST2 `fixie-ai` train coverage — confirm ja_en train rows load. ShareAlike→weights legally unsettled.
