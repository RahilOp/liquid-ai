# Datasets — JP↔EN Code-Switching (speech + text)

> Scope: real/published code-switching (CS) data to **seed, augment, and evaluate** a fine-tune of `LiquidAI/LFM2.5-Audio-1.5B-JP` into an on-device JP↔EN code-switching meeting assistant. Consumed as **audio+transcript pairs**. Commercial deployability matters (product ships under the permissive LFM Open License), so **license clarity (commercial use + redistribution of derivatives)** is a first-class ranking axis.
>
> **Headline finding:** There is essentially **no openly-licensed, commercial-grade, real-voice JP↔EN CS speech corpus.** Every JP↔EN CS speech resource that exists is either (a) **non-commercial / research-only** (JECS, CS-FLEURS ja-en, SwitchLingua), (b) **not real speech** (synthetic TTS), or (c) **not English** (KO-JA). This **validates the synthetic-first strategy**: real CS data should be used almost exclusively as a **small held-out EVAL gold set**, plus as a **prosody/style reference**, not as bulk training data. Bulk training = your synthetic pipeline. The commercially-clean assets are mostly **TEXT** (to drive synthetic generation) and **CS-technique transfer** from other language pairs.

Dates verified June 2026. License strings are quoted from the primary source (HF card tag / paper / repo) where possible; unverified items are flagged explicitly.

---

## 1. Recommended shortlist

### (a) Training-seed / augmentation
| Priority | Asset | Why | How much to use |
|---|---|---|---|
| **1** | **Your own synthetic CS audio** (LLM transcripts → bilingual TTS), optionally using the **UniCoM/SWORDS** pipeline (`github.com/sanghyang00/unicom`, code; corpus CC-BY-4.0) as a recipe | Only scalable source of **commercially-clean, native-script JP↔EN** audio in business/meeting domain. UniCoM's "SWORDS" (POS-aware synonym/translation substitution) is a published, reusable algorithm for generating natural intra-sentential switches. | **Bulk of training (80–95%)**. |
| **2** | **CS TEXT corpora to drive transcript generation** — primarily **JESC** (Japanese-English **parallel** subtitles, CC-BY-SA-4.0, 2.8M pairs) for colloquial bilingual phrasing + loanword lists; **SwitchLingua_text** (NC — see caveat) only as *prompt inspiration*, not redistributed | Real bilingual phrasing/loanword inventories make synthetic switches realistic. JESC is fully commercial-OK. | JESC freely; SwitchLingua_text only to inspect patterns (do **not** ship its rows). |
| **3** | **ASCEND** (`CAiRE/ASCEND`, zh-en, CC-BY-SA-4.0, 10.6h spontaneous conversational) + **TALCS** (zh-en, ~587h, license-unverified) | **CS-technique transfer / baseline pretraining** for the *acoustics of switching* (the model only knows EN or JP, not switching). ASCEND is the cleanest license; conversational domain matches. | ASCEND fully (small); TALCS only if its license verifies as permissive. Mix in as auxiliary CS-acoustic data. |
| **4** | **CS-FLEURS ja-en `xtts` split** (`byan/cs-fleurs`, **CC-BY-NC-4.0**) | A ready-made synthetic JP-EN CS set — but **NC + romanized JP**. Use as a **dev/sanity set or augmentation if you stay non-commercial**; otherwise replace with your own synthetic. | Small; **dev only** under commercial constraints. |

### (b) Evaluation (real-audio gold)
| Priority | Asset | Why | How much |
|---|---|---|---|
| **1** | **JECS** (NAIST/Takamichi, ~2.5h, 1 bilingual voice actor) | The **only real-voice JP↔EN CS speech with parallel JP/EN/CS utterances and native script**. License is research/non-commercial + **no redistribution** → perfect as an **internal, non-shipped EVAL gold set**. | **Primary JP-EN CS eval** (hold out entirely; never train). |
| **2** | **CS-FLEURS ja-en read-test if present, else xtts** (`byan/cs-fleurs`) | Standardized CS-ASR/ST benchmark numbers; comparable to literature. NC + romanized JP (WER caveat). | Secondary eval (report separately due to romanization). |
| **3** | **ASCEND test split** (zh-en) | Cross-pair generalization check of switching robustness; clean license to publish numbers. | Auxiliary eval. |
| **4** | **SwitchLingua_audio** ja-en subset (gated, NC) | Real multi-ethnic conversational CS incl. JP-EN; good *additional* real eval **if** you accept gated NC terms for internal eval only. | Optional internal eval. |

> **Rule of thumb:** Treat **all** real JP↔EN CS speech (JECS, CS-FLEURS, SwitchLingua) as **eval/reference**, not as shippable training data, because of license (NC / no-redistribute) and tiny size. Ship the model trained on **your synthetic data + permissive text-derived transcripts + (optionally) permissively-licensed other-pair CS acoustics**.

---

## 2. Comparison table

**JP↔EN CS speech (the target gap)**

| Dataset | Languages | Hours / Rows | Domain | License (commercial? redistribute?) | Format | HF ID / Download | Notes |
|---|---|---|---|---|---|---|---|
| **JECS** (NAIST / Takamichi) | JP, EN, **JP↔EN CS** (parallel) | **~2.5 h**, 1 speaker | Acted read speech; neutral + emotion (sad/happy/angry) + word-emphasis | **Audio: research/non-commercial only; redistribution PROHIBITED**. **Text: CC BY 3.0**. ❌ commercial ❌ redistribute | 24 kHz WAV + transcripts, ~0.4 GB zip | **Not on HF.** Google Drive via [project page](https://sites.google.com/site/shinnosuketakamichi/research-topics/jecs_corpus) | Single bilingual voice actor → low speaker diversity. **Native JP script** (not romanized). Best as internal eval gold. |
| **CS-FLEURS** ja-en split | jpn↔eng (+112 other pairs) | 300 h total; ja-en is a slice of the **xtts** subset | Read sentences (FLEURS-derived), **synthetic** CS | **CC-BY-NC-4.0** ❌ commercial ✅ redistribute-NC | Parquet/audio under `read/`,`xtts/`,`mms/` | [`byan/cs-fleurs`](https://huggingface.co/datasets/byan/cs-fleurs) | **JP text is ROMANIZED** (XTTS-v2 requirement) → breaks native-script WER. **Viewer shows 0 rows → NOT a clean `load_dataset`; download folder files manually.** |
| **SwitchLingua_audio** (NeurIPS'25) | 12 langs incl. **ja↔en** | **80+ h**, 174 speakers, 63 ethnicities | **Multi-domain conversational** | **CC-BY-NC-4.0**, **GATED**. ❌ commercial ❌ redistribute | audiofolder | [`Shelton1013/SwitchLingua_audio`](https://huggingface.co/datasets/Shelton1013/SwitchLingua_audio) (gated) | GitHub code is MIT, but the **DATA card is CC-BY-NC-4.0 — the data governs use.** JP-EN per-pair count not published. |
| **NAIST Speech-Chain JP-EN CS** | JP↔EN CS | ~280k synthetic utts (bilingual TTS) | Read/synthetic | **Not openly released** ❌ | — | papers only | Technique (machine speech chain) reusable; corpus is not. |
| **thetaone-ai KO-JA CS** | **ko↔ja (NO English)** | n<1K rows | Read sentence-level CS | **Apache-2.0** ✅✅ | Parquet | [`thetaone-ai/Korean-Japanese-Code-Switching-Speech`](https://huggingface.co/datasets/thetaone-ai/Korean-Japanese-Code-Switching-Speech) | Not your pair (no EN). |

**Canonical CS speech, other pairs (technique transfer + baselines)**

| Dataset | Languages | Hours | Domain | License | HF ID / Download | Notes |
|---|---|---|---|---|---|---|
| **ASCEND** | zh↔en | **10.62 h** | **Spontaneous conversational** (HK) | **CC-BY-SA-4.0** ✅ (share-alike) | [`CAiRE/ASCEND`](https://huggingface.co/datasets/CAiRE/ASCEND) | **Cleanest license + best domain match.** One-line load. |
| **TALCS** | zh↔en | **~587 h** | Online English teaching | **"permissive" but NOT named — binding terms = 100TAL ToS — UNVERIFIED** ⚠️ | [ai.100tal.com/dataset](https://ai.100tal.com/dataset) | Largest open zh-en CS. **Verify license before commercial use.** |
| **CS-Dialogue** | zh↔en | **104.02 h** | **Spontaneous dialogue** | **CC-BY-NC-SA-4.0**, GATED ❌ commercial | [`BAAI/CS-Dialogue`](https://huggingface.co/datasets/BAAI/CS-Dialogue) (gated) | Great domain, NC → eval/technique only. |
| **DOTA-ME-CS** | zh↔en | **18.54 h** | Daily read; AI-augmented | License not stated ⚠️ | [paper 2501.12122](https://arxiv.org/abs/2501.12122) | Treat research-only until verified. |
| **SEAME** (LDC2015S04) | zh↔en | **63 h** base | Spontaneous interview | **LDC — paid / ToS-encumbered** ❌ | [LDC catalog](https://catalog.ldc.upenn.edu/) | Canonical CS-ASR benchmark, paywalled. |
| **ArzEn / ArzEn-ST** | arz↔en | **~12 h** | Spontaneous interviews | "publicly available"; license not stated ⚠️ | [project site](https://sites.google.com/view/arzen-corpus/home) | License vague; request from authors. |
| **Bangor Miami** | es↔en | **35 h** | Spontaneous conversation | **GPLv3+** ⚠️ copyleft | [bangortalk.org.uk](https://bangortalk.org.uk/) | GPL on data is product-hostile. |
| **KasaSpeech** (en↔Twi) | en↔tw | 10K–100K rows | Ghana CS ASR | **MIT** ✅✅, GATED | [`Kennethdot/Ghana_English-Twi_Code-switching_ASR`](https://huggingface.co/datasets/Kennethdot/Ghana_English-Twi_Code-switching_ASR) | MIT (rare clean CS license); wrong pair; methodology reference. |

**Pre-built SYNTHETIC CS data**

| Dataset | Languages | License | HF ID | Notes |
|---|---|---|---|---|
| **CS-FLEURS** | 113 pairs incl. **ja-en** | **CC-BY-NC-4.0** | [`byan/cs-fleurs`](https://huggingface.co/datasets/byan/cs-fleurs) | ja-en = synthetic (xtts), romanized JP, NC. |
| **UniCoM CS-FLEURS** (*different* dataset, same name) | **253 pairs, NO Japanese** | **CC-BY-4.0** ✅✅ (data) | code: [`github.com/sanghyang00/unicom`](https://github.com/sanghyang00/unicom) | No JP, but **SWORDS pipeline = reusable recipe**, commercial-clean. |
| **BrunoHays FLEURS-CS** | many incl. ja (concatenative) | cc-by-4.0/other ⚠️ | [`BrunoHays/fleurs_code_switching_test`](https://huggingface.co/datasets/BrunoHays/fleurs_code_switching_test) | Concatenation (not intra-sentential) → language-boundary eval. |
| **Resvand EN-Hindi synthetic** | en↔hi | **Apache-2.0** ✅✅ | [`Resvand/Code-Switching_English-Hindi_Synthetic`](https://huggingface.co/datasets/Resvand/Code-Switching_English-Hindi_Synthetic) | Wrong pair; clean-license synthetic template. |

**CS TEXT corpora (drive synthetic transcript generation)**

| Dataset | Languages | Size | License | HF ID / Source | Notes |
|---|---|---|---|---|---|
| **JESC** | ja‖en **parallel** (NOT CS) | **2.8M pairs** | **CC-BY-SA-4.0** ✅ | [`Hoshikuzu/JESC`](https://huggingface.co/datasets/Hoshikuzu/JESC) | Most useful **commercial-clean** JP-EN colloquial resource to seed an LLM CS generator. |
| **LinCE** | es-en, hi-en… **(no JP)** | 11 corpora | portal/mixed ⚠️ | [ritual.uh.edu/lince](https://ritual.uh.edu/lince/) | CS-LID/NER methodology reference. |
| **GLUECoS** | en-es, en-hi **(no JP)** | 6 tasks | code MIT; data via scripts ⚠️ | [`github.com/microsoft/GLUECoS`](https://github.com/microsoft/GLUECoS) | CS eval-design reference. |
| **Malikeh1375 tokenizer-robustness** | 16 variants incl. **en-ja CS** | n<1K | **CC-BY-4.0** ✅✅ | [`Malikeh1375/code-switching-tokenizer-robustness`](https://huggingface.co/datasets/Malikeh1375/code-switching-tokenizer-robustness) | Tiny but real + commercial-clean en-ja CS text. |
| **SwitchLingua_text** | 12 langs incl. **ja-en** | 420K total | **CC-BY-NC-4.0**, GATED ❌ | [`Shelton1013/SwitchLingua_text`](https://huggingface.co/datasets/Shelton1013/SwitchLingua_text) | Largest ja-en CS **text**, NC+gated → inspect patterns only. |

---

## 3. Per-dataset details (highlights)

### JECS (top JP-EN eval pick)
- **Access:** [project page](https://sites.google.com/site/shinnosuketakamichi/research-topics/jecs_corpus) → Google Drive (~0.4 GB), open download but **academic/non-commercial only, redistribution forbidden** (sample sharing of ~5 files only). Text = CC BY 3.0.
- **Content:** parallel JP / EN / **CS** utterances + word-emphasized variants + emotions, 24 kHz, **1 bilingual voice actor**, ~2.5 h.
- **Gotchas:** single speaker → content/style gold, not speaker-robustness; not redistributable → keep off shipped artifacts; native JP script (good).
- **Role:** internal **EVAL gold** for JP↔EN intra-sentential CS.

### CS-FLEURS (`byan/cs-fleurs`)
- **License:** `cc-by-nc-4.0`. ja-en lives under `xtts/` (synthetic). **JP transcripts ROMANIZED.**
- **Load:** Viewer shows 0 rows → use `snapshot_download("byan/cs-fleurs", repo_type="dataset", allow_patterns=["xtts/*ja*","xtts/*jpn*"])` then parse manually.

### ASCEND (`CAiRE/ASCEND`) — best-licensed other-pair CS
- **License:** `cc-by-sa-4.0`. Commercial OK with share-alike. `load_dataset("CAiRE/ASCEND")` (train/val/test 8:1:1). 10.62h spontaneous zh-en.
- **Role:** auxiliary **CS-acoustic training** + cross-pair eval.

### JESC (`Hoshikuzu/JESC`) — commercial-clean JP-EN text fuel
- CC-BY-SA-4.0, 2.8M JP‖EN parallel subtitle pairs (colloquial). Not code-switched, but the richest commercial-OK source of bilingual phrasing + loanword pairs to ground an LLM CS generator. `load_dataset("Hoshikuzu/JESC")`.

---

## 4. Licensing posture

**✅ Safe (commercial + redistribution):** JESC (CC-BY-SA), ASCEND (CC-BY-SA), UniCoM CS-FLEURS data (CC-BY-4.0, **no JP**), Malikeh1375 (CC-BY-4.0), KasaSpeech (MIT, wrong pair), Resvand EN-Hi (Apache), **your own synthetic**.

**⚠️ Unverified — check at source:** TALCS, DOTA-ME-CS, ArzEn, LinCE.

**❌ Research-only / NC / paid / no-redistribute (eval & technique only):** JECS, CS-FLEURS, SwitchLingua, CS-Dialogue, BSC CA-ES, SEAME (LDC), Bangor Miami (GPL).

> **Net:** only commercially-clean **JP-EN-relevant** assets are **text** (JESC, Malikeh1375) + the **UniCoM recipe** + **other-pair CS acoustics** (ASCEND; TALCS if verified). **No commercial real-voice JP↔EN CS speech exists** → synthetic is mandatory.

---

## 5. How to use in our pipeline

**A. EVAL gold:** JECS (primary, native script, internal-only — publish numbers not audio) + CS-FLEURS ja-en (secondary, romaji-normalized scoring) + ASCEND test (cross-pair, publishable) + **a tiny in-domain meeting eval recorded by your own bilingual colleagues** (the most decision-relevant, fully yours).

**B. Training:** bulk = **synthetic CS** (LLM CS transcripts → bilingual TTS), grounded in **JESC + Malikeh1375 en-ja**; optionally implement **SWORDS** (POS-aware translation substitution) for natural intra-sentential switches; shape toward **business/meeting** register. Optional small **ASCEND** auxiliary for CS-acoustic transfer (keep minority to avoid Mandarin phonotactic bias). Do **not** put JECS/CS-FLEURS/SwitchLingua/CS-Dialogue audio into shipped training.

**C. Loanword-vs-switch caveat (critical for JP↔EN):** define CS operationally as switching into **English orthography/phonology** vs **nativized katakana loanwords** (treated as Japanese). Prefer genuine matrix-frame switches over katakana-izing English. For eval, use token-level language-ID tags and report CS metrics conditioned on *true* switch points, plus a separate plain WER/CER. Romanized-JP datasets erase the katakana/Latin distinction → secondary eval only.

---

## 6. Sources & uncertainties

**Sources:** JECS https://sites.google.com/site/shinnosuketakamichi/research-topics/jecs_corpus · CS-FLEURS https://huggingface.co/datasets/byan/cs-fleurs , https://arxiv.org/abs/2509.14161 · SwitchLingua https://arxiv.org/html/2506.00087v1 , https://huggingface.co/datasets/Shelton1013/SwitchLingua_audio · UniCoM https://arxiv.org/abs/2508.15244 , https://github.com/sanghyang00/unicom · ASCEND https://huggingface.co/datasets/CAiRE/ASCEND , https://arxiv.org/abs/2112.06223 · TALCS https://arxiv.org/abs/2206.13135 , https://ai.100tal.com/dataset · CS-Dialogue https://huggingface.co/datasets/BAAI/CS-Dialogue , https://arxiv.org/abs/2502.18913 · SEAME https://catalog.ldc.upenn.edu/ · ArzEn https://sites.google.com/view/arzen-corpus/home · Bangor Miami https://bangortalk.org.uk/ · JESC https://huggingface.co/datasets/Hoshikuzu/JESC , https://web.stanford.edu/~jurafsky/pubs/jesc.pdf · LinCE https://ritual.uh.edu/lince/ · GLUECoS https://github.com/microsoft/GLUECoS · Malikeh1375 https://huggingface.co/datasets/Malikeh1375/code-switching-tokenizer-robustness

**Uncertainties:** TALCS license UNVERIFIED (check 100TAL ToS). DOTA-ME-CS/ArzEn licenses unresolved. CS-FLEURS not a clean `load_dataset` (snapshot_download xtts ja-en, parse manually); per-pair ja-en hours unpublished. SwitchLingua per-pair JP-EN counts unpublished; MIT(code) vs CC-BY-NC(data) discrepancy → data governs. NAIST 280k-utt corpus not released. **No dedicated openly-licensed real-voice JP↔EN CS speech corpus exists** despite targeted 2024–2026 searches — JECS (~2.5h, NC, no-redistribute) is the only real one. This is the gap the synthetic pipeline fills.
