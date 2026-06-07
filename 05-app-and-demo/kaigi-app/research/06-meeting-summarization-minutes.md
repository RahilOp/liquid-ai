# Datasets — Meeting/Dialogue Summarization & Minutes (text SFT)

*For optionally fine-tuning a small on-device text LFM (**LFM2.5-1.2B-JP** / **LFM2.5-1.2B-Thinking**) to generate confidential bilingual (JP↔EN) meeting minutes — summary + action items — from a possibly code-switched transcript. Verified against HF cards/papers (2026-06-06).*

---

## 0. TL;DR licensing reality check

The famous meeting/dialogue summarization sets are **mostly non-commercial / research-only** — the single most important finding:

| Dataset | License | Commercial-deployable? |
|---|---|---|
| **AMI / ICSI** | CC BY 4.0 | ✅ **YES** |
| **QMSum** (`pszemraj` Apache-2.0 / orig MIT) | MIT / Apache-2.0 | ✅ **YES** |
| **Aya** (multilingual, incl. JA) | Apache-2.0 | ✅ **YES** |
| **dolly-15k-ja** | CC BY-SA 3.0 | ✅ **YES** (share-alike) |
| SAMSum | CC BY-**NC-ND** 4.0 | ❌ NC **and** No-Derivatives |
| DialogSum | CC BY-**NC**-SA 4.0 | ❌ NC |
| MeetingBank | CC BY-**NC**-SA 4.0 | ❌ NC |
| XLSum (incl. xlsum_ja) | CC BY-**NC**-SA 4.0 | ❌ NC |
| MediaSum | **Research-only** (NPR/CNN ToS) | ❌ |
| ELITR Minuting Corpus | CC BY-SA 4.0 *(per LINDAT — verify)* | ⚠️ likely YES |
| ichikara-instruction | CC BY-NC-SA 4.0 (free) / **paid** commercial | ❌ free tier NC |

**Consequence:** the most domain-relevant corpora (DialogSum, SAMSum, MeetingBank, MediaSum, ichikara) are **off-limits for shipped weights** → internal experimentation/ablation only. Shipped training = ✅ rows + **synthetic data you generate yourself**.

**LFM license is permissive enough:** LFM Open License v1.0 (Apache-2.0-based, no copyleft, proprietary fine-tunes allowed, free commercial **<$10M revenue**). A hackathon team is clearly in the free tier.

---

## 1. Recommended shortlist

### (a) Commercial-safe meeting-summ seed → **AMI + QMSum**
- **AMI** (`knkarthick/AMI`, **CC BY 4.0**) — real multi-party meetings with human extractive + abstractive summaries. Small (~137 meetings) but genuine meeting→summary, clean license. Gated (click-through).
- **QMSum** (`pszemraj/qmsum-cleaned`, **Apache-2.0**) — **query-based** meeting summarization (AMI+ICSI+committee), ~1.8k query-summary pairs. Maps to "extract action items / summarize topic X."
- Optionally **ICSI** (CC BY 4.0) for variety.
- **Use:** all of AMI + all of QMSum (a few thousand examples) — enough for *format/style*, not enough alone → pair with synthetic. Chunk long transcripts to context window.

### (b) Best Japanese option → **Aya (JA) + dolly-15k-ja** + synthetic
There is **no clean, commercial, Japanese meeting-minutes dataset** (JA summ sets are NC or paid). So:
- **Aya Dataset** (`CohereLabs/aya_dataset`, **Apache-2.0**) — 204K human-written multilingual incl. **Japanese (`jpn`)**, summarization-style prompts. Commercial-safe.
- **databricks-dolly-15k-ja** (`kunishou/databricks-dolly-15k-ja`, **CC BY-SA 3.0**) — 15K JA instructions incl. `summarization`. Commercial OK (share-alike), machine-translated (uneven JA).
- For meeting-minutes in JA specifically → **synthesize**.

### (c) Synthetic-minutes plan (the workhorse)
Confidential minutes are inherently private + need code-switched JP↔EN transcript → structured JA/bilingual minutes, which **no public dataset covers**. So synthetic SFT is primary:
1. **Transcript → minutes** over your own transcripts (or AMI/ICSI, CC BY 4.0): prompt a teacher LLM for structured minutes (decisions / action items / owners / due dates) in JA + bilingual.
2. **Synthetic transcript generation:** teacher LLM fabricates realistic code-switched JP↔EN meeting transcripts (varied domains, ASR-style noise) → gold minutes. Fixes the code-switch + ASR-noise gap.

**Crux — the binding constraint is the teacher model's ToS, not copyright.** Machine-generated text generally isn't copyrightable, so outputs usually carry no blocking copyright — but the **provider's ToS can forbid training a competing model** (OpenAI/Anthropic/Google). **Safe route: generate with an open-weight commercially-licensed teacher run locally** (Apache/Llama-community/Qwen, or a larger LFM). Given the *confidential* framing, local generation is doubly attractive (no transcript leaves device/VPC). Log teacher model+version+license per batch.

**Mix (~3–8K examples is plenty for LoRA on 1.2B):** 55–70% **synthetic** transcript→minutes (JA + bilingual + code-switched) · 15–25% **AMI + QMSum** (real structure, EN) · 10–20% **Aya-JA + dolly-15k-ja** (keep JA instruction-following sharp). (DialogSum/SAMSum sanity-checks only — never in shipped weights.)

---

## 2. Comparison table

| Dataset | Lang | Size | Domain | License (commercial? redistribute?) | HF ID | Notes |
|---|---|---|---|---|---|---|
| **AMI** | EN | ~137 meetings | **Meeting** (extractive+abstractive summ) | **CC BY 4.0** ✅✅ | `knkarthick/AMI` (🔒 gated) | Best clean meeting→summary. Small, long transcripts. |
| **ICSI** | EN | 75 meetings (~72h) | **Meeting** (academic) | **CC BY 4.0** ✅✅ | `guokan-shang/ami-and-icsi-corpora` (JSON) | Companion to AMI. |
| **QMSum** | EN | ~1.8K query–summ | **Meeting, query-based** | orig MIT; `pszemraj` **Apache-2.0** ✅✅ | `pszemraj/qmsum-cleaned`; `MocktaiLEngineer/qmsum-processed` | Maps to action-item extraction. |
| **MeetingBank** | EN | 1,366 meetings → ~6.9K rows | **Meeting MINUTES** (city councils) | **CC BY-NC-SA 4.0** ❌ NC | `huuuyeah/meetingbank` | Real transcript→minutes but NC. Experiments only. Long-context (~28K tok). |
| **ELITR Minuting Corpus** | EN, CS | 113 EN + 53 CS meetings (~400 minutes) | **Actual MINUTES** | **CC BY-SA 4.0** *(per LINDAT — verify)* ⚠️✅✅(SA) | LINDAT `11234/1-4692` | **Only public real-minutes corpus with plausibly-commercial license.** Multiple minutes/meeting. No JA. |
| **MediaSum** | EN | 463,596 interviews | Interview/dialogue (NPR/CNN) | **Research-only** ❌ | `ccdv/mediasum` | Large, high-quality, but research-only. |
| **DialogSum** | EN | 13,460 dialogues | Daily-life dialogue summ | **CC BY-NC-SA 4.0** ❌ NC | `knkarthick/dialogsum` | Popular; NC (the "MIT" rumor is false). |
| **SAMSum** | EN | ~16.4K chats | Messenger dialogue | **CC BY-NC-ND 4.0** ❌ NC+ND | `knkarthick/samsum` | NC **and** No-Derivatives. Experiments only. |
| **XLSum (JA)** | JA (+44) | JA 7,113/889/889 | **News** summ (BBC) | **CC BY-NC-SA 4.0** ❌ NC | `csebuetnlp/xlsum`; `mkshing/xlsum_ja` | NC + off-domain. |
| **ichikara-instruction** | JA | ~10K+ | JA instruction (incl. summ) | **CC BY-NC-SA 4.0** free / **paid** commercial ❌ | `kinokokoro/ichikara-instruction-003` | High-quality human JA; commercial = paid RIKEN license. |
| **databricks-dolly-15k-ja** | JA | ~15K | JA instruction (incl. `summarization`) | **CC BY-SA 3.0** ✅✅(SA) | `kunishou/databricks-dolly-15k-ja` | Commercial-clean; MT-uneven JA. |
| **Aya Dataset** | 65 langs incl. **JA** | 204K | Multilingual instruction (incl. summ) | **Apache-2.0** ✅✅ | `CohereLabs/aya_dataset` | Human-written; filter `language=="jpn"`. Best clean JA instruction source. |

---

## 3. Per-dataset details (highlights)

### AMI — `knkarthick/AMI` (CC BY 4.0) ✅
```python
ami = load_dataset("knkarthick/AMI")  # gated: accept terms + login. splits train/val/test; dialogue/summary
```
- Gated click-through; long transcripts (segment/truncate); scripted product-design meetings (low lexical diversity); attribution required.

### QMSum — `pszemraj/qmsum-cleaned` (Apache-2.0) ✅
```python
qmsum = load_dataset("pszemraj/qmsum-cleaned")              # configs "default" (query-prefixed) / "no-prefix"
# cols: id, pid, input, output, input_token_count, output_token_count
```
- `default` embeds the query in `input` — strip/repurpose to your template, or use `no-prefix`. Great for query-conditioned ("give me the action items").

### ELITR Minuting Corpus — LINDAT `11234/1-4692` ⚠️
- Not a `load_dataset` repo — download from LINDAT (handle `http://hdl.handle.net/11234/1-4692`). Transcripts + minutes (recordings excluded for privacy). **License: CC BY-SA 4.0 per LINDAT catalog — verify the badge on the download page.** EN + Czech only (no JA). The only public *real, multiply-annotated* minutes corpus.

### MeetingBank — `huuuyeah/meetingbank` (CC BY-NC-SA 4.0) ❌NC — experiments only
```python
mb = load_dataset("huuuyeah/meetingbank")  # transcript, summary, uid, id; long context ~28K tok
```

### Aya — `CohereLabs/aya_dataset` (Apache-2.0) ✅
```python
aya = load_dataset("CohereLabs/aya_dataset")
aya_ja = aya["train"].filter(lambda r: r["language"] == "jpn")  # cols: inputs, targets, language
```

### dolly-15k-ja — `kunishou/databricks-dolly-15k-ja` (CC BY-SA 3.0) ✅ — filter the `summarization` category; spot-check JA (MT).

---

## 4. Licensing posture

**🟢 Ship-safe:** AMI, ICSI (CC BY 4.0); QMSum (MIT/Apache); Aya (Apache); dolly-15k-ja (CC BY-SA 3.0, SA); ELITR (*likely* CC BY-SA 4.0 — verify); **your own synthetic** (open-weight teacher); LFM2.5 bases (LFM Open, free <$10M).

**🔴 Research-only / NC (experiments & eval only):** MediaSum, SAMSum (NC+ND), DialogSum (NC), MeetingBank (NC), XLSum/xlsum_ja (NC), ichikara free (NC; commercial=paid), TWEETSUMM (X ToS).

> **Rule:** ship-train only on 🟢 + synthetic. NC sets are fine to *measure* quality (held-out eval/ablation) but exclude from shipped weights for an audit-friendly "commercially deployable" claim.

---

## 5. How to use — small SFT recipe (LoRA) for bilingual minutes

**Goal:** transcript (possibly JP↔EN code-switched, ASR-style) → structured minutes (summary + decisions + action items w/ owner/due) in JA / bilingual.

**Fixed I/O schema** (so a small model latches on):
- **Input:** transcript + short instruction, e.g. `"次の会議の文字起こしから議事録を作成してください。要約・決定事項・アクションアイテム（担当者・期限）を含めてください。"`
- **Output:** fixed structure (Markdown or JSON — JSON if code parses action items):
  ```
  ## 要約 / Summary
  ## 決定事項 / Decisions
  ## アクションアイテム / Action Items
  - [担当: 田中] ... (期限: 2026-07-01)
  ```
  For `-Thinking`, keep visible minutes in the answer; don't train reasoning into the minutes body.

**Data assembly (~3–8K pairs):** 55–70% synthetic (local open-weight teacher; Path A = teacher over AMI/ICSI/own transcripts → gold minutes; Path B = teacher fabricates code-switched JP↔EN transcripts w/ ASR noise → gold minutes) · 15–25% AMI+QMSum reformatted to your schema · 10–20% Aya-JA + dolly-ja summarization.

**Training:** base = `LFM2.5-1.2B-JP` (best JA prior) or `-Thinking`; **use the native chat template**; LoRA (r=16–32, α=32, dropout 0.05) or QLoRA 4-bit; 2–3 epochs, LR 1e-4–2e-4 cosine, warmup 3%, eff. batch 16–32, bf16; **mask loss on prompt tokens**; set seq length to model max (read `config.json` `max_position_embeddings`); **chunk long transcripts** (map-reduce: summarize segments → final minute) and train on the same chunking you deploy.

**Eval (may use NC since measurement-only):** ROUGE-L / BERTScore + task checks (action-item recall, owner/date accuracy, JA fluency, bilingual consistency) + human faithfulness rubric (no hallucinated decisions — critical for confidential minutes). Include negative/edge examples (empty meeting, no decisions). Keep all generation/training **on-device/in-VPC** (confidentiality).

---

## 6. Sources & uncertainties

**Sources:** LFM license https://www.liquid.ai/lfm-license · AMI https://huggingface.co/datasets/knkarthick/AMI , https://groups.inf.ed.ac.uk/ami/corpus/ · QMSum https://huggingface.co/datasets/pszemraj/qmsum-cleaned · MeetingBank https://huggingface.co/datasets/huuuyeah/meetingbank , https://arxiv.org/abs/2305.17529 · ELITR https://lindat.mff.cuni.cz/repository/handle/11234/1-4692 , https://aclanthology.org/2022.lrec-1.340/ · MediaSum https://huggingface.co/datasets/ccdv/mediasum , https://arxiv.org/abs/2103.06410 · DialogSum https://huggingface.co/datasets/knkarthick/dialogsum · SAMSum https://huggingface.co/datasets/knkarthick/samsum · XLSum https://huggingface.co/datasets/csebuetnlp/xlsum · ichikara https://huggingface.co/datasets/kinokokoro/ichikara-instruction-003 , http://liat-aip.sakura.ne.jp/wp/ · dolly-15k-ja https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja · Aya https://huggingface.co/datasets/CohereLabs/aya_dataset , https://arxiv.org/abs/2402.06619

**Uncertainties:** ELITR license string not byte-confirmed (LINDAT catalogs CC BY-SA 4.0 — verify badge). LFM2.5-1.2B-JP context length not exposed in search — read `config.json`. ichikara commercial terms paraphrased (free=NC, commercial=paid RIKEN). **Teacher-LLM ToS is the real gating factor** for synthetic data → use a local open-weight/LFM teacher; get legal sign-off for a shipped product. `knkarthick/*` are community re-uploads — cite canonical sources for an audit.
