# Datasets — Monolingual English ASR (replay + meeting-domain)

> A small, high-quality English ASR **replay** slice (~10–15% of the mix) to prevent **catastrophic forgetting of English**, ideally also matching the **meeting/business/conversational** domain so it doubles as domain adaptation. Build the slice from **CC0 + CC-BY-4.0** only. Verified against HF cards/papers (2026-06-06).

---

## 1. Recommended shortlist

| Role | Pick | Hours | License | Why |
|---|---|---|---|---|
| **Small commercial-clean replay** | **LibriSpeech** `train.clean.100` | **~10h** sampled | **CC-BY-4.0** | cleanest licensing, 16 kHz, canonical read-English anchor |
| **Accent/spontaneous diversity** | **Common Voice EN** (CC0 mirror) | **~3–5h** | **CC0** | crowd accent variety; most permissive |
| **Meeting-domain add-on (replay + adaptation)** | **AMI** `ihm` | **~8–15h** | **CC-BY-4.0** | ⭐ direct domain match (multi-party business meetings), clean license, one-line load |
| **Business/earnings flavor** | **Earnings-22** | **2–5h** | **CC-BY-SA-4.0** ⚠️ copyleft | real accented business speech |

**Concrete ~25h slice (~12% of a ~210h mix):** AMI `ihm` ~10h + LibriSpeech `train.clean.100` ~10h + Common Voice EN (CC0) ~3h + Earnings-22 ~2h (optional). **For zero copyleft:** drop Earnings-22 → 100% CC0 + CC-BY-4.0.

> **Why not GigaSpeech / SPGISpeech / TED-LIUM as backbone?** GigaSpeech + SPGISpeech are gated research/non-commercial (SPGISpeech also forbids redistribution); TED-LIUM 3 is CC-BY-NC-ND. All three are **blockers** for a commercial deploy.

---

## 2. Comparison table

| Dataset | Hours | Domain | License (commercial? redistribute?) | HF ID | Notes |
|---|---|---|---|---|---|
| **LibriSpeech** | ~960h (100/360/500 splits) | Read audiobooks | **CC-BY-4.0** ✅✅ | `openslr/librispeech_asr` | **Best replay backbone.** Use `train.clean.100`. |
| **Common Voice EN** | ~3,200h validated (v17) | Crowd read + accents | **CC0-1.0** ✅✅ | `mozilla-foundation/common_voice_17_0` (gated) / **CC0 mirror `fsicoli/common_voice_17_0`** | Most permissive. Official repo gated since Oct 2025. |
| **GigaSpeech** | 10,000h (XS=10h…XL) | Audiobooks+podcasts+YouTube | tag `apache-2.0` **but access = non-commercial research only**; SpeechColab doesn't own audio ⚠️ **RESTRICTED** | `speechcolab/gigaspeech` (gated) | License conflict (§4). Don't use commercially without clearance. |
| **People's Speech** | `clean` ≈12k h | Diverse web | `clean`/`dirty`=**CC-BY-4.0** ✅; `*_sa`=CC-BY-SA ⚠️ | `MLCommons/peoples_speech` | **Use only `clean`.** Transcripts partly machine-gen. |
| **TED-LIUM 3** | 452h | TED talks | **CC-BY-NC-ND 3.0** ❌ **BLOCKER** | `LIUM/tedlium` (`release3`) | NC + ND — unusable commercially. |
| **VoxPopuli (en)** | ~543h | EU Parliament speeches | transcripts **CC0-1.0** ✅ | `facebook/voxpopuli` (`en`) | CC0 transcripts → clean free GigaSpeech alternative. |
| **AMI** | ~100h | **Business MEETINGS** | **CC-BY-4.0** ✅✅ | `edinburghcstr/ami` (`ihm`,`sdm`) | ⭐ Highest domain value. `ihm`=clean close-talk. |
| **ICSI** | ~72h | **Research MEETINGS** | **CC-BY-4.0** ✅✅ | `argmaxinc/icsi-meetings`, `ggfox00000/dia-ICSIMeetingCorpus-all` | No clean first-party ASR card; supplementary to AMI. |
| **Earnings-22** | ~119h | **Earnings calls** (accented) | **CC-BY-SA-4.0** ✅✅ ⚠️ copyleft | `distil-whisper/earnings22` | Built as eval; sample for replay flavor. |
| **SPGISpeech** | 5,000h | **Earnings calls** (pro transcripts) | `other` (Kensho): **academic/internal only, no redistribution** ❌ **BLOCKER** | `kensho/spgispeech` (gated) | Highest transcript quality but legally unusable. |
| **YODAS (en)** | huge (en shards) | YouTube | **CC-BY-3.0** repo; per-video varies; noisy | `espnet/yodas` (`en000`…) | Overkill + noisy for a small replay slice. |

---

## 3. Per-dataset details (highlights)

### LibriSpeech ⭐ replay backbone
```python
ls = load_dataset("openslr/librispeech_asr", "clean", split="train.100")  # ~100h, CC-BY-4.0, 16 kHz
ls_replay = ls.shuffle(seed=42).select(range(int(len(ls)*0.10)))           # ~10h
```
- `text` is UPPERCASE, no punctuation — normalize to your target style (§5).

### Common Voice EN (CC0 mirror)
```python
cv = load_dataset("fsicoli/common_voice_17_0", "en", split="train")  # CC0, ungated
cv = cv.cast_column("audio", Audio(sampling_rate=16000))             # mp3 48k → 16k
```
- Official Mozilla repos gated since Oct 2025 → use CC0 mirror or older ungated version.

### AMI ⭐ meeting-domain centerpiece
```python
ami = load_dataset("edinburghcstr/ami", "ihm", split="train")  # clean close-talk meetings, 16 kHz, CC-BY-4.0
ami_replay = ami.shuffle(seed=42).select(range(int(len(ami)*0.10)))  # ~10h
```
- `ihm`=clean (use for replay), `sdm`=far-field (noise robustness). Per-utterance segments; has speaker labels/timestamps (bonus for diarization). British/European + non-native accents — *good* for a code-switch product.

### People's Speech — use only `clean` (CC-BY-4.0)
```python
ps = load_dataset("MLCommons/peoples_speech", "clean", split="train", streaming=True)  # CC-BY-4.0 only
```

### VoxPopuli (en) — `facebook/voxpopuli` (config `en`), CC0 transcripts. Formal speeches, multi-accent.

---

## 4. Licensing posture

**🟢 Commercial-deployable:** Common Voice EN (CC0), VoxPopuli transcripts (CC0), LibriSpeech (CC-BY-4.0), **AMI** (CC-BY-4.0), ICSI (CC-BY-4.0), People's Speech `clean` (CC-BY-4.0), YODAS en (CC-BY-3.0, per-video caveat).

**⚠️ Copyleft (note in data card):** Earnings-22 (CC-BY-SA-4.0); People's Speech `*_sa` (avoid — use `clean`).

**❌ Blockers (do NOT ship):** GigaSpeech (conflicting/non-commercial; Apache tag likely scripts-only; SpeechColab doesn't own audio — needs written clearance); SPGISpeech (academic/internal only + no redistribution); TED-LIUM 3 (CC-BY-NC-ND).

> **Rule:** build the replay slice from **CC0 + CC-BY-4.0** only (LibriSpeech, AMI, CV CC0 mirror, VoxPopuli, People's Speech `clean`). Add Earnings-22 only if you accept/document copyleft. Don't rely on "model-trained-on-NC-is-fine-under-fair-use" for a shipped product.

---

## 5. How to use

- **Replay = 10–15% of audio-hours.** ~20–30h curated English is plenty to anchor against forgetting; more eats Japanese/CS budget.
- **Add AMI — emphatically.** It's simultaneously replay *and* in-domain meeting adaptation; CC-BY-4.0, 16 kHz, segment-level. Make AMI `ihm` the **largest** component of the English slice; add a little `sdm` if you capture room audio.
- **Preprocessing:** resample 16 kHz mono (CV 48k, YODAS2 24k need it); loudness/peak normalize to match JA data; **normalize transcripts to ONE scheme** (LibriSpeech is UPPERCASE/no-punct; AMI/Earnings are cased with disfluencies) — mismatched conventions confuse the model; chunk ≤20–30s; **stream+sample** the giant sets; **hold out a small English eval** (LibriSpeech `test.clean` + AMI `test`) to *measure* forgetting before/after.

---

## 6. Sources & uncertainties

**Sources:** LibriSpeech https://huggingface.co/datasets/openslr/librispeech_asr · CV17 https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 , mirror https://huggingface.co/datasets/fsicoli/common_voice_17_0 · GigaSpeech https://huggingface.co/datasets/speechcolab/gigaspeech (license discussion #13) · People's Speech https://huggingface.co/datasets/MLCommons/peoples_speech , https://arxiv.org/abs/2111.09344 · TED-LIUM https://huggingface.co/datasets/LIUM/tedlium · VoxPopuli https://huggingface.co/datasets/facebook/voxpopuli , https://arxiv.org/abs/2101.00390 · AMI https://huggingface.co/datasets/edinburghcstr/ami , https://arxiv.org/abs/1906.11047 · ICSI https://huggingface.co/datasets/argmaxinc/icsi-meetings · Earnings-22 https://huggingface.co/datasets/distil-whisper/earnings22 , https://arxiv.org/abs/2203.15591 · SPGISpeech https://huggingface.co/datasets/kensho/spgispeech , https://arxiv.org/abs/2104.02014 · YODAS https://huggingface.co/datasets/espnet/yodas , https://arxiv.org/abs/2406.00899

**Uncertainties:** GigaSpeech license genuinely ambiguous (Apache tag vs non-commercial access agreement; discussion #13 unresolved) → flagged RESTRICTED. Hours approximate (±10–20%). TED-LIUM CC-BY-NC-ND confirmed via LIUM/OpenSLR. CV post-v17 audio behind Mozilla Data Collective; `fsicoli` is a community CC0 mirror (verify CC0 grant per release for production). People's Speech split→license mapping inferred from card. ICSI not plug-and-play (assemble audio↔text yourself).
