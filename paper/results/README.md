# Experiment evidence log (2026-07)

Every number here traces to a results file on disk (score JSONs pulled into this
directory, or logs / manifests on mactrn01 under `/awshesh/code-switching`). Numbers
are re-derived from `eval/score.py` output, not copied from prose. This is the
evidence base for the paper's new claims; see `../OUTLINE.md` for the claim map.

Metrics (defined in `04-evaluation/eval-ft/eval/score.py`): **MER** mixed-token
error rate; **JA-CER** Japanese character error rate; **EN-WER** English word error
rate; **ScriptAcc** fraction of reference English words the model kept in Latin
script. All corpus-pooled.

---

## 1. Synthetic CS corpus (new, multi-engine)

5,400 clips, 16 kHz mono, built from 300 style-varied JA-EN transcripts.
- Transcripts: `data/transcripts/cs_transcripts.jsonl` (300), generator
  `01-data-generation/asr-ft/gen_cs_transcripts.py`. Styles span density
  (sparse/moderate/dense) × span (word/phrase/clause).
- Clean: 900 = 300 × 3 engines (edge-tts multilingual, Kokoro, MeloTTS);
  `data/synth/{edge_tts,kokoro,melo}/manifest.jsonl`.
- Augmented: 4,500 = 900 × 5 families (MUSAN noise/music/babble, RIR reverb, G.711
  telephony); `data/augmented/manifest.jsonl`, built by
  `02-preprocessing/asr-ft/augment_multi.py`. Augmentation verified against the
  audio (SNRs exact, telephony band-limited, reverb embedded at RIR delay).

## 2. Evaluation sets (JA-EN, frozen)

| Set | N | Source | Role |
|---|---|---|---|
| CS-FLEURS JA-EN read/test | 196 | `data/eval-real/csfleurs` via `load_csfleurs.py` | real CS (headline) |
| Artificial held-out synthetic | 200 | `data/eval-synth` (edge-tts, transcripts held out from training) | synthetic CS |
| Dissertation synthetic test | 30 | `data/eval-diss-synth` (70/30 split, seed 42) | reproduces the original ~3% |
| FLEURS ja_jp / en_us test | 200 / 200 | `data/eval-mono` via `load_fleurs.py` | monolingual forgetting controls |

## 3. Baseline eval — base vs dissertation fine-tune (CS-FLEURS)

Source: `results/eval-preds_scores_csfleurs.json`, `..._artificial.json`.
Decoding: `--repetition-penalty 1.3 --no-repeat-ngram-size 3`, `run_whisper_hf.py`.

| Model | MER | JA-CER | EN-WER | ScriptAcc |
|---|---|---|---|---|
| Whisper large-v3 (base) | 40.9% | 49.8% | 47.4% | 64.6% |
| dissertation FT (base + `lora_v3_r8_1e4_ep15_best`) | 26.1% | 30.2% | 32.2% | 80.5% |

Base MER 40.9% / ScriptAcc 64.6% **reproduces the documented zero-shot benchmark
exactly** → the harness is validated. (The dissertation logged 36.3% MER for the FT
checkpoint; this run gets 26.1% under repetition-penalty decoding.)

## 4. The "~3%" reproduced — synthetic vs real, same model

The dissertation's ~3% was **synthetic in-domain**, not a bug. Same FT checkpoint,
dissertation synthetic test set (`data/eval-diss-synth/ft_preds.jsonl`):
- **CS-WER 4.13%** (jiwer, whitespace — the dissertation's own metric)
- **MER 1.1%, ScriptAcc 100%** (this harness); first predictions exact.

Same model on **real** CS-FLEURS: **MER 26.1%**. The synthetic→real gap is now
**commensurable** (MER on both): **1.1% → 26.1%, ~24×**. This is the key result the
original draft could only assert qualitatively (CS-WER ≠ MER).

Full commensurable table (MER, same decoding; `data/eval-diss-synth/{base,ft}_preds.jsonl`):

| Setting | Zero-shot MER | +LoRA MER |
|---|---|---|
| Synthetic CS test (n=30) | 26.6% | 1.1% |
| CS-FLEURS (n=196) | 40.9% | 26.1% |

Fine-tuning cuts synthetic MER 24× (26.6→1.1) but real MER by a third (40.9→26.1).

## 5. Whisper trained on the new corpus (CS-FLEURS, real)

Source: `results/eval-preds_trained_scores_csfleurs.json`. LoRA r=8, 3 epochs,
`03-training/asr-ft/train_whisper.py`; adapters `models/trained/whisper-*`.

| Size | base MER | **FT MER** | base ScriptAcc | **FT ScriptAcc** |
|---|---|---|---|---|
| tiny | 95.4% | 79.1% | 28.1% | 77.3% |
| base | 80.0% | 58.2% | 36.5% | 88.2% |
| small | 60.5% | 54.2% | 50.3% | 90.0% |
| large-v3 | 40.9% | **21.1%** | 64.6% | **97.1%** |

large-v3 trained on the new corpus (**21.1% MER**) beats base (40.9%) and the
dissertation FT (26.1%). Small models achieve high ScriptAcc but EN-WER >100% on
real speech (hallucination) — only large-v3 transfers. On the artificial set all FT
models score 0.4–4% MER (in-domain, same TTS engine); see
`..._trained_scores_artificial.json`.

## 6. Ablations (whisper-large-v3, CS-FLEURS real)

Source: `results/eval-preds_ablation_scores_csfleurs.json`, `..._synth.json`.

| Config | train data | real MER | real ScriptAcc | synth MER |
|---|---|---|---|---|
| full (r8) | 5,400 | **21.1%** | 97.1% | 6.0% |
| clean-only | 900 (no aug) | 67.3% | 87.8% | 3.5% |
| single-engine (edge) | 1,800 | 33.5% | 95.3% | 4.5% |
| rank 16 | 5,400 | 26.2% | 96.0% | 4.4% |
| rank 32 | 5,400 | 26.6% | 96.1% | 4.7% |

- **Augmentation dominates transfer:** clean-only 67.3% vs full 21.1% MER on real,
  yet clean-only is *best* on synthetic (3.5%). Best-on-synthetic = worst-on-real —
  synthetic in-domain eval inverts model selection.
- **Engine diversity matters:** 1 engine 33.5% → 3 engines 21.1%.
- **LoRA rank barely matters (r8 best):** more capacity slightly overfits synthetic.

## 7. Forgetting + monolingual replay

Source: `results/forgetting_tables.txt` (from `run_cs_mono.sh`). Mono replay = 2,000
FLEURS clips (1,000 JA + 1,000 EN) added to the 5,400 CS mix; `models/trained_mono/`.

JA-CER on FLEURS ja_jp test (JA forgetting):

| Size | base | CS-only | **CS+mono** |
|---|---|---|---|
| tiny | 37.6% | 43.8% | 38.9% |
| base | 25.1% | 33.7% | 26.1% |
| small | 14.1% | 42.7% | 15.2% |
| large-v3 | 6.0% | 20.5% | **6.0%** |

large-v3 full picture: JA-CER 6.0 → 20.5 (CS-only) → **6.0** (CS+mono restores it);
EN-WER 4.2 → 4.7 → 4.9 (English never at risk); **CS-FLEURS MER 40.9 → 21.1 →
17.0** (mono replay also *improves* the CS task, since better Japanese helps the JA
side of CS utterances). CS+mono is strictly the better model set.

## Reproduce

Scripts are in-repo (`01-`…`04-` stages), synced to
`/awshesh/code-switching/scripts` and run there. Drivers: `run_full_pipeline.sh`
(corpus), `run_whisper_all.sh` (train sizes), `run_ablations.sh`, `run_cs_mono.sh`,
`eval_trained_models.sh`.
