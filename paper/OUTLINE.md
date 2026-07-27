# Paper story, outline, and claim–evidence map

Working document for `main.tex`. Not part of the paper.

---

## The story in one paragraph

A LoRA-adapted Whisper large-v3 fine-tuned on a small synthetic (TTS) Japanese–English
code-switching corpus reaches **2.48% CS-WER on its own held-out synthetic test set** —
a number that reads like a solved task. We take *that exact checkpoint*, unchanged, and
evaluate it on **real human-voiced code-switched read speech** (CS-FLEURS JA–EN, n=196).
It scores **36.3% MER** and still renders **27% of English words outside Latin script**.
The adaptation is real — it beats zero-shot Whisper large-v3 (40.9% MER, 64.6% ScriptAcc) —
but the in-domain number overstates deployable performance by a wide margin. We then show
that a 1.5B on-device audio LM (LFM2.5-Audio-1.5B-JP) with **encoder-side** LoRA reaches
**80.8% ScriptAcc / 27.0% MER** on the same frozen benchmark, surpassing both Whisper
systems on code-switching while leaving Japanese intact — but trailing badly on
English-monolingual audio.

**Thesis:** in low-resource code-switching ASR, (a) synthetic single-voice corpora
inflate in-domain metrics far beyond what transfers to real speech, (b) script fidelity,
not WER, is the metric that exposes this, and (c) adapting the *acoustic encoder* — not
just the decoder's language prior — is what actually buys code-switching capability.

## Why this is publishable and the dissertation was not

The dissertation names the synthetic-to-real gap in its own Threats to Validity as the
"highest-priority follow-up" and cannot address it. This paper addresses it with a
frozen, held-out, real-speech benchmark and a second architecture. The negative result
(the gap) is the contribution; the LFM result is the constructive answer.

---

## Section outline (target ~8–10 pp)

1. **Introduction** — CS-ASR matters for JA-EN; synthetic data is the standard low-resource
   workaround; nobody measures what it costs you. Contributions list.
2. **Background & Related Work** — Whisper/multilingual ASR; CS-ASR (SEAME lineage);
   PEFT/LoRA; synthetic speech for ASR training; the evaluation-metric problem for mixed script.
3. **Corpora** — 3.1 synthetic TTS corpus (construction, why it is convenient, single-voice);
   3.2 CS-FLEURS JA-EN read/test (real voices, align-then-swap text); 3.3 FLEURS mono controls.
4. **Evaluation methodology** — MER, ScriptAcc (defn + why), JA-CER/JA-WER, EN-WER; normalisation;
   why CS-WER and MER are *not* commensurable (explicit).
5. **Whisper LoRA adaptation** — recipe, phases, the r=8/1e-4 config.
6. **LFM2.5-Audio adaptation** — hybrid SSM + 17-layer Conformer; LM LoRA vs encoder LoRA;
   two-param-group LR.
7. **Results & ablations** — in-domain synthetic table; real-speech table; encoder-LoRA ladder;
   noise/rank/composition ablations.
8. **Synthetic-to-real analysis** — the core section. Per-metric decomposition, failure modes.
9. **Deployment** — RTF/VRAM on RTX 4060 laptop; observed generation artefacts.
10. **Limitations** — n=196, single speaker, align-then-swap density, no seeds/CIs, undocumented best run.
11. **Conclusion**

---

## Claim–evidence map

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | Whisper large-v3 + LoRA (r=8, lr 1e-4, 15 ep) reaches 2.48% CS-WER on the synthetic held-out set | `clean_submission/results/evaluation_results_advanced/exp_v3_r8_1e4_ep15.json`; dissertation Table 5 | **supported** |
| C2 | The same merged checkpoint scores 36.3% MER / 73.0% ScriptAcc on CS-FLEURS JA-EN read/test | `docs/eval-ft-context.md` §2; model `Awshesh12/whisper-large-v3-ja-en-cs-merged` built by `remerge_large_v3.py` from `lora_v3_r8_1e4_ep15_best` | **supported (log-reported; raw preds git-ignored)** |
| C3 | Synthetic in-domain metrics overstate real-speech performance | C1 + C2 | **supported only as a qualitative gap** — CS-WER and MER are different metrics on different sets; must NOT be stated as "10× worse" |
| C4 | Fine-tuning still transfers: FT-Whisper beats zero-shot Whisper large-v3 on real CS speech | 36.3 vs 40.9 MER; 73.0 vs 64.6 ScriptAcc | **supported** (gain is modest: +8.4 ScriptAcc) |
| C5 | LFM2.5-Audio + r32 encoder LoRA + FLEURS mix reaches 80.8 ScriptAcc / 27.0 MER | `README.md`, `docs/eval-ft-context.md` §5 | **number supported; method undocumented** — training log stops at Run 7, no config/curve for this run. Flagged `\todo` in text. |
| C6 | Encoder-side LoRA is the decisive component | ladder 43.6 → 61.8 → 80.8 ScriptAcc | **partially supported** — 43.6→61.8 isolates encoder LoRA (+18.2); 61.8→80.8 confounds rank (r8→r32) *and* FLEURS data mix. Must state the confound. |
| C7 | No catastrophic forgetting of Japanese in the best LFM run | JA-CER 6.6 → 6.4; JA-WER 7.6 → 7.2 | **supported** |
| C8 | English-monolingual remains far behind Whisper | EN-WER 37.1 vs 4.2 | **supported** — own it, do not bury |
| C9 | All systems run well faster than real time on an 8 GB laptop GPU | `laptop_hardware_and_inference_specs.md` (RTX 4060, avg RTF 0.020/0.048/0.064) | **supported** |
| C10 | Noise augmentation is essential (271.9 → 13.2 CS-WER) | dissertation Table 6 | **supported, synthetic-only** — must be scoped to the synthetic corpus |
| — | FT-Whisper's JA-CER / JA-WER / EN-WER on FLEURS mono controls | **not measured** | **gap** — reported as `—`; named in Limitations as the obvious next run |

## Citation corrections applied

- **Hsu et al. (2023), arXiv 2310.12477** — dissertation cites this ~6× as the source for
  "q_proj+v_proj, r=8 is optimal for Whisper LoRA." The paper is *In-Context Learning of
  Textless Speech LM for Speech Classification*: not LoRA, not Whisper, not ASR. **Removed.**
  Those hyperparameter choices are re-grounded on this work's own rank ablation (Table:
  rank sweep) and on Hu et al. (2021).
- **"Indra, W.G. et al. (2020)"** → **Winata, G.I. et al. (2020)**, *Meta-Transfer Learning
  for Code-Switched Speech Recognition*. Given name had been parsed as surname.

## New experiments (2026-07) — multi-engine corpus + trained Whisper

Evidence base: `paper/results/README.md` and the score JSONs in `paper/results/`.
These claims are measured in this repo, not inherited from the dissertation. They
strengthen C3 (now commensurable) and close the "not measured" mono-controls gap.

| # | Claim | Evidence | Status |
|---|---|---|---|
| C11 | A multi-engine, style-varied, augmented synthetic corpus (5,400 clips; 3 TTS engines × 300 transcripts × clean+5 augmentations) can be built and verified | `results/README.md` §1; manifests under `data/synth`, `data/augmented` | **supported** |
| C12 | The synthetic→real gap is commensurable: the *same* FT checkpoint scores **1.1% MER on synthetic vs 26.1% MER on real** CS-FLEURS (~24×), plus 4.13% CS-WER reproducing the dissertation's ~3% | `results/README.md` §4; `data/eval-diss-synth/ft_preds.jsonl` | **supported — upgrades C3 from qualitative to a fair ratio (same metric)** |
| C13 | Whisper large-v3 + LoRA on the new corpus reaches **21.1% MER / 97.1% ScriptAcc** on real CS-FLEURS, beating base (40.9%) and the dissertation FT (26.1%) | `results/eval-preds_trained_scores_csfleurs.json` | **supported** |
| C14 | Only large-v3 transfers to real speech; tiny/base/small reach high ScriptAcc but EN-WER >100% (hallucinate, drop Japanese) | same JSON; `results/README.md` §5 | **supported** — ScriptAcc must be read with MER |
| C15 | Augmentation is the dominant transfer lever: clean-only 67.3% vs full 21.1% MER on real, yet clean-only is *best* on synthetic (3.5%) — synthetic eval inverts model selection | `results/eval-preds_ablation_scores_{csfleurs,synth}.json` | **supported** |
| C16 | Engine/speaker diversity helps: 1 engine 33.5% → 3 engines 21.1% MER on real | ablation JSONs | **supported** |
| C17 | LoRA rank barely matters (r8 best; r16 26.2%, r32 26.6% on real) — more capacity slightly overfits synthetic | ablation JSONs | **supported** — contrast with dissertation's encoder-LoRA rank finding (different adapter site) |
| C18 | CS-only fine-tuning forgets Japanese (JA-CER rises every size; large-v3 6.0→20.5) while English is preserved (4.2→4.7 EN-WER) | `results/forgetting_tables.txt` | **supported — closes the old "FT-Whisper mono controls not measured" gap** |
| C19 | Monolingual replay (2,000 FLEURS clips) restores Japanese (→6.0 JA-CER) and *improves* the CS task (CS-FLEURS MER 21.1→17.0) | `results/forgetting_tables.txt`; `results/README.md` §7 | **supported** |

Superseded/updated: C2 now measured directly (26.1% MER under repetition-penalty
decoding; note vs the dissertation's log-reported 36.3%). C3 → C12 (commensurable).
The old `—` mono-controls row → C18. C10 (noise essential) now has a real-speech
counterpart in C15 (augmentation essential for *transfer*, not just in-domain).

## Open items for the author

1. Recover the training config + val curve for the best LFM run (r32 encoder LoRA + FLEURS mix)
   and add it to `docs/asr-ft-training-log.md`. Currently the paper's strongest number has no
   documented recipe.
2. ~~Run FT-Whisper over `fleurs_ja_jp_test200` / `fleurs_en_us_test200`~~ — **done** (C18).
3. Bootstrap CIs over the 196 CS utterances for every row of the real-speech table.
4. Reconcile FT-Whisper CS-FLEURS MER (26.1% here vs 36.3% dissertation) — decoding difference.
5. The artificial eval shares the edge-tts engine with training (in-domain acoustics); a
   held-out-engine synthetic eval would be a fairer generalization test.
