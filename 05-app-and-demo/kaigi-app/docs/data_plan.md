# Data plan

> **License-vetted dataset selection per task lives in [`../research/`](../research/README.md)** — read that for concrete commercial-safe picks, exact HF IDs, and license posture. This file is the high-level plan.

## Goal

Teach LFM2.5-Audio two things at once, without breaking what it already knows:
1. **Japanese ASR** (the base model is English-only; the `-JP` base already has this — we start from `-JP`).
2. **JP↔EN intra-sentence code-switching** — the differentiator. Specifically, the loanword↔switch boundary
   (see `transcription_convention.md`).

## Corpus mix (target)

| Bucket | Share | Source | Why |
| --- | --- | --- | --- |
| Synthetic JP↔EN code-switch | 20–35% | **this repo** (LLM/template transcripts → bilingual TTS) | the new skill |
| Monolingual Japanese | 50–70% | Common Voice JA (CC0), ReazonSpeech slice, JSUT | keep/improve Japanese |
| Monolingual English replay | 10–15% | LibriSpeech / Common Voice EN slice | prevent forgetting English |

Start small: literature shows **10–50h synthetic CS + 100–500h mono JP** moves the needle; even <20h CS helps.
For the hackathon, aim for a few hours of synthetic CS + a few-thousand-utterance mono mix and iterate.

## Synthetic generation (this repo)

Two generators feed `transcripts.jsonl`:
- **Seed bank** (`data/seeds/meeting_codeswitch_seed.jsonl`) — hand-written, gold, convention-correct. Doubles as
  eval seed and as few-shot for the LLM generator.
- **Templated contrast pairs** — from `data/seeds/domain_terms.json`, emit each business term **both** ways
  (`deadline` switch vs `デッドライン` loanword) in the same frames. This directly teaches the boundary.
- **(optional) LLM** — OpenAI-compatible endpoint expands coverage/naturalness, few-shot-prompted with the seeds
  and the convention. Off by default (no key needed to run the pipeline).

Then `synthesize.py` renders audio with **script-matched voices** (Latin span → native English voice, JP/katakana
span → Japanese voice; same-gender pair per utterance so a splice sounds like one bilingual speaker), 24 kHz wav.

## Output contract

`manifest.jsonl` rows (one per utterance) — consumed by `cs_asr_iterator.py`:
```json
{"id":"...", "audio_path":"audio/0001.wav", "transcript":"来週までにdeadlineを設定します。",
 "spans":[{"text":"来週までに","lang":"ja"},{"text":"deadline","lang":"en"},{"text":"を設定します。","lang":"ja"}],
 "style":"splice", "voice":"f1", "domain":"meeting", "source":"seed", "duration_s":3.1, "sr":24000}
```

## Evaluation (build alongside)

- **Primary: MER** (word-level for Latin spans + char-level for JP) and **PIER** (error rate *at switch points* — the
  metric that proves we fixed the hard part).
- Plus separate **JP-CER** / **EN-WER**, and a **before/after monolingual** check to show no catastrophic forgetting.
- Eval set = held-out synthetic slice + (if obtainable) **JECS** (2.5h real JP↔EN, NAIST) as the gold real-audio test.
- Use CER for Japanese (no word boundaries; MeCab/`fugashi` tokenize). Allow a katakana-normalized reference variant.

## Open work (not yet in the scaffold)

- [ ] `scripts/download_mono.py` — pull Common Voice JA + an English slice into `data/raw/`.
- [ ] `data/eval/` — JECS loader + held-out split + PIER scorer.
- [ ] multi-voice prosody smoothing (crossfade) for nicer splices.
- [ ] LLM generator endpoint wiring (interface is stubbed in `gen_transcripts.py`).
- [ ] final `mix.py` — assemble the 30/55/15 mix into one preprocessed dataset on the GPU job.
