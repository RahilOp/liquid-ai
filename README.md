# LFM2.5-Audio JP↔EN Code-Switching — Hackathon Submission

**Hack the Liquid WAY — June 6–7, 2026, Tokyo**

A unified, on-device speech stack built by fine-tuning **`LiquidAI/LFM2.5-Audio-1.5B-JP`**
for **Japanese↔English code-switching**. Generic cloud ASR mangles bilingual speech
(forcing English into katakana, dropping the minority language); a tuned, fully offline
model fixes this for privacy-sensitive (APPI-bound) meetings.

The project is organized into three workstreams — **ASR fine-tuning**, **TTS
fine-tuning**, and **evaluation** — grouped by pipeline stage. Within stages
01–04 each appears as an `asr-ft/`, `tts-ft/`, `eval-ft/` subfolder.

## The three workstreams

| Workstream | Focus | Lives mainly in |
|------------|-------|-----------------|
| **`asr-ft/`** | Synthetic data generation + **ASR LoRA fine-tuning** pipeline (data → preprocess → train → eval → GGUF export) | stages 01–04, 06 |
| **`eval-ft/`** | The **shared evaluation harness** + frozen CS-FLEURS / FLEURS benchmark (the held-out scoreboard every model is judged on) | stage 04 |
| **`tts-ft/`** | The **Kaigi (会議) meeting assistant** — one audio base + 3 LoRA adapters (ASR / translate-and-speak / minutes), React web UI, CPU/GGUF path | stages 01–06 |

## Layout (by pipeline stage)

```
01-data-generation/   synthetic CS/EN/JA speech + transcript generation, dataset downloads
02-preprocessing/     JSONL->Arrow preprocessing, dataset balancing, manifest tooling, augmentation
03-training/          LoRA training — ASR (asr-ft) + translate-TTS / minutes / ASR (tts-ft)
04-evaluation/        eval-ft shared benchmark (primary) + per-workstream eval scripts
05-app-and-demo/      kaigi-app (full Kaigi assistant + web UI) + asr-ft Gradio servers
06-deployment/        on-device export — GGUF (llama.cpp) + ONNX notes
docs/                 cross-cutting design docs: ASR training log, eval context, system architecture
```

Within stages 01–04 the work is split into `asr-ft/`, `eval-ft/`, `tts-ft/` subfolders.

## Key result (frozen benchmark, n=196 CS + 200/200 mono)

ScriptAcc = % of English reference words kept in **Latin** script (vs forced into
katakana) — the headline code-switching metric. JA-CER (char) / JA-WER (word, via
fugashi) and EN-WER are the monolingual forgetting controls.

| Model | CS ScriptAcc ↑ | CS MER ↓ | JA-CER ↓ | JA-WER ↓ | EN-WER ↓ |
|-------|------|------|------|------|------|
| LFM2.5-Audio-1.5B-JP (base) | 37.4% | 67.3% | 6.6% | 7.6% | 68.5% |
| Whisper large-v3 (base) | 64.6% | 40.9% | 6.0% | 7.9% | **4.2%** |
| LFM + LoRA — data-aug, no encoder LoRA | 43.6% | 54.3% | 8.2% | 8.7% | 80.2% |
| LFM + LoRA — + Conformer encoder LoRA | 61.8% | 54.1% | 9.4% | 10.2% | 73.1% |
| **LFM + LoRA — r32 encoder LoRA + FLEURS mix (best)** | **80.8%** | **27.0%** | **6.4%** | **7.2%** | 37.1% |

Our best LoRA (**rank-32 Conformer-encoder LoRA + a FLEURS-balanced data mix**)
lifts code-switch Script Accuracy **37% → 81%** and cuts MER **67% → 27%** —
**surpassing Whisper large-v3 on code-switching** — while keeping Japanese fully
intact (CER 6.6 → 6.4, **no forgetting**) and roughly halving the base model's
English error (WER 68.5% → 37.1%). The ablation rows trace the gains: data
augmentation alone (+6 ScriptAcc), adding Conformer-encoder LoRA (+24), then
rank-32 + FLEURS mix (+43). English-mono WER remains the gap to close vs Whisper's 4.2%.

## Model

- Base: `LiquidAI/LFM2.5-Audio-1.5B-JP` — Liquid Foundation Model (hybrid SSM + attention)
  + 17-layer Conformer audio encoder, 1.5B params.
- Adaptation: **LoRA** (added manually via PEFT — not built into `liquid-audio`). Best ASR
  recipe: LM LoRA r=8 + **Conformer encoder LoRA r=32**, trained on a FLEURS-balanced
  synthetic+real data mix, system prompt `Perform ASR.`
- Training: **Hugging Face GPUs** (fine-tuned using Hugging Face GPU credits).

## 🤗 Hugging Face

Trained adapters and published data samples (the full corpora are large and
reproducible from the scripts in `01-data-generation/` / `02-preprocessing/`):

- **Fine-tuned LoRA adapters** — [`AwsheshNath/lfm2.5-audio-jp-kaigi-adapters`](https://huggingface.co/AwsheshNath/lfm2.5-audio-jp-kaigi-adapters)
  · code-switch ASR · translate-and-speak · meeting-minutes
- **Synthetic code-switching data (sample)** — [`AwsheshNath/synthetic-cs-sample`](https://huggingface.co/datasets/AwsheshNath/synthetic-cs-sample)
- **CS-FLEURS JA–EN (sample)** — [`AwsheshNath/cs-fleurs-jpn-eng-sample`](https://huggingface.co/datasets/AwsheshNath/cs-fleurs-jpn-eng-sample)

> Datasets, model weights, checkpoints, virtualenvs and logs are intentionally
> excluded — they are large and reproducible from the scripts here. Augmented audio
> is omitted for the same reason (regenerate with the augmentation scripts in `02-preprocessing/`).
