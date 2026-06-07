# LFM2.5-Audio-1.5B-JP Fine-tuning Log
**Hackathon: Hack the Liquid WAY — June 6–7, 2026, Tokyo**

---

## 1. Project Overview

**Goal**: Fine-tune `LiquidAI/LFM2.5-Audio-1.5B-JP` for multilingual ASR with code-switching support (Japanese + English).

**Base model**: `LiquidAI/LFM2.5-Audio-1.5B-JP`
- Architecture: Liquid Foundation Model (hybrid SSM + attention) + ConformerEncoder audio encoder
- 1.5B parameters total
- Conformer encoder: 17 layers, `RelPositionMultiHeadAttention`

**Training**: Hugging Face GPUs (fine-tuned using Hugging Face GPU credits).
- Compute: Hugging Face GPU instances
- Python: 3.12, venv at `/awshesh/lfm2.5/awshesh/.venv_train/`
- Workspace: `/awshesh/lfm2.5/awshesh/`

**W&B project**: `lfm25-asr-lora-jp` (entity: `awsheshnath708-lfm`)

---

## 2. Model Architecture Notes

### Top-level children
```
lfm           → Lfm2Model          (language model / depthformer)
conformer     → ConformerEncoder   (audio encoder, 17 layers)
audio_adapter → MLP                (512 → 2048 → 2048)
audio_embedding → SharedEmbedding
depthformer   → RawLMBackbone
depth_linear  → Linear
depth_embeddings → ModuleList
```

### Conformer encoder linear module names (for LoRA targeting)
Per layer (`conformer.layers.{0..16}`):
- Self-attention (`RelPositionMultiHeadAttention`): `linear_q`, `linear_k`, `linear_v`, `linear_out`, `linear_pos`
- FFN (`ConformerFeedForward`): `linear1`, `linear2`
- Conv: `pointwise_conv1`, `pointwise_conv2` (Conv1d — not LoRA-able), `depthwise_conv`

### LM / depthformer linear module names
- Attention: `q_proj`, `k_proj`, `v_proj`, `out_proj`
- SSM: `in_proj`
- FFN: `w1`, `w2`, `w3`

---

## 3. Synthetic Dataset

### Generation
- **CS (code-switching)**: 500 audio samples — JA/EN mixed speech, generated with TTS
  - Metadata: `/awshesh/lfm2.5/awshesh/data/code_switching/metadata/transcriptions.json`
  - Fields: `id`, `audio_file`, `full_transcription`, `english_transcription`, `japanese_transcription`
- **EN (English-only)**: 300 audio samples
  - Metadata: `/awshesh/lfm2.5/awshesh/data/english_only/metadata/transcriptions.json`
- **JA (Japanese-only)**: 300 audio samples
  - Metadata: `/awshesh/lfm2.5/awshesh/data/japanese_only/metadata/transcriptions.json`

### JSONL format (used by preprocessor)
```json
{"audio_file": "data/code_switching/audio/cs_001.wav", "system": "Perform ASR.", "target": "...", "type": "cs_native"}
```

### Type taxonomy
| type | audio | system prompt | target |
|------|-------|--------------|--------|
| `cs_native` | CS audio | Perform ASR. | full_transcription (mixed) |
| `cs_ja` | CS audio | Perform ASR in Japanese. | japanese_transcription |
| `cs_en` | CS audio | Perform ASR in English. | english_transcription |
| `en_native` | EN audio | Perform ASR. | english_transcription |
| `en_ja` | EN audio | Perform ASR in Japanese. | japanese transcription |
| `en_en` | EN audio | Perform ASR in English. | english_transcription |
| `ja_native` | JA audio | Perform ASR. | japanese_transcription |
| `ja_ja` | JA audio | Perform ASR in Japanese. | japanese_transcription |
| `ja_en` | JA audio | Perform ASR in English. | english transcription |

### System prompt evolution
| Phase | Native/CS | Force Japanese | Force English |
|-------|-----------|---------------|--------------|
| Runs 1–4 | `Transcribe the audio.` | `Transcribe in Japanese.` | `Transcribe in English.` |
| Runs 5+ | `Perform ASR.` | `Perform ASR in Japanese.` | `Perform ASR in English.` |

---

## 4. Data Augmentation (MUSAN)

### Download
- Source: `https://www.openslr.org/resources/17/musan.tar.gz` (11 GB)
- Downloaded to: `/awshesh/lfm2.5/musan/musan.tar.gz`
- Extracted (noise + music only): `/awshesh/lfm2.5/musan_extracted/musan/`
  - `noise/`: 930 WAV files
  - `music/`: 660 WAV files

### Augmentation script
`/awshesh/lfm2.5/awshesh/augment_data.py`

Each original audio → 3 augmented versions:
| Suffix | Noise type | SNR |
|--------|-----------|-----|
| `_aug1` | MUSAN noise | 20 dB |
| `_aug2` | MUSAN noise | 13 dB |
| `_aug3` | MUSAN music | 20 dB |

Augmented audio saved to: `data/{category}/audio_aug/`

### Result
- 1100 originals × 3 = 3300 augmented WAV files
- `train.jsonl`: 2969 → 6267 rows
- `eval.jsonl`: 329 → 1215 rows

---

## 5. Preprocessing Pipeline

**Script**: `/awshesh/lfm2.5/awshesh/preprocess_asr_data.py`

**Usage**:
```bash
CUDA_VISIBLE_DEVICES=0 python preprocess_asr_data.py \
  --train_jsonl data/training/train.jsonl \
  --eval_jsonl  data/training/eval.jsonl \
  --output_dir  data/preprocessed \
  --hf_token    <HF_TOKEN>
```

**Output format**: HuggingFace Arrow dataset
- Columns: `text`, `audio_in`, `audio_in_lens`, `audio_out`, `modality_flag`, `supervision_mask`

### Preprocessed dataset directories
| Directory | JSONL source | Rows (train/eval) | Notes |
|-----------|-------------|-------------------|-------|
| `data/preprocessed/` | Original train/eval (Transcribe prompts) | 2969/329 | First run |
| `data/preprocessed_cs/` | cs_only_train/eval (Transcribe the audio.) | 458/42 | CS-only exp 1 |
| `data/preprocessed_aug/` | Augmented train/eval (Transcribe prompts) | 6267/1215 | With MUSAN aug |
| `data/preprocessed_asr_prompt/` | Augmented (Perform ASR prompts) | 6267/1215 | New prompts |
| `data/preprocessed_cs_asr_prompt/` | CS-only (Perform ASR.) | 929/159 | CS-only exp 2 |
| `data/preprocessed_native_asr/` | Native-only (Perform ASR., EN up-weighted) | 2593/568 | **Current run** |

---

## 6. LoRA Configuration

### Base config (all runs)
```python
LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)
```

### Target modules evolution
| Phase | LM targets | Encoder targets |
|-------|-----------|----------------|
| Runs 1–5 | `q_proj, k_proj, v_proj, out_proj, in_proj, w1, w2, w3` | ❌ none |
| Runs 6+ (encoder LoRA) | `q_proj, k_proj, v_proj, out_proj, in_proj, w1, w2, w3` | `linear_q, linear_k, linear_v, linear_out, linear1, linear2` |

### Trainable parameters
- LM LoRA only: ~4.7M params (~0.32% of 1.46B)
- LM + encoder LoRA: **8,155,136 params (0.5579%)**

---

## 7. Training Runs

### Run 1 — Mixed-task, LM LoRA only (original)
- **Script**: `train_lora.py`
- **Data**: `data/preprocessed/` (2969 train, Transcribe prompts)
- **Steps**: 500 → extended to 1000
- **LR**: 3e-5 cosine, warmup 50
- **Batch**: 8
- **GPU**: a Hugging Face GPU
- **Checkpoint dir**: `checkpoints/lfm_lora/`
- **W&B**: run `eyb6wb7z`
- **Val loss**: 0.3962 (step 500) → **0.3154 (step 1000)**
- **Best checkpoint**: `checkpoints/lfm_lora/step_1000`

### Run 2 — HF Space continued training (steps 1000→2500)
- **Script**: `hf_space/train_space.py` (Docker on HF Space)
- **Space**: `AwsheshNath/lfm2-lora-train` (A10G hardware)
- **Data**: HF dataset `AwsheshNath/lfm2-lora-data`
- **Resumes from**: `AwsheshNath/lfm2-lora-adapter` (step_1000)
- **Steps**: 1000 → 2500 (1500 more steps)
- **LR**: 3e-5 cosine from step 1000
- **W&B**: run `hf-space-steps1000-2500` (id: `gtl7c9lc`)
- **Final val loss**: 0.2361 (step 2500)
- **HF Hub checkpoints**: `step_1200, 1400, 1600, 1800, 2000, 2200, 2400, final_step2500`
- **Best checkpoint**: `AwsheshNath/lfm2-lora-adapter/final_step2500` (val history not synced, use final)

### Run 3 — CS-only, LM LoRA (Transcribe prompts)
- **Data**: 458 `cs_native` rows, prompt: `Transcribe the audio.`
- **Steps**: 1000, warmup 30
- **GPU**: a Hugging Face GPU
- **Checkpoint dir**: `checkpoints/lfm_lora_cs/`
- **W&B**: run `volcanic-capybara-6`
- **Val loss**: 0.5631 → **0.0018 (step 1000)**
- **Final**: `checkpoints/lfm_lora_cs/final`
- **Note**: Very low loss — model memorized 458 CS samples

### Run 4 — Augmented, LM LoRA (Transcribe prompts)
- **Data**: `data/preprocessed_aug/` (6267 train with MUSAN augmentation)
- **Steps**: 2000, warmup 100
- **GPU**: a Hugging Face GPU
- **Checkpoint dir**: `checkpoints/lfm_lora_aug/`
- **W&B**: run `icy-water-8` (id: `vi3l0zul`)
- **Val losses**:
  ```
  step 500  → 0.3464
  step 800  → 0.2816
  step 1200 → 0.2620
  step 1400 → 0.2567
  step 1500 → 0.2477  ← best val (not saved, nearest saved: step_1400/1600)
  step 1600 → 0.2531
  step 2000 → 0.2575
  ```
- **Best saved checkpoint**: `checkpoints/lfm_lora_aug/step_1600` (val_loss 0.2531)
- **Final**: `checkpoints/lfm_lora_aug/final`

### Run 5 — Full encoder LoRA (Perform ASR prompts, all tasks)
- **Data**: `data/preprocessed_asr_prompt/` (6267 train, all types, new prompts)
- **System prompts**: `Perform ASR.` / `Perform ASR in Japanese.` / `Perform ASR in English.`
- **LoRA targets**: LM + conformer encoder (`linear_q/k/v/out`, `linear1/linear2`)
- **LR**: LM `3e-5`, encoder `6e-6` (5× lower) — two AdamW param groups
- **Steps**: 2000, warmup 100, batch 8
- **GPU**: a Hugging Face GPU
- **Checkpoint dir**: `checkpoints/lfm_lora_enc_lora/`
- **W&B**: run `quiet-glitter-10` (id: `08ip09g8`)
- **Trainable params**: 8,155,136 (0.5579%)
- **Val losses**:
  ```
  step 100  → 1.0300
  step 500  → 0.3561
  step 900  → 0.2762
  step 1000 → 0.2740
  step 1100 → 0.2680
  step 1200 → 0.2596
  step 1300 → 0.2491  ← best val
  step 1400 → 0.2500
  step 2000 → 0.2523
  ```
- **Best checkpoint**: `checkpoints/lfm_lora_enc_lora/step_1200` (val_loss 0.2596, saved) or `step_1400` (closest to best)
- **Final**: `checkpoints/lfm_lora_enc_lora/final`
- **Note**: Starts higher than LM-only runs due to encoder LoRA cold init

### Run 6 — CS-only + encoder LoRA (Perform ASR.)
- **Data**: `data/preprocessed_cs_asr_prompt/` (929 train: 458 orig + 471 aug, CS only)
- **System prompt**: `Perform ASR.` only
- **LoRA targets**: LM + encoder (same as Run 5)
- **LR**: LM `3e-5`, encoder `6e-6`
- **Steps**: 1000, warmup 30, batch 8
- **GPU**: a Hugging Face GPU
- **Checkpoint dir**: `checkpoints/lfm_lora_cs_enc/`
- **W&B**: run `vibrant-voice-11` (id: `ub9e4xxv`)
- **Trainable params**: 8,155,136 (0.5579%)
- **Val losses**:
  ```
  step 50  → 0.5049
  step 100 → 0.1093
  step 200 → 0.0179
  step 400 → 0.0048
  step 700 → 0.0026
  step 950 → 0.0023
  step 1000 → 0.0022  ← best
  ```
- **Best checkpoint**: `checkpoints/lfm_lora_cs_enc/final`

### Run 7 — Native-ASR, encoder LoRA, EN up-weighted (IN PROGRESS)
- **Data**: `data/preprocessed_native_asr/` (2593 train, 568 eval)
  - `cs_native`: 929 rows
  - `en_native`: 534 × 2 = 1068 rows (up-weighted ×2 per anti-forgetting advice)
  - `ja_native`: 596 rows
  - All shuffled, single prompt: `Perform ASR.`
- **LoRA targets**: LM + encoder (same as Runs 5–6)
- **LR**: LM `3e-5`, encoder `6e-6`
- **Steps**: 2000, warmup 80, batch 8
- **GPU**: a Hugging Face GPU
- **Checkpoint dir**: `checkpoints/lfm_lora_native_asr/`
- **W&B**: project `lfm25-asr-lora-jp`
- **Log**: `logs/train_lora_native_asr.log`
- **Rationale**: Restore English mono-lingual capability lost in CS-focused runs; agent recommended up-weighting `en_native` as the key anti-forgetting fix

---

## 8. Evaluation Results

### CS-FLEURS evaluation (196 real JA-EN code-switching utterances)
Evaluated with `eval_lora.py` — 3 prompts on base model vs LoRA (step_1000 mixed adapter):

| Model | Prompt | MER | JA-CER | EN-WER | ScriptAcc |
|-------|--------|-----|--------|--------|-----------|
| Base | Transcribe the audio. | ~35% | - | - | ~50% |
| LoRA step_1000 | Transcribe the audio. | **~27%** | - | - | **~61%** |

> Note: LoRA beat Whisper on ScriptAcc (60.8% vs 59.5%)

### Synthetic dataset evaluation (eval_synthetic.py, step_1000 adapter)
| Task | N | WER | CER |
|------|---|-----|-----|
| CS-native (Transcribe the audio.) | 500 | 3.4% | 2.2% |
| EN-force_en (Transcribe in English.) | 300 | 11.3% | 6.4% |
| JA-force_jp (Transcribe in Japanese.) | 300 | 9.3% | 1.5% |

---

## 9. Key Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `preprocess_asr_data.py` | `/awshesh/lfm2.5/awshesh/` | Convert JSONL → Arrow dataset |
| `train_lora.py` | `/awshesh/lfm2.5/awshesh/` | LoRA training (all runs) |
| `eval_lora.py` | `/awshesh/lfm2.5/awshesh/` | Evaluate on CS-FLEURS |
| `eval_synthetic.py` | `/awshesh/lfm2.5/awshesh/` | Evaluate on synthetic dataset |
| `augment_data.py` | `/awshesh/lfm2.5/awshesh/` | MUSAN noise augmentation |
| `train_space.py` | `hf_space/` | HF Space Docker training |

---

## 10. train_lora.py Key Arguments

```bash
python train_lora.py \
  --data       data/preprocessed_native_asr/train \
  --val_data   data/preprocessed_native_asr/eval \
  --output_dir checkpoints/lfm_lora_native_asr \
  --max_steps  2000 \
  --warmup_steps 80 \
  --batch_size 8 \
  --lr         3e-5 \          # LM LoRA learning rate
  --encoder_lr 6e-6 \          # Conformer encoder LoRA LR (5x lower)
  --log_interval  10 \
  --val_interval  100 \
  --save_interval 200 \
  --resume_adapter <path> \    # Optional: resume from existing adapter
  --start_step <N> \           # Optional: resume step counter
  --wandb_run_id <id> \        # Optional: resume W&B run
  --hf_token   <token> \
  --wandb_key  <key> \
  --wandb_project lfm25-asr-lora-jp
```

---

## 11. Environment Setup

```bash
# SSH

# Activate venv
cd /awshesh/lfm2.5/awshesh
source .venv_train/bin/activate

# Proxy (required for all network access)

# Packages (installed in .venv_train)
# torch==2.11.0+cu128, torchaudio==2.11.0
# liquid-audio==1.3.0
# peft==0.19.1
# wandb==0.27.2
# huggingface_hub==1.18.0
# safetensors==0.7.0
# soundfile==0.13.1
```

---

## 12. Checkpoint Summary Table

| Checkpoint path | Run | Best val_loss | Notes |
|---------------------------|-----|--------------|-------|
| `checkpoints/lfm_lora/step_1000` | Run 1 | 0.3154 | LM LoRA, Transcribe prompts |
| `AwsheshNath/lfm2-lora-adapter/final_step2500` (HF Hub) | Run 2 | 0.2361 | HF Space continuation |
| `checkpoints/lfm_lora_cs/final` | Run 3 | 0.0018 | CS-only specialization |
| `checkpoints/lfm_lora_aug/step_1600` | Run 4 | 0.2531 | Augmented, Transcribe prompts |
| `checkpoints/lfm_lora_enc_lora/step_1400` | Run 5 | **0.2491** | Enc LoRA, Perform ASR, all tasks |
| `checkpoints/lfm_lora_cs_enc/final` | Run 6 | 0.0022 | CS-only + enc LoRA |
| `checkpoints/lfm_lora_native_asr/` | Run 7 | in progress | **Current run** |

---

## 13. Recommended Next Steps

1. **Evaluate Run 7** (`lfm_lora_native_asr`) on CS-FLEURS + synthetic eval once training completes
2. **Compare EN-mono WER** across runs — this is the key metric per agent feedback
3. **If EN still weak**: escalate to Option B (unfreeze last N conformer layers with very low LR ~1e-6)
4. **Early stopping signal**: watch EN-mono WER — first metric to degrade during overfitting
5. **Best checkpoint selection**: use step where EN-mono WER is minimized (not just val_loss)
