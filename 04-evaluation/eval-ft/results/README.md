# Tracked evaluation results

This directory is the permanent home for the artefacts that back Table 5 in the
paper. Large generated artefacts (`artifacts/`) are git-ignored, but prediction
JSONLs and score JSONs here are small and are committed so the table is
reproducible from checked-in files.

## Layout

```
results/
  preds/              # one JSONL prediction file per model × subset
  scores/             # scorer output JSON for each subset
  table5.json         # regenerated Table 5 values
  table5.txt          # human-readable Table 5
```

## Prediction file naming convention

Each row of Table 5 has three prediction files (CS, JA-only, EN-only):

| System (Table 5 label)                  | CS prediction                         | JA prediction                         | EN prediction                         |
|-----------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|
| LFM2.5-Audio-1.5B-JP (zero-shot)        | `lfm_base_cs.jsonl`                   | `lfm_base_ja.jsonl`                   | `lfm_base_en.jsonl`                   |
| Whisper large-v3 (zero-shot)            | `whisper_base_cs.jsonl`               | `whisper_base_ja.jsonl`               | `whisper_base_en.jsonl`               |
| Whisper large-v3 + LoRA (merged)        | `whisper_ft_merged_cs.jsonl`          | `whisper_ft_merged_ja.jsonl`          | `whisper_ft_merged_en.jsonl`          |
| LFM + LoRA, augmentation only           | `lfm_lora_aug_cs.jsonl`               | `lfm_lora_aug_ja.jsonl`               | `lfm_lora_aug_en.jsonl`               |
| LFM + LoRA, + Conformer encoder LoRA    | `lfm_lora_encoder_cs.jsonl`           | `lfm_lora_encoder_ja.jsonl`           | `lfm_lora_encoder_en.jsonl`           |
| LFM + LoRA, r=32 encoder + FLEURS mix   | `lfm_lora_r32_cs.jsonl`               | `lfm_lora_r32_ja.jsonl`               | `lfm_lora_r32_en.jsonl`               |

Add new rows by adding the three prediction files and editing `bin/run_table5.sh`.

## Regenerate Table 5

After prediction files are populated:

```bash
make -C 04-evaluation/eval-ft benchmark
```

This runs `eval/score.py` on each subset, merges the subset scores into one
Table 5 JSON, and prints the table.
