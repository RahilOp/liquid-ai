#!/usr/bin/env python3
"""LoRA fine-tune a Whisper checkpoint on the synthetic CS corpus.

Parameterised by model size so the same script trains whisper-tiny / base /
small / large-v3. Reads our engine + augmentation manifests (clean + augmented),
extracts log-mel features on the fly (no big feature cache -> disk-friendly),
and fine-tunes q_proj/v_proj LoRA with a Seq2SeqTrainer. Saves a PEFT adapter.

On-the-fly feature extraction keeps the disk footprint tiny (the 10 GB budget is
for models, not a mel cache). Language is left unforced (the CS data mixes JA+EN);
labels are the raw code-switched transcript.

Run (on mactrn01, in the venv, on a free GPU):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
     scripts/03-training/asr-ft/train_whisper.py \
     --model-id openai/whisper-small \
     --manifests data/synth/edge_tts/manifest.jsonl data/synth/kokoro/manifest.jsonl \
                 data/synth/melo/manifest.jsonl data/augmented/manifest.jsonl \
     --out models/trained/whisper-small --epochs 3
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


def load_rows(manifests: list[str]) -> list[dict]:
    rows = []
    for m in manifests:
        for line in Path(m).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append({"audio_path": r["audio_path"], "text": r["transcript"]})
    return rows


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a  # corpus is already 16 kHz mono


@dataclass
class Collator:
    processor: Any

    def __call__(self, batch: list[dict]) -> dict:
        feats = [{"input_features": b["input_features"]} for b in batch]
        out = self.processor.feature_extractor.pad(feats, return_tensors="pt")
        labels = self.processor.tokenizer.pad(
            [{"input_ids": b["labels"]} for b in batch], return_tensors="pt")
        lab = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (lab[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            lab = lab[:, 1:]
        out["labels"] = lab
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    print(f"[train] model={args.model_id}")
    processor = WhisperProcessor.from_pretrained(args.model_id, task="transcribe")

    rows = load_rows(args.manifests)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"[train] {len(train_rows)} train / {len(val_rows)} val clips")

    def prepare(batch):
        audio = load_audio(batch["audio_path"])
        batch["input_features"] = processor.feature_extractor(
            audio, sampling_rate=16000).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    train_ds = Dataset.from_list(train_rows).map(prepare, remove_columns=["audio_path", "text"])
    val_ds = Dataset.from_list(val_rows).map(prepare, remove_columns=["audio_path", "text"])

    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none"))
    model.print_trainable_parameters()

    out = Path(args.out)
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out), per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio, num_train_epochs=args.epochs,
        bf16=True, gradient_checkpointing=False,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        logging_steps=25, report_to=[], remove_unused_columns=False,
        label_names=["labels"], dataloader_num_workers=4,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        predict_with_generate=False,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=Collator(processor), processing_class=processor.feature_extractor)

    trainer.train()
    final = out / "final"
    model.save_pretrained(str(final))
    processor.save_pretrained(str(final))
    print(f"[train] adapter saved -> {final}")


if __name__ == "__main__":
    main()
