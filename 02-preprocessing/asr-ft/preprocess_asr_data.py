"""
Converts our ASR JSONL dataset into liquid-audio's preprocessed format.
Reads data/training/train.jsonl and data/training/eval.jsonl.
Outputs preprocessed HuggingFace datasets to data/preprocessed/{train,eval}.

Usage:
  python preprocess_asr_data.py --hf_token hf_xxx [--model_id LiquidAI/LFM2.5-Audio-1.5B-JP]
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from pathlib import Path

from huggingface_hub import login

from liquid_audio import LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",    default="LiquidAI/LFM2.5-Audio-1.5B-JP")
    p.add_argument("--train_jsonl", default="data/training/train.jsonl")
    p.add_argument("--eval_jsonl",  default="data/training/eval.jsonl")
    p.add_argument("--output_dir",  default="data/preprocessed")
    p.add_argument("--hf_token",    default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--max_context", type=int, default=512)
    return p.parse_args()


class ASRIterator:
    """Yields list[ChatMessage] from our JSONL rows."""

    def __init__(self, jsonl_path: str) -> None:
        self.examples: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    if Path(ex["audio_file"]).exists():
                        self.examples.append(ex)
        print(f"  Loaded {len(self.examples)} examples from {jsonl_path}")

    def __iter__(self) -> Iterator[list[ChatMessage]]:
        for ex in self.examples:
            audio_bytes = Path(ex["audio_file"]).read_bytes()
            yield [
                ChatMessage(
                    role="system",
                    content=[TextSegment(text=ex["system"])],
                ),
                ChatMessage(
                    role="user",
                    content=[AudioSegment(audio=audio_bytes)],
                ),
                ChatMessage(
                    role="assistant",
                    content=[TextSegment(text=ex["target"])],
                ),
            ]


def main() -> None:
    args = parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    print(f"Loading processor: {args.model_id}")
    processor = LFM2AudioProcessor.from_pretrained(args.model_id, device="cuda").eval()
    mapper = LFM2AudioChatMapper(processor)

    out_root = Path(args.output_dir)

    for split, jsonl in [("train", args.train_jsonl), ("eval", args.eval_jsonl)]:
        out_path = out_root / split
        if out_path.exists():
            print(f"[{split}] already exists at {out_path}, skipping.")
            continue
        print(f"\n[{split}] Preprocessing {jsonl} → {out_path}")
        data = ASRIterator(jsonl)
        preprocess_dataset(
            data=data,
            output_path=out_path,
            mapper=mapper,
            max_context_length=args.max_context,
        )
        print(f"[{split}] Done.")

    print("\nPreprocessing complete. Dataset ready at:", args.output_dir)


if __name__ == "__main__":
    main()
