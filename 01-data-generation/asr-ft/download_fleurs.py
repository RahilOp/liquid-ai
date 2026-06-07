"""
Download 200 samples each from:
  - FLEURS EN  (google/fleurs, en_us)  — real English speech
  - FLEURS JA  (google/fleurs, ja_jp)  — real Japanese speech
  - CS-FLEURS  (skit-ai/cs-fleurs)     — JA-EN code-switching

Saves WAV files + appends to native_asr JSONL files.
All use prompt: "Perform ASR."

Usage:
  python download_fleurs.py --data-root /awshesh/lfm2.5/awshesh --hf-token hf_xxx
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import io
import numpy as np
import soundfile as sf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",  default="/awshesh/lfm2.5/awshesh")
    p.add_argument("--hf-token",   default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--n-samples",  type=int, default=200)
    p.add_argument("--seed",       type=int, default=99)
    p.add_argument("--split",      default="train")
    return p.parse_args()


def save_wav(path: str, array: np.ndarray, sr: int):
    arr = array.astype(np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    # Normalize
    peak = np.max(np.abs(arr))
    if peak > 0:
        arr = arr / peak * 0.9
    sf.write(path, arr, sr, subtype="PCM_16")


def download_fleurs_mono(dataset_name: str, config: str, split: str,
                         out_audio_dir: str, label_prefix: str,
                         n: int, rng: random.Random,
                         transcription_key: str = "transcription") -> list[dict]:
    """Download n samples from a FLEURS-style dataset, return JSONL rows."""
    from datasets import load_dataset

    print(f"\nLoading {dataset_name} ({config}, {split})...")
    # Disable torchcodec — use soundfile backend via decode=False
    import datasets as hf_datasets
    hf_datasets.config.IN_MEMORY_MAX_SIZE = 0
    ds = load_dataset(dataset_name, config, split=split, streaming=True)
    # Cast audio column to avoid torchcodec; get raw bytes instead
    from datasets import Audio as HfAudio
    ds = ds.cast_column("audio", HfAudio(decode=False))

    os.makedirs(out_audio_dir, exist_ok=True)
    rows = []
    seen = 0

    for i, ex in enumerate(ds):
        if seen >= n:
            break

        # Decode audio manually with soundfile (bypasses torchcodec)
        audio_data = ex["audio"]
        raw_bytes = audio_data.get("bytes") or audio_data.get("path")
        if raw_bytes is None:
            continue
        try:
            if isinstance(raw_bytes, bytes):
                arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
            else:
                arr, sr = sf.read(raw_bytes, dtype="float32", always_2d=False)
        except Exception as e:
            print(f"  skip {i}: audio decode error: {e}")
            continue
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        # Get transcription
        text = ex.get(transcription_key) or ex.get("sentence") or ex.get("raw_transcription", "")
        text = text.strip()
        if not text:
            continue

        # Skip very short clips (<1s)
        if len(arr) / sr < 1.0:
            continue

        fname = f"{label_prefix}_{seen:03d}.wav"
        fpath = os.path.join(out_audio_dir, fname)
        save_wav(fpath, arr, sr)

        rows.append({
            "audio_file": os.path.join("data", "fleurs", label_prefix, fname),
            "system": "Perform ASR.",
            "target": text,
            "type": f"fleurs_{label_prefix}",
        })
        seen += 1
        if seen % 50 == 0:
            print(f"  {seen}/{n} saved...")

    print(f"  Done: {len(rows)} samples saved to {out_audio_dir}")
    return rows


def append_to_jsonl(path: str, rows: list[dict]):
    existing = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                existing.add(r["audio_file"])
    new = [r for r in rows if r["audio_file"] not in existing]
    with open(path, "a") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Appended {len(new)} rows to {path}")


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    root = args.data_root
    n    = args.n_samples
    split = args.split

    # Output dirs
    en_dir = os.path.join(root, "data", "fleurs", "en")
    ja_dir = os.path.join(root, "data", "fleurs", "ja")
    cs_dir = os.path.join(root, "data", "fleurs", "cs")

    train_jsonl = os.path.join(root, "data", "training", "native_asr_train.jsonl")
    eval_jsonl  = os.path.join(root, "data", "training", "native_asr_eval.jsonl")

    all_rows = []

    # ── FLEURS EN ────────────────────────────────────────────────────────────
    en_rows = download_fleurs_mono(
        dataset_name="google/fleurs",
        config="en_us",
        split="train",
        out_audio_dir=en_dir,
        label_prefix="en",
        n=n, rng=rng,
    )
    all_rows.extend(en_rows)

    # ── FLEURS JA ────────────────────────────────────────────────────────────
    ja_rows = download_fleurs_mono(
        dataset_name="google/fleurs",
        config="ja_jp",
        split="train",
        out_audio_dir=ja_dir,
        label_prefix="ja",
        n=n, rng=rng,
    )
    all_rows.extend(ja_rows)

    # ── CS-FLEURS ────────────────────────────────────────────────────────────
    # Use train split to avoid contaminating our eval set (196 test utterances)
    try:
        cs_rows = download_fleurs_mono(
            dataset_name="skit-ai/cs-fleurs",
            config="default",
            split="train",
            out_audio_dir=cs_dir,
            label_prefix="cs",
            n=n, rng=rng,
            transcription_key="transcription",
        )
    except Exception as e:
        print(f"  cs-fleurs load error: {e}, trying alternate name...")
        try:
            cs_rows = download_fleurs_mono(
                dataset_name="google/fleurs",
                config="ja_jp",   # fallback: extra JA samples
                split="validation",
                out_audio_dir=cs_dir,
                label_prefix="cs",
                n=n, rng=rng,
            )
        except Exception as e2:
            print(f"  fallback also failed: {e2}")
            cs_rows = []
    all_rows.extend(cs_rows)

    # ── Split 90/10 train/eval and append to JSONL ──────────────────────────
    rng.shuffle(all_rows)
    cut = int(len(all_rows) * 0.9)
    train_rows = all_rows[:cut]
    eval_rows  = all_rows[cut:]

    print(f"\nTotal: {len(all_rows)} rows → {len(train_rows)} train / {len(eval_rows)} eval")
    append_to_jsonl(train_jsonl, train_rows)
    append_to_jsonl(eval_jsonl,  eval_rows)

    print("\nDone. Summary:")
    print(f"  EN: {len(en_rows)}")
    print(f"  JA: {len(ja_rows)}")
    print(f"  CS: {len(cs_rows)}")
    print(f"  Total new rows: {len(all_rows)}")
    print(f"\nNext: re-run preprocess_asr_data.py on native_asr_train/eval.jsonl")


if __name__ == "__main__":
    main()
