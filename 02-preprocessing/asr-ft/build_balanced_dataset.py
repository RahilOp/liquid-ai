"""
Build a balanced dataset with:
  - 500 FLEURS EN  (real English speech)
  - 500 FLEURS JA  (real Japanese speech)
  - 500 FLEURS CS  (real code-switching speech)
  - 300 sampled cs_native  (synthetic)
  - 300 sampled en_native  (synthetic, deduped)
  - 300 sampled ja_native  (synthetic)

Total: ~2400 rows, FLEURS ~62%, synthetic ~38%.
Outputs: data/training/balanced_train.jsonl + balanced_eval.jsonl  (90/10 split)

Usage:
  python build_balanced_dataset.py --data-root /awshesh/lfm2.5/awshesh --hf-token hf_xxx
"""
from __future__ import annotations

import argparse, io, json, os, random
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",       default="/awshesh/lfm2.5/awshesh")
    p.add_argument("--hf-token",        default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--n-fleurs",        type=int, default=500)
    p.add_argument("--n-synthetic",     type=int, default=300)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--existing-train",  default="data/training/native_asr_train.jsonl")
    p.add_argument("--existing-eval",   default="data/training/native_asr_eval.jsonl")
    p.add_argument("--out-train",       default="data/training/balanced_train.jsonl")
    p.add_argument("--out-eval",        default="data/training/balanced_eval.jsonl")
    return p.parse_args()


def save_wav(path: str, arr: np.ndarray, sr: int):
    arr = arr.astype(np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    peak = np.max(np.abs(arr))
    if peak > 0:
        arr = arr / peak * 0.9
    sf.write(path, arr, sr, subtype="PCM_16")


def download_fleurs(dataset_name, config, split, out_dir, label_prefix, n, transcription_key="transcription"):
    from datasets import load_dataset, Audio as HfAudio
    print(f"\nDownloading {dataset_name} ({config}) — {n} samples...")
    ds = load_dataset(dataset_name, config, split=split, streaming=True)
    ds = ds.cast_column("audio", HfAudio(decode=False))
    os.makedirs(out_dir, exist_ok=True)

    # Find already-downloaded files
    existing = {f.stem for f in Path(out_dir).glob("*.wav")}
    start_idx = len(existing)
    rows = []

    # Re-read existing rows
    for idx in range(start_idx):
        fname = f"{label_prefix}_{idx:03d}.wav"
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            rows.append({
                "audio_file": os.path.join("data", "fleurs_500", label_prefix, fname),
                "system": "Perform ASR.",
                "target": None,  # will be filled below if needed
                "type": f"fleurs_{label_prefix}",
            })

    # Stream and save new ones
    seen = start_idx
    for ex in ds:
        if seen >= n:
            break
        audio_data = ex["audio"]
        raw = audio_data.get("bytes") or audio_data.get("path")
        if raw is None:
            continue
        try:
            if isinstance(raw, bytes):
                arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            else:
                arr, sr = sf.read(raw, dtype="float32", always_2d=False)
        except Exception as e:
            print(f"  skip: {e}")
            continue
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if len(arr) / sr < 1.0:
            continue
        text = (ex.get(transcription_key) or ex.get("sentence") or ex.get("raw_transcription", "")).strip()
        if not text:
            continue
        fname = f"{label_prefix}_{seen:03d}.wav"
        fpath = os.path.join(out_dir, fname)
        save_wav(fpath, arr, sr)
        rows.append({
            "audio_file": os.path.join("data", "fleurs_500", label_prefix, fname),
            "system": "Perform ASR.",
            "target": text,
            "type": f"fleurs_{label_prefix}",
        })
        seen += 1
        if seen % 100 == 0:
            print(f"  {seen}/{n}")

    # For the pre-existing rows we need the target text — re-stream to get it
    # Simpler: rebuild from scratch if cache incomplete
    if any(r["target"] is None for r in rows):
        print(f"  WARNING: some existing rows missing target text; these will be skipped.")
        rows = [r for r in rows if r["target"] is not None]

    print(f"  Done: {len(rows)} rows")
    return rows


def sample_synthetic(existing_train, existing_eval, type_name, n, rng):
    """Sample n rows of given type from existing train+eval, deduplicating by audio_file."""
    all_rows = []
    for path in [existing_train, existing_eval]:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("type") == type_name:
                    all_rows.append(r)
    # Deduplicate by audio_file
    seen = set()
    deduped = []
    for r in all_rows:
        af = r["audio_file"]
        if af not in seen:
            seen.add(af)
            deduped.append(r)
    rng.shuffle(deduped)
    sampled = deduped[:n]
    print(f"  {type_name}: {len(deduped)} unique → sampled {len(sampled)}")
    return sampled


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Written {len(rows)} rows → {path}")


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    root = args.data_root

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    # ── Download FLEURS ───────────────────────────────────────────────────────
    fleurs_root = os.path.join(root, "data", "fleurs_500")

    en_rows = download_fleurs(
        "google/fleurs", "en_us", "train",
        os.path.join(fleurs_root, "en"), "en", args.n_fleurs,
    )
    ja_rows = download_fleurs(
        "google/fleurs", "ja_jp", "train",
        os.path.join(fleurs_root, "ja"), "ja", args.n_fleurs,
    )
    try:
        cs_rows = download_fleurs(
            "skit-ai/cs-fleurs", "default", "train",
            os.path.join(fleurs_root, "cs"), "cs", args.n_fleurs,
            transcription_key="transcription",
        )
    except Exception as e:
        print(f"  cs-fleurs failed ({e}), using extra JA as fallback")
        cs_rows = download_fleurs(
            "google/fleurs", "ja_jp", "validation",
            os.path.join(fleurs_root, "cs"), "cs", args.n_fleurs,
        )

    # ── Sample synthetic ──────────────────────────────────────────────────────
    train_jl = os.path.join(root, args.existing_train)
    eval_jl  = os.path.join(root, args.existing_eval)

    print("\nSampling synthetic data...")
    syn_cs = sample_synthetic(train_jl, eval_jl, "cs_native",  args.n_synthetic, rng)
    syn_en = sample_synthetic(train_jl, eval_jl, "en_native",  args.n_synthetic, rng)
    syn_ja = sample_synthetic(train_jl, eval_jl, "ja_native",  args.n_synthetic, rng)

    # ── Combine & split 90/10 ─────────────────────────────────────────────────
    all_rows = en_rows + ja_rows + cs_rows + syn_cs + syn_en + syn_ja
    rng.shuffle(all_rows)
    cut = int(len(all_rows) * 0.9)
    train_rows = all_rows[:cut]
    eval_rows  = all_rows[cut:]

    print(f"\nTotal: {len(all_rows)} → train {len(train_rows)} / eval {len(eval_rows)}")
    print(f"  FLEURS EN: {len(en_rows)}  JA: {len(ja_rows)}  CS: {len(cs_rows)}")
    print(f"  Synthetic CS: {len(syn_cs)}  EN: {len(syn_en)}  JA: {len(syn_ja)}")

    write_jsonl(os.path.join(root, args.out_train), train_rows)
    write_jsonl(os.path.join(root, args.out_eval),  eval_rows)
    print("\nDone. Next: run preprocess_asr_data.py on balanced_train/eval.jsonl")


if __name__ == "__main__":
    main()
