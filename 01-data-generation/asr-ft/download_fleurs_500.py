"""
Download 500 samples each from FLEURS EN, JA, and CS-FLEURS.
Saves WAV files + a meta.jsonl per language so transcriptions survive restarts.

Usage:
  python download_fleurs_500.py --data-root /awshesh/lfm2.5/awshesh --hf-token hf_xxx
"""
from __future__ import annotations

import argparse, io, json, os
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/awshesh/lfm2.5/awshesh")
    p.add_argument("--hf-token",  default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--n",         type=int, default=500)
    return p.parse_args()


def save_wav(path, arr, sr):
    arr = arr.astype(np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    peak = np.max(np.abs(arr))
    if peak > 0:
        arr = arr / peak * 0.9
    sf.write(path, arr, sr, subtype="PCM_16")


def download_lang(dataset_name, config, split, out_dir, prefix, n, transcription_key="transcription"):
    from datasets import load_dataset, Audio as HfAudio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.jsonl"

    # Load already-saved transcriptions
    saved = {}
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                r = json.loads(line)
                saved[r["fname"]] = r["text"]

    print(f"\n{dataset_name} ({config}) — have {len(saved)}/{n} cached")
    if len(saved) >= n:
        print("  Already complete.")
        rows = []
        for fname, text in list(saved.items())[:n]:
            rel = os.path.join("data", "fleurs_500", prefix, fname)
            rows.append({"audio_file": rel, "system": "Perform ASR.", "target": text, "type": f"fleurs_{prefix}"})
        return rows

    ds = load_dataset(dataset_name, config, split=split, streaming=True)
    ds = ds.cast_column("audio", HfAudio(decode=False))

    meta_f = open(meta_path, "a", buffering=1)  # line-buffered
    seen = len(saved)

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

        fname = f"{prefix}_{seen:03d}.wav"
        wav_path = out_dir / fname
        if not wav_path.exists():
            save_wav(str(wav_path), arr, sr)

        # Save transcription immediately
        meta_f.write(json.dumps({"fname": fname, "text": text}, ensure_ascii=False) + "\n")
        saved[fname] = text
        seen += 1
        if seen % 100 == 0:
            print(f"  {seen}/{n}", flush=True)

    meta_f.close()
    print(f"  Done: {len(saved)} rows")

    rows = []
    for fname, text in list(saved.items())[:n]:
        rel = os.path.join("data", "fleurs_500", prefix, fname)
        rows.append({"audio_file": rel, "system": "Perform ASR.", "target": text, "type": f"fleurs_{prefix}"})
    return rows


def main():
    args = parse_args()
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    root = Path(args.data_root)
    base = root / "data" / "fleurs_500"

    # EN + JA train splits
    en_rows = download_lang("google/fleurs", "en_us", "train", base / "en", "en", args.n)
    ja_rows = download_lang("google/fleurs", "ja_jp", "train", base / "ja", "ja", args.n)

    # CS-FLEURS: combine train + validation + test to reach 500
    cs_rows = []
    cs_target = args.n
    for split in ["train", "validation", "test"]:
        if len(cs_rows) >= cs_target:
            break
        needed = cs_target - len(cs_rows)
        suffix = split[:3]  # "tra", "val", "tes" → use split name as subdir
        try:
            new_rows = download_lang(
                "skit-ai/cs-fleurs", "default", split,
                base / f"cs_{split}", f"cs_{split}", needed,
            )
            cs_rows.extend(new_rows)
            print(f"  cs-fleurs {split}: {len(new_rows)} rows (total so far: {len(cs_rows)})")
        except Exception as e:
            print(f"  cs-fleurs {split} failed: {e}")

    # EN + JA validation splits for extra diversity
    en_val_rows = download_lang("google/fleurs", "en_us", "validation", base / "en_val", "en_val", args.n)
    ja_val_rows = download_lang("google/fleurs", "ja_jp", "validation", base / "ja_val", "ja_val", args.n)

    all_rows = en_rows + ja_rows + cs_rows + en_val_rows + ja_val_rows
    print(f"\nFLEURS download complete:")
    print(f"  EN train: {len(en_rows)}  JA train: {len(ja_rows)}  CS total: {len(cs_rows)}")
    print(f"  EN val:   {len(en_val_rows)}  JA val:   {len(ja_val_rows)}")
    print(f"  Total:    {len(all_rows)}")

    out = root / "data" / "training" / "fleurs500_rows.jsonl"
    with open(out, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(all_rows)} rows → {out}")


if __name__ == "__main__":
    main()
