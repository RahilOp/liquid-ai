"""
Combine 500 FLEURS per language + 300 synthetic per type into balanced JSONL.
Run after download_fleurs_500.py has completed.

Usage:
  python combine_balanced.py --data-root /awshesh/lfm2.5/awshesh
"""
import argparse, collections, json, random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/awshesh/lfm2.5/awshesh")
    p.add_argument("--n-synthetic", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    root = Path(args.data_root)

    # Load 500 FLEURS rows (saved by download_fleurs_500.py)
    fleurs_path = root / "data" / "training" / "fleurs500_rows.jsonl"
    fleurs_rows = [json.loads(l) for l in open(fleurs_path) if l.strip()]
    by_fleurs = collections.Counter(r["type"] for r in fleurs_rows)
    print(f"FLEURS loaded: {dict(by_fleurs)}")

    # Load REAL CS-FLEURS jpn-eng rows (saved by download_cs_fleurs_byan.py)
    cs_fleurs_rows = []
    csf_path = root / "data" / "training" / "cs_fleurs_byan_rows.jsonl"
    if csf_path.exists():
        cs_fleurs_rows = [json.loads(l) for l in open(csf_path) if l.strip()]
        print(f"CS-FLEURS (byan jpn-eng) loaded: {len(cs_fleurs_rows)} rows")
    else:
        print("WARNING: cs_fleurs_byan_rows.jsonl not found — no real CS-FLEURS added")

    # Load synthetic from existing native_asr JSONL (deduplicated)
    all_syn = []
    for p in ["data/training/native_asr_train.jsonl", "data/training/native_asr_eval.jsonl"]:
        with open(root / p) as f:
            all_syn.extend(json.loads(l) for l in f if l.strip())

    by_syn = collections.defaultdict(list)
    seen_af = set()
    for r in all_syn:
        t = r["type"]
        if t.startswith("fleurs"):
            continue
        if r["audio_file"] not in seen_af:
            seen_af.add(r["audio_file"])
            by_syn[t].append(r)
    print(f"Synthetic unique: { {k: len(v) for k, v in by_syn.items()} }")

    n = args.n_synthetic
    # Use ALL cs_native (code-switching is the primary task)
    syn_cs = by_syn.get("cs_native", [])
    syn_en = rng.sample(by_syn.get("en_native", []), min(n, len(by_syn.get("en_native", []))))
    syn_ja = rng.sample(by_syn.get("ja_native", []), min(n, len(by_syn.get("ja_native", []))))
    print(f"Synthetic: CS={len(syn_cs)} (all) EN={len(syn_en)} JA={len(syn_ja)}")

    combined = fleurs_rows + cs_fleurs_rows + syn_cs + syn_en + syn_ja
    rng.shuffle(combined)
    cut = int(len(combined) * 0.9)
    train_rows = combined[:cut]
    eval_rows  = combined[cut:]
    print(f"Total: {len(combined)} -> train {len(train_rows)} / eval {len(eval_rows)}")

    train_path = root / "data" / "training" / "balanced_train.jsonl"
    eval_path  = root / "data" / "training" / "balanced_eval.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written: {train_path}")
    print(f"Written: {eval_path}")


if __name__ == "__main__":
    main()
