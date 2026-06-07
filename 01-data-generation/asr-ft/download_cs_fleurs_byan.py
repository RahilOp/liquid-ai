"""
Download Japanese<->English code-switching data from byan/cs-fleurs.

Only the `jpn-eng` subset is used (the only JP/EN code-switching pair).
Sources:
  read/test   -> 196 REAL human recordings (clean text, no markup)
  xtts/train  -> 2097 synthetic TTS  (English wrapped in **...**, stripped)
  xtts/test1  -> 650  synthetic TTS

All real + a sample of synthetic go into TRAINING (type="fleurs_cs").

Usage:
  python download_cs_fleurs_byan.py --data-root /awshesh/lfm2.5/awshesh \
      --hf-token hf_xxx --n-xtts 500
"""
from __future__ import annotations

import argparse, json, os, random, re, shutil
from pathlib import Path

REPO = "byan/cs-fleurs"
LANG = "jpn-eng"
# metadata dir -> nothing extra; full repo path = f"{meta_dir}/{file_name}"
SOURCES = {
    "read/test":  "real",
    "xtts/train": "syn",
    "xtts/test1": "syn",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/awshesh/lfm2.5/awshesh")
    p.add_argument("--hf-token",  default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--n-xtts",    type=int, default=500,
                   help="How many synthetic xtts jpn-eng samples to add (real read are ALL kept)")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def clean_text(t: str) -> str:
    # strip the **English** code-switch markers used in the synthetic splits
    t = t.replace("**", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    args = parse_args()
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
    from huggingface_hub import hf_hub_download

    rng = random.Random(args.seed)
    root = Path(args.data_root)
    out_audio = root / "data" / "cs_fleurs_byan"
    out_audio.mkdir(parents=True, exist_ok=True)

    # 1) collect jpn-eng rows per source from each metadata.jsonl
    pools = {"real": [], "syn": []}
    for meta_dir, kind in SOURCES.items():
        mf = hf_hub_download(REPO, f"{meta_dir}/metadata.jsonl",
                             repo_type="dataset", token=args.hf_token)
        rows = [json.loads(l) for l in open(mf, encoding="utf-8") if l.strip()]
        rows = [r for r in rows if r.get("language") == LANG]
        for r in rows:
            r["_meta_dir"] = meta_dir
        pools[kind].extend(rows)
        print(f"{meta_dir}: {len(rows)} jpn-eng ({kind})")

    # 2) keep ALL real, sample n-xtts synthetic
    real_rows = pools["real"]
    syn_rows = pools["syn"]
    rng.shuffle(syn_rows)
    syn_rows = syn_rows[: args.n_xtts]
    selected = real_rows + syn_rows
    print(f"\nSelected: real={len(real_rows)}  synthetic={len(syn_rows)}  total={len(selected)}")

    # 3) download each audio file, copy into data/cs_fleurs_byan/, write rows
    out_rows = []
    ok = fail = 0
    for i, r in enumerate(selected):
        repo_path = f"{r['_meta_dir']}/{r['file_name']}".replace("//", "/")
        try:
            cached = hf_hub_download(REPO, repo_path, repo_type="dataset", token=args.hf_token)
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  fail {repo_path}: {e}")
            continue
        fname = f"csf_{i:04d}.wav"
        dest = out_audio / fname
        if not dest.exists():
            shutil.copy(cached, dest)
        text = clean_text(r["text"])
        if not text:
            continue
        out_rows.append({
            "audio_file": os.path.join("data", "cs_fleurs_byan", fname),
            "system": "Perform ASR.",
            "target": text,
            "type": "fleurs_cs",
        })
        ok += 1
        if ok % 100 == 0:
            print(f"  {ok}/{len(selected)}", flush=True)

    print(f"\nDownloaded ok={ok}  fail={fail}")
    out = root / "data" / "training" / "cs_fleurs_byan_rows.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(out_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
