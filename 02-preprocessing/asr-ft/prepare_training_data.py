"""
Converts generated audio + metadata into a flat JSONL training file.

For each audio sample, creates 3 training examples (one per task):
  - "Transcribe."             → output in native/detected language
  - "Transcribe in Japanese." → all output in Japanese
  - "Transcribe in English."  → all output in English

Output: data/training/train.jsonl, data/training/eval.jsonl
Each line:
  {"audio_file": "...", "system": "...", "target": "..."}
"""

import json
import random
from pathlib import Path

random.seed(77)

DATA_BASE  = Path("data")
TRAIN_OUT  = DATA_BASE / "training"
TRAIN_OUT.mkdir(parents=True, exist_ok=True)

EVAL_RATIO = 0.1   # 10% held out for eval

SYS_NATIVE = "Transcribe the audio."
SYS_JA     = "Transcribe in Japanese."
SYS_EN     = "Transcribe in English."


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_cs() -> list[dict]:
    path = DATA_BASE / "code_switching" / "metadata" / "transcriptions.json"
    if not path.exists():
        print(f"  [SKIP] {path} not found yet")
        return []
    samples = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for s in samples:
        af = s["audio_file"]
        if not Path(af).exists():
            continue
        examples.append({"audio_file": af, "system": SYS_NATIVE, "target": s["full_transcription"], "type": "cs_native"})
        examples.append({"audio_file": af, "system": SYS_JA,     "target": s["all_japanese"],       "type": "cs_ja"})
        examples.append({"audio_file": af, "system": SYS_EN,     "target": s["all_english"],         "type": "cs_en"})
    print(f"  CS : {len(samples)} audio → {len(examples)} examples")
    return examples


def load_en() -> list[dict]:
    path = DATA_BASE / "english_only" / "metadata" / "transcriptions.json"
    if not path.exists():
        print(f"  [SKIP] {path} not found yet")
        return []
    samples = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for s in samples:
        af = s["audio_file"]
        if not Path(af).exists():
            continue
        examples.append({"audio_file": af, "system": SYS_NATIVE, "target": s["english_transcription"], "type": "en_native"})
        examples.append({"audio_file": af, "system": SYS_EN,     "target": s["english_transcription"], "type": "en_en"})
        examples.append({"audio_file": af, "system": SYS_JA,     "target": s["japanese_translation"],  "type": "en_ja"})
    print(f"  EN : {len(samples)} audio → {len(examples)} examples")
    return examples


def load_ja() -> list[dict]:
    path = DATA_BASE / "japanese_only" / "metadata" / "transcriptions.json"
    if not path.exists():
        print(f"  [SKIP] {path} not found yet")
        return []
    samples = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for s in samples:
        af = s["audio_file"]
        if not Path(af).exists():
            continue
        examples.append({"audio_file": af, "system": SYS_NATIVE, "target": s["japanese_transcription"], "type": "ja_native"})
        examples.append({"audio_file": af, "system": SYS_JA,     "target": s["japanese_transcription"], "type": "ja_ja"})
        examples.append({"audio_file": af, "system": SYS_EN,     "target": s["english_translation"],    "type": "ja_en"})
    print(f"  JA : {len(samples)} audio → {len(examples)} examples")
    return examples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading datasets...")
    all_examples = load_cs() + load_en() + load_ja()

    if not all_examples:
        print("No examples found. Run data generation first.")
        return

    random.shuffle(all_examples)

    n_eval  = max(1, int(len(all_examples) * EVAL_RATIO))
    n_train = len(all_examples) - n_eval
    train   = all_examples[:n_train]
    eval_   = all_examples[n_train:]

    # ── Write JSONL ──────────────────────────────────────────────────────────
    train_path = TRAIN_OUT / "train.jsonl"
    eval_path  = TRAIN_OUT / "eval.jsonl"

    train_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in train),
        encoding="utf-8"
    )
    eval_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in eval_),
        encoding="utf-8"
    )

    # ── Stats ────────────────────────────────────────────────────────────────
    from collections import Counter
    train_types = Counter(e["type"] for e in train)
    print(f"\nTrain: {n_train} examples → {train_path}")
    print(f"Eval : {n_eval}  examples → {eval_path}")
    print("\nTrain type breakdown:")
    for t, c in sorted(train_types.items()):
        print(f"  {t:<12}: {c}")

    # ── Sample preview ───────────────────────────────────────────────────────
    print("\nSample training examples:")
    for ex in random.sample(train, min(3, len(train))):
        print(f"\n  audio : {ex['audio_file']}")
        print(f"  system: {ex['system']}")
        print(f"  target: {ex['target'][:80]}{'...' if len(ex['target'])>80 else ''}")
        print(f"  type  : {ex['type']}")


if __name__ == "__main__":
    main()
