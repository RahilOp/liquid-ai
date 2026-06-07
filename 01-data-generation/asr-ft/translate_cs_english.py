"""
Translates all CS samples' all_english field into proper fluent English
using an OpenAI-compatible Qwen vLLM endpoint (set base_url below)
Then rebuilds the training JSONL.
"""

import json, sys, io, time, random
from pathlib import Path
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CLIENT = OpenAI(
    base_url="http://localhost:9000/v1",
    api_key="none",
)
MODEL = "Qwen/Qwen3.5-122B-A10B"

CS_META   = Path("data/code_switching/metadata/transcriptions.json")
TRAIN_OUT = Path("data/training")

SYS_NATIVE = "Transcribe the audio."
SYS_JA     = "Transcribe in Japanese."
SYS_EN     = "Transcribe in English."

SYSTEM_PROMPT = (
    "You are a professional translator. "
    "You will receive a short Japanese-English code-switched sentence. "
    "Translate it into one fluent, natural English sentence. "
    "Do not add explanations, notes, or punctuation beyond the sentence. "
    "Output only the English translation."
)

def translate(text: str, retries: int = 3) -> str:
    for attempt in range(1, retries + 1):
        try:
            resp = CLIENT.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text},
                ],
                max_tokens=128,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [attempt {attempt}/{retries}] error: {e}")
            if attempt < retries:
                time.sleep(2 * attempt)
    return text   # fallback to original if all retries fail


def main():
    samples = json.loads(CS_META.read_text(encoding="utf-8"))
    total   = len(samples)
    print(f"Translating {total} CS samples via the Qwen vLLM endpoint...\n")

    for i, s in enumerate(samples, 1):
        full = s.get("full_transcription", "")
        print(f"[{i:03d}/{total}] {s['id']}  |  {full[:60]}")
        translation = translate(full)
        s["all_english"] = translation
        print(f"         → {translation[:80]}")

        # Save every 50 samples as checkpoint
        if i % 50 == 0:
            CS_META.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  💾 Checkpoint saved at {i}/{total}")

    # Final save of metadata
    CS_META.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ All {total} translations done. Metadata saved.")

    # Rebuild JSONL
    print("\nRebuilding training JSONL...")
    random.seed(77)

    def load_cs():
        ex = []
        for s in samples:
            af = s["audio_file"]
            ex.append({"audio_file": af, "system": SYS_NATIVE, "target": s["full_transcription"], "type": "cs_native"})
            ex.append({"audio_file": af, "system": SYS_JA,     "target": s["all_japanese"],       "type": "cs_ja"})
            ex.append({"audio_file": af, "system": SYS_EN,     "target": s["all_english"],         "type": "cs_en"})
        return ex

    def load_en():
        p = json.loads(Path("data/english_only/metadata/transcriptions.json").read_text(encoding="utf-8"))
        ex = []
        for s in p:
            af = s["audio_file"]
            ex.append({"audio_file": af, "system": SYS_NATIVE, "target": s["english_transcription"], "type": "en_native"})
            ex.append({"audio_file": af, "system": SYS_EN,     "target": s["english_transcription"], "type": "en_en"})
            ex.append({"audio_file": af, "system": SYS_JA,     "target": s["japanese_translation"],  "type": "en_ja"})
        return ex

    def load_ja():
        p = json.loads(Path("data/japanese_only/metadata/transcriptions.json").read_text(encoding="utf-8"))
        ex = []
        for s in p:
            af = s["audio_file"]
            ex.append({"audio_file": af, "system": SYS_NATIVE, "target": s["japanese_transcription"], "type": "ja_native"})
            ex.append({"audio_file": af, "system": SYS_JA,     "target": s["japanese_transcription"], "type": "ja_ja"})
            ex.append({"audio_file": af, "system": SYS_EN,     "target": s["english_translation"],    "type": "ja_en"})
        return ex

    all_ex = load_cs() + load_en() + load_ja()
    random.shuffle(all_ex)
    n_eval = max(1, int(len(all_ex) * 0.1))
    train, eval_ = all_ex[n_eval:], all_ex[:n_eval]

    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    (TRAIN_OUT / "train.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in train), encoding="utf-8"
    )
    (TRAIN_OUT / "eval.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in eval_), encoding="utf-8"
    )
    print(f"Train: {len(train)} | Eval: {len(eval_)}")
    print("✅ Done — dataset ready for training.")


if __name__ == "__main__":
    main()
