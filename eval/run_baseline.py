#!/usr/bin/env python3
"""Run ONE ASR baseline over a manifest and write predictions for scoring.

Models:
  lfm     : LiquidAI/LFM2.5-Audio-1.5B-JP (or --model-id), ASR via ChatState +
            generate_sequential with system prompt "Perform ASR." (text path only).
  whisper : faster-whisper large-v3 (language auto-detected).

Input manifest (JSONL), one obj per line, as produced by data/load_csfleurs.py:
    {"id": "...", "reference": "...", "audio": "<path rel to --audio-root>", ...}

Output predictions (JSONL) ready for eval/score.py:
    {"id": "...", "reference": "...", "hypothesis": "<asr output>"}

Examples:
    python eval/run_baseline.py --model lfm \
        --manifest data/csfleurs/read/manifest.jsonl \
        --audio-root data/csfleurs --out artifacts/preds/lfm_read.jsonl --limit 20

    python eval/run_baseline.py --model whisper \
        --manifest data/csfleurs/read/manifest.jsonl \
        --audio-root data/csfleurs --out artifacts/preds/whisper_read.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ASR_SYSTEM_PROMPT = "Perform ASR."
LFM_JP = "LiquidAI/LFM2.5-Audio-1.5B-JP"
LFM_BASE = "LiquidAI/LFM2.5-Audio-1.5B"
IM_END_TOKEN = 7  # <|im_end|>


def read_manifest(path: str, audio_root: str, limit: int | None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["_audio_path"] = os.path.join(audio_root, r["audio"])
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def load_audio_mono(path: str):
    """-> (np.float32 mono array, sample_rate)."""
    import soundfile as sf
    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, sr


# ---------------------------------------------------------------------------
# LFM2.5-Audio
# ---------------------------------------------------------------------------


def run_lfm(rows: list[dict], model_id: str, max_new_tokens: int) -> list[dict]:
    import torch
    from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

    print(f"[lfm] loading processor+model: {model_id}")
    proc = LFM2AudioProcessor.from_pretrained(model_id)
    proc.eval()
    model = LFM2AudioModel.from_pretrained(model_id).eval()
    print("[lfm] loaded.")

    out = []
    for i, r in enumerate(rows):
        arr, sr = load_audio_mono(r["_audio_path"])
        wave = torch.tensor(arr, dtype=torch.float).unsqueeze(0)  # [1, T]

        chat = ChatState(proc)
        chat.new_turn("system")
        chat.add_text(ASR_SYSTEM_PROMPT)
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(wave, sr)
        chat.end_turn()
        chat.new_turn("assistant")

        toks = []
        with torch.no_grad():
            for t in model.generate_sequential(**chat, max_new_tokens=max_new_tokens):
                if t.numel() == 1:  # text token (ASR path)
                    tid = int(t.reshape(-1)[0].item())
                    if tid == IM_END_TOKEN:
                        break
                    toks.append(tid)
                # audio frames (numel 8) ignored — shouldn't occur for ASR
        hyp = proc.text.decode(toks, skip_special_tokens=True).strip() if toks else ""
        out.append({"id": r["id"], "reference": r["reference"], "hypothesis": hyp})
        if i < 5 or i % 20 == 0:
            print(f"  [{i+1}/{len(rows)}] {r['id']}\n     REF: {r['reference'][:90]}\n     HYP: {hyp[:90]}")
    return out


# ---------------------------------------------------------------------------
# Whisper (faster-whisper)
# ---------------------------------------------------------------------------


def run_whisper(rows: list[dict], model_id: str, device: str) -> list[dict]:
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[whisper] loading {model_id} on {device} ({compute_type})")
    model = WhisperModel(model_id, device=device, compute_type=compute_type)
    print("[whisper] loaded.")

    out = []
    for i, r in enumerate(rows):
        segments, info = model.transcribe(r["_audio_path"], beam_size=5)  # language auto
        hyp = "".join(seg.text for seg in segments).strip()
        out.append({"id": r["id"], "reference": r["reference"], "hypothesis": hyp})
        if i < 5 or i % 20 == 0:
            print(f"  [{i+1}/{len(rows)}] {r['id']} (det_lang={info.language})"
                  f"\n     REF: {r['reference'][:90]}\n     HYP: {hyp[:90]}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one ASR baseline")
    ap.add_argument("--model", required=True, choices=["lfm", "whisper"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", default=".", help="Prefix for manifest 'audio' paths")
    ap.add_argument("--out", required=True, help="Output predictions JSONL")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model-id", default=None,
                    help="Override HF id (lfm default JP variant; whisper default large-v3)")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="LFM only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fallback-base", action="store_true",
                    help="If JP LFM fails to load, retry with the non-JP base model")
    args = ap.parse_args(argv)

    rows = read_manifest(args.manifest, args.audio_root, args.limit)
    print(f"Loaded {len(rows)} manifest rows from {args.manifest}")
    if not rows:
        raise SystemExit("Empty manifest — nothing to run.")

    t0 = time.time()
    if args.model == "lfm":
        model_id = args.model_id or LFM_JP
        try:
            preds = run_lfm(rows, model_id, args.max_new_tokens)
        except Exception as e:  # noqa: BLE001
            if args.fallback_base and model_id != LFM_BASE:
                print(f"[lfm] {type(e).__name__}: {e}\n[lfm] falling back to {LFM_BASE}")
                preds = run_lfm(rows, LFM_BASE, args.max_new_tokens)
            else:
                raise
    else:
        model_id = args.model_id or "large-v3"
        preds = run_whisper(rows, model_id, args.device)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    dt = time.time() - t0
    print(f"\nWrote {len(preds)} predictions -> {args.out}  ({dt:.1f}s, "
          f"{dt/max(len(preds),1):.2f}s/utt)")
    print(f"Score with:\n  python eval/score.py {args.model}:{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
