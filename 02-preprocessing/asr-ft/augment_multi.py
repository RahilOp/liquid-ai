#!/usr/bin/env python3
"""Multi-type acoustic augmentation for the synthetic CS corpus (issue #8).

Takes clean engine manifests and produces augmented copies spanning several
distortion families — deliberately "beyond MUSAN noise":

  noise      additive MUSAN noise at a swept SNR
  music      additive MUSAN music at a swept SNR
  babble     several MUSAN speech files summed -> multi-speaker babble
  reverb     convolution with a room impulse response (RIRS_NOISES)
  telephony  300-3400 Hz band-limit + G.711 mu-law codec round-trip

The reference transcript never changes — augmentation alters acoustics, not
words — so every augmented row keeps the source row's transcript / en_words /
segments and just swaps audio_path and records the augmentation applied.

Run (on mactrn01, in the venv):
  .venv/bin/python scripts/02-preprocessing/asr-ft/augment_multi.py \
     --manifests data/synth/edge_tts/manifest.jsonl \
                 data/synth/kokoro/manifest.jsonl \
                 data/synth/melo/manifest.jsonl \
     --musan  data/noise_sources/musan \
     --rir    data/noise_sources/RIRS_NOISES \
     --out    data/augmented --seed 42
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt, resample_poly

SR = 16000

# --- I/O ----------------------------------------------------------------------

def load_wav(path: str, target_sr: int = SR) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != target_sr:
        a = resample_poly(a, target_sr, sr).astype(np.float32)
    return a


def peak_norm(a: np.ndarray, peak: float = 0.97) -> np.ndarray:
    m = float(np.max(np.abs(a))) if a.size else 0.0
    return (a * (peak / m)).astype(np.float32) if m > peak else a.astype(np.float32)


# --- augmentation primitives --------------------------------------------------

def _fit_len(noise: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
    if len(noise) < n:
        noise = np.tile(noise, int(np.ceil(n / len(noise))))
    if len(noise) > n:
        s = rng.randint(0, len(noise) - n)
        noise = noise[s:s + n]
    return noise


def add_at_snr(sig: np.ndarray, noise: np.ndarray, snr_db: float,
               rng: random.Random) -> np.ndarray:
    noise = _fit_len(noise, len(sig), rng)
    s_rms = np.sqrt(np.mean(sig ** 2)) + 1e-9
    n_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
    noise = noise * ((s_rms / (10 ** (snr_db / 20))) / n_rms)
    return peak_norm(sig + noise)


def make_babble(speech_pool: list[str], n: int, k: int,
                rng: random.Random) -> np.ndarray:
    """Sum k random speech files into a multi-speaker babble bed of length n."""
    bed = np.zeros(n, dtype=np.float32)
    for _ in range(k):
        sp = load_wav(rng.choice(speech_pool))
        bed += _fit_len(sp, n, rng)
    return bed


def apply_reverb(sig: np.ndarray, rir_path: str) -> np.ndarray:
    rir = load_wav(rir_path)
    if rir.size == 0:
        return sig
    rir = rir / (np.max(np.abs(rir)) + 1e-9)
    wet = fftconvolve(sig, rir)[:len(sig)]        # keep original length
    return peak_norm(wet)


def apply_telephony(sig: np.ndarray) -> np.ndarray:
    """Band-limit to 300-3400 Hz then mu-law (G.711) encode/decode."""
    sos = butter(6, [300 / (SR / 2), 3400 / (SR / 2)], btype="band", output="sos")
    band = sosfilt(sos, sig).astype(np.float32)
    mu = 255.0
    band = np.clip(band, -1.0, 1.0)
    comp = np.sign(band) * np.log1p(mu * np.abs(band)) / np.log1p(mu)   # encode
    q = np.round((comp + 1) / 2 * mu)                                   # 8-bit
    comp = q / mu * 2 - 1
    dec = np.sign(comp) * (1 / mu) * ((1 + mu) ** np.abs(comp) - 1)     # decode
    return peak_norm(dec.astype(np.float32))


# --- driver -------------------------------------------------------------------

def collect(root: str, sub: str) -> list[str]:
    files = glob.glob(os.path.join(root, sub, "**", "*.wav"), recursive=True)
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--musan", required=True, help="MUSAN root (noise/ music/ speech/)")
    ap.add_argument("--rir", required=True, help="RIRS_NOISES root")
    ap.add_argument("--out", default="data/augmented")
    ap.add_argument("--snr-sweep", nargs="+", type=float, default=[20, 15, 10, 5])
    ap.add_argument("--babble-k", type=int, default=4, help="speakers per babble bed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    noise_pool = collect(args.musan, "noise")
    music_pool = collect(args.musan, "music")
    speech_pool = collect(args.musan, "speech")
    rir_pool = glob.glob(os.path.join(args.rir, "**", "*.wav"), recursive=True)
    for name, pool in [("noise", noise_pool), ("music", music_pool),
                       ("speech", speech_pool), ("rir", rir_pool)]:
        assert pool, f"empty {name} pool under given root"
    print(f"pools: noise={len(noise_pool)} music={len(music_pool)} "
          f"speech={len(speech_pool)} rir={len(rir_pool)}")

    out = Path(args.out)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"

    written = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        for man in args.manifests:
            rows = [json.loads(l) for l in Path(man).read_text(encoding="utf-8").splitlines() if l.strip()]
            print(f"[{man}] {len(rows)} clean rows")
            for row in rows:
                sig = load_wav(row["audio_path"])
                if sig.size == 0:
                    continue
                # one augmentation of each family per clip (SNR/file drawn at random)
                snr_n = rng.choice(args.snr_sweep)
                snr_m = rng.choice(args.snr_sweep)
                snr_b = rng.choice(args.snr_sweep)
                plan = [
                    (f"noise_snr{int(snr_n)}", add_at_snr(sig, load_wav(rng.choice(noise_pool)), snr_n, rng)),
                    (f"music_snr{int(snr_m)}", add_at_snr(sig, load_wav(rng.choice(music_pool)), snr_m, rng)),
                    (f"babble_snr{int(snr_b)}", add_at_snr(sig, make_babble(speech_pool, len(sig), args.babble_k, rng), snr_b, rng)),
                    ("reverb", apply_reverb(sig, rng.choice(rir_pool))),
                    ("telephony", apply_telephony(sig)),
                ]
                for aug_name, aug_audio in plan:
                    uid = f"{row['id']}_{aug_name}"
                    wav_path = audio_dir / f"{uid}.wav"
                    sf.write(str(wav_path), aug_audio, SR, subtype="PCM_16")
                    new = dict(row)
                    new.update({
                        "id": uid,
                        "audio_path": str(wav_path.resolve()),
                        "duration_s": round(len(aug_audio) / SR, 3),
                        "augmentation": aug_name,
                        "clean_id": row["id"],
                        "source": f"augmented/{row.get('engine','?')}",
                    })
                    mf.write(json.dumps(new, ensure_ascii=False) + "\n")
                    written += 1

    from collections import Counter
    kinds = Counter(json.loads(l)["augmentation"].split("_")[0]
                    for l in manifest_path.read_text(encoding="utf-8").splitlines())
    print(f"\ndone: {written} augmented clips -> {manifest_path}")
    print(f"  by family: {dict(kinds)}")


if __name__ == "__main__":
    main()
