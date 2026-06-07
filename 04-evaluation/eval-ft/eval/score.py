#!/usr/bin/env python3
"""Code-switching ASR scoring harness.

Reads one JSONL of {id, reference, hypothesis} per model and emits a metrics
table with:
  - MER       : Mixed Error Rate over a mixed token stream
                (each Japanese char = 1 token, each English word = 1 token).
  - JA-CER    : Character Error Rate computed over the Japanese-only character
                stream of ref vs hyp (Japanese has no word boundaries -> chars).
  - EN-WER    : Word Error Rate computed over the English (Latin) word stream.
  - ScriptAcc : of the English words in the reference, the fraction the model
                rendered in Latin script (vs. wrongly forced into katakana or
                dropped). This is the headline "cloud can't do this" metric.

Identical normalization is applied to references AND every model's hypotheses
before scoring. All error rates are corpus-aggregated (sum of edits / sum of
reference tokens), which is the standard way to pool WER/CER across utterances.

Usage:
    python eval/score.py whisper:preds_whisper.jsonl lfm:preds_lfm.jsonl
    python eval/score.py preds.jsonl                       # label from filename
    python eval/score.py --no-fillers whisper:preds.jsonl  # keep English fillers
    python eval/score.py whisper:preds.jsonl --json out.json

Each input line: {"id": "...", "reference": "...", "hypothesis": "..."}
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

try:
    import regex as _re  # better Unicode property support if available
except Exception:  # pragma: no cover - stdlib fallback
    import re as _re

# ---------------------------------------------------------------------------
# Character / token classification
# ---------------------------------------------------------------------------

# English fillers removed by default (whole-token). We deliberately do NOT strip
# Japanese fillers (あの, えーと, ...) by regex: without morphological analysis
# that corrupts legitimate Japanese substrings. Documented choice.
ENGLISH_FILLERS = {"um", "uh", "uhh", "umm", "er", "err", "erm", "hmm", "mm"}


def _is_kana(ch: str) -> bool:
    o = ord(ch)
    # Hiragana, Katakana, Katakana phonetic extensions, half-width katakana
    return (0x3040 <= o <= 0x309F) or (0x30A0 <= o <= 0x30FF) or (0xFF66 <= o <= 0xFF9D)


def _is_katakana(ch: str) -> bool:
    o = ord(ch)
    return (0x30A0 <= o <= 0x30FF) or (0xFF66 <= o <= 0xFF9D)


def _is_han(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF) or (0x3400 <= o <= 0x4DBF) or (0xF900 <= o <= 0xFAFF)


def _is_japanese_char(ch: str) -> bool:
    # Prolonged sound mark ー (30FC) handled by katakana range.
    return _is_kana(ch) or _is_han(ch)


def _is_latin_letter(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


# A token is an "English word" if it is made of Latin letters (with optional
# internal digits/apostrophes/hyphens) and contains at least one letter.
_EN_WORD_RE = _re.compile(r"^[a-z][a-z0-9'’\-\.]*$")


def is_english_word(tok: str) -> bool:
    return bool(_EN_WORD_RE.match(tok)) and any("a" <= c <= "z" for c in tok)


def is_katakana_token(tok: str) -> bool:
    return len(tok) > 0 and all(_is_katakana(c) for c in tok)


# ---------------------------------------------------------------------------
# Normalization (applied identically to reference and hypothesis)
# ---------------------------------------------------------------------------


def normalize(text: str, remove_fillers: bool = True) -> str:
    """NFKC (folds full/half-width & most number forms) -> lowercase -> strip
    punctuation -> collapse whitespace. Optionally drop English fillers."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Replace anything that is not a letter, digit, Japanese char, or space with
    # a space. \w would keep underscore; be explicit about letters/digits.
    out = []
    for ch in text:
        if ch.isspace():
            out.append(" ")
        elif _is_latin_letter(ch) or ch.isdigit() or _is_japanese_char(ch):
            out.append(ch)
        else:
            out.append(" ")
    text = "".join(out)
    text = _re.sub(r"\s+", " ", text).strip()
    if remove_fillers and text:
        text = " ".join(t for t in text.split(" ") if t not in ENGLISH_FILLERS)
    return text


# ---------------------------------------------------------------------------
# Tokenizers for each metric's token stream
# ---------------------------------------------------------------------------


def mixed_tokens(norm_text: str) -> list[str]:
    """Mixed stream: each Japanese char is its own token; runs of Latin letters
    (and digits attached to them) form word tokens; standalone digit runs form
    number tokens."""
    tokens: list[str] = []
    buf: list[str] = []

    def flush():
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    for ch in norm_text:
        if ch == " ":
            flush()
        elif _is_japanese_char(ch):
            flush()
            tokens.append(ch)  # one token per Japanese char
        else:  # latin letter or digit
            buf.append(ch)
    flush()
    return tokens


def japanese_chars(norm_text: str) -> list[str]:
    """Japanese-only character stream (kana + Han) for CER."""
    return [c for c in norm_text if _is_japanese_char(c)]


def english_words(norm_text: str) -> list[str]:
    """English (Latin) word stream for WER."""
    return [t for t in mixed_tokens(norm_text) if is_english_word(t)]


_TAGGER = None


def _tagger():
    global _TAGGER
    if _TAGGER is None:
        import fugashi
        _TAGGER = fugashi.Tagger()
    return _TAGGER


def japanese_words(norm_text: str) -> list[str]:
    """Japanese word stream (fugashi/MeCab segmentation) for JA-WER. Keeps tokens
    that contain at least one Japanese char (drops Latin/number tokens)."""
    if not norm_text.strip():
        return []
    return [tok.surface for tok in _tagger()(norm_text)
            if tok.surface and any(_is_japanese_char(c) for c in tok.surface)]


# ---------------------------------------------------------------------------
# Edit distance with backtrace
# ---------------------------------------------------------------------------


@dataclass
class EditCounts:
    sub: int = 0
    dele: int = 0
    ins: int = 0
    ref_len: int = 0

    def __iadd__(self, other: "EditCounts") -> "EditCounts":
        self.sub += other.sub
        self.dele += other.dele
        self.ins += other.ins
        self.ref_len += other.ref_len
        return self

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def rate(self) -> float:
        return self.errors / self.ref_len if self.ref_len else 0.0


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Levenshtein alignment. Returns ops: (kind, ref_tok, hyp_tok) where kind in
    {eq, sub, del, ins}. O(n*m); fine for utterance-length sequences."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    i, j = n, m
    ops: list[tuple[str, str | None, str | None]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            kind = "eq" if ref[i - 1] == hyp[j - 1] else "sub"
            ops.append((kind, ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def edit_counts(ref: list[str], hyp: list[str]) -> EditCounts:
    c = EditCounts(ref_len=len(ref))
    for kind, _, _ in align(ref, hyp):
        if kind == "sub":
            c.sub += 1
        elif kind == "del":
            c.dele += 1
        elif kind == "ins":
            c.ins += 1
    return c


# ---------------------------------------------------------------------------
# Script accuracy (headline metric)
# ---------------------------------------------------------------------------


@dataclass
class ScriptCounts:
    en_ref: int = 0          # English words in the reference
    latin: int = 0           # ...rendered in Latin script in the hypothesis
    katakana: int = 0        # ...wrongly forced into katakana
    other_jp: int = 0        # ...replaced by other Japanese (kanji/hiragana)
    dropped: int = 0         # ...dropped entirely (minority language lost)

    def __iadd__(self, other: "ScriptCounts") -> "ScriptCounts":
        for f in ("en_ref", "latin", "katakana", "other_jp", "dropped"):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        return self

    @property
    def accuracy(self) -> float:
        return self.latin / self.en_ref if self.en_ref else float("nan")


def script_counts(ref_mixed: list[str], hyp_mixed: list[str]) -> ScriptCounts:
    """Align mixed token streams; for every English-word token in the reference,
    inspect what the hypothesis put there."""
    sc = ScriptCounts()
    for kind, r, h in align(ref_mixed, hyp_mixed):
        if r is None or not is_english_word(r):
            continue
        sc.en_ref += 1
        if kind == "del" or h is None:
            sc.dropped += 1
        elif is_english_word(h):
            sc.latin += 1
        elif is_katakana_token(h):
            sc.katakana += 1
        else:
            sc.other_jp += 1
    return sc


# ---------------------------------------------------------------------------
# Per-model scoring
# ---------------------------------------------------------------------------


@dataclass
class ModelMetrics:
    label: str
    n: int = 0
    mer: EditCounts = field(default_factory=EditCounts)
    ja: EditCounts = field(default_factory=EditCounts)      # JA char-level (CER)
    jaw: EditCounts = field(default_factory=EditCounts)     # JA word-level (WER)
    en: EditCounts = field(default_factory=EditCounts)
    script: ScriptCounts = field(default_factory=ScriptCounts)

    @staticmethod
    def _ec(ec: EditCounts) -> dict:
        return {"sub": ec.sub, "del": ec.dele, "ins": ec.ins, "ref": ec.ref_len}

    def as_row(self) -> dict:
        return {
            "model": self.label,
            "n": self.n,
            "MER": round(self.mer.rate, 4),
            "JA_CER": round(self.ja.rate, 4),
            "JA_WER": round(self.jaw.rate, 4),
            "EN_WER": round(self.en.rate, 4),
            "ScriptAcc": (None if self.script.en_ref == 0 else round(self.script.accuracy, 4)),
            "en_ref_words": self.script.en_ref,
            "ja_ref_chars": self.ja.ref_len,
            "ja_ref_words": self.jaw.ref_len,
            # per-metric edit decomposition (sub/del/ins/ref) — for UI drill-down
            "edits": {"mer": self._ec(self.mer), "ja_cer": self._ec(self.ja),
                      "ja_wer": self._ec(self.jaw), "en_wer": self._ec(self.en)},
            "script_breakdown": {
                "latin": self.script.latin,
                "katakana": self.script.katakana,
                "other_jp": self.script.other_jp,
                "dropped": self.script.dropped,
            },
        }


def score_pair(reference: str, hypothesis: str, remove_fillers: bool = True) -> dict:
    """Score a single utterance; returns the per-metric counts for aggregation."""
    rn = normalize(reference, remove_fillers)
    hn = normalize(hypothesis, remove_fillers)
    r_mixed, h_mixed = mixed_tokens(rn), mixed_tokens(hn)
    return {
        "mer": edit_counts(r_mixed, h_mixed),
        "ja": edit_counts(japanese_chars(rn), japanese_chars(hn)),
        "jaw": edit_counts(japanese_words(rn), japanese_words(hn)),
        "en": edit_counts(english_words(rn), english_words(hn)),
        "script": script_counts(r_mixed, h_mixed),
    }


def score_records(records: Iterable[dict], label: str, remove_fillers: bool = True) -> ModelMetrics:
    mm = ModelMetrics(label=label)
    for rec in records:
        ref = rec.get("reference", "")
        hyp = rec.get("hypothesis", "")
        parts = score_pair(ref, hyp, remove_fillers)
        mm.n += 1
        mm.mer += parts["mer"]
        mm.ja += parts["ja"]
        mm.jaw += parts["jaw"]
        mm.en += parts["en"]
        mm.script += parts["script"]
    return mm


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln}: invalid JSON: {e}")
    return rows


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def render_table(models: list[ModelMetrics]) -> str:
    headers = ["Model", "N", "MER↓", "JA-CER↓", "JA-WER↓", "EN-WER↓", "ScriptAcc↑", "EN-w", "JA-ch"]
    rows = []
    for mm in models:
        r = mm.as_row()
        sa = "n/a" if r["ScriptAcc"] is None else f"{r['ScriptAcc'] * 100:5.1f}%"
        rows.append([
            r["model"], str(r["n"]),
            f"{r['MER'] * 100:5.1f}%", f"{r['JA_CER'] * 100:5.1f}%",
            f"{r['JA_WER'] * 100:5.1f}%",
            f"{r['EN_WER'] * 100:5.1f}%", sa,
            str(r["en_ref_words"]), str(r["ja_ref_chars"]),
        ])
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
              for i in range(len(headers))]
    sep = "  "
    line = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out = [line, sep.join("-" * widths[i] for i in range(len(headers)))]
    for row in rows:
        out.append(sep.join(c.ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(out)


def parse_spec(spec: str) -> tuple[str, str]:
    """'label:path' -> (label, path). Bare path -> (filename-stem, path).
    Handles Windows-style 'C:\\...' minimally by requiring the label part to be
    non-empty and not look like a drive letter when a colon split is ambiguous."""
    if ":" in spec:
        label, path = spec.split(":", 1)
        if label and path:
            return label, path
    import os
    return os.path.splitext(os.path.basename(spec))[0], spec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Code-switching ASR scoring harness")
    ap.add_argument("inputs", nargs="+", help="One or more label:path (or path) JSONL prediction files")
    ap.add_argument("--no-fillers", action="store_true",
                    help="Keep English fillers (default removes um/uh/...)")
    ap.add_argument("--json", metavar="OUT", help="Also write metrics rows as JSON to OUT")
    args = ap.parse_args(argv)

    remove_fillers = not args.no_fillers
    models: list[ModelMetrics] = []
    for spec in args.inputs:
        label, path = parse_spec(spec)
        rows = load_jsonl(path)
        models.append(score_records(rows, label, remove_fillers))

    print(render_table(models))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([m.as_row() for m in models], f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
