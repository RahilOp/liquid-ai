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

Known limitation: a phonetic romaji rendering of an English word (e.g.
``meeting`` -> ``miitingu``) is counted as Latin script because it is made of
Latin letters. Telling valid English insertions from romaji requires lexical or
semantic matching that the current token-only alignment does not perform. The
``--audit`` output flags Latin tokens that do not exactly match the reference so
these cases can be inspected manually.

Usage:
    python eval/score.py whisper:preds_whisper.jsonl lfm:preds_lfm.jsonl
    python eval/score.py preds.jsonl                       # label from filename
    python eval/score.py --no-fillers whisper:preds.jsonl  # keep English fillers
    python eval/score.py whisper:preds.jsonl --json out.json
    python eval/score.py model:preds.jsonl --bootstrap 1000 --ci 95
    python eval/score.py model:preds.jsonl --switch-report switch.json --audit audit.json

Each input line: {"id": "...", "reference": "...", "hypothesis": "..."}
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

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


def _token_script_category(r: str, h: str | None) -> str:
    """Category for one aligned English reference token."""
    if h is None:
        return "dropped"
    if is_english_word(h):
        return "latin"
    if is_katakana_token(h):
        return "katakana"
    return "other_jp"


def script_counts_detailed(ref_mixed: list[str], hyp_mixed: list[str]) -> list[dict]:
    """Per-English-word script decision for switch / audit analysis."""
    out = []
    for kind, r, h in align(ref_mixed, hyp_mixed):
        if r is None or not is_english_word(r):
            continue
        out.append({"ref": r, "hyp": h, "kind": kind,
                    "category": _token_script_category(r, h)})
    return out


# ---------------------------------------------------------------------------
# Switch-point analysis
# ---------------------------------------------------------------------------

def _token_lang(tok: str) -> str:
    if _is_japanese_char(tok[0]):
        return "ja"
    if is_english_word(tok):
        return "en"
    return "other"


def switch_points(tokens: list[str]) -> list[int]:
    """Return boundary positions where language switches between EN and JP.

    A boundary between token i-1 and token i is represented by index i.
    """
    pts = []
    for i in range(1, len(tokens)):
        prev, cur = _token_lang(tokens[i - 1]), _token_lang(tokens[i])
        if (prev == "en" and cur == "ja") or (prev == "ja" and cur == "en"):
            pts.append(i)
    return pts


def _nearest_switch_distance(idx: int, points: list[int]) -> int | None:
    if not points:
        return None
    return min(abs(idx - p) for p in points)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split `text` into sentence-like spans using EN/JA sentence finals."""
    spans = []
    start = 0
    for m in _re.finditer(r"[。！？.!?]+", text):
        end = m.end()
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _english_spans_in_original(original: str) -> list[tuple[str, int, int]]:
    """Find English-word-like spans in the original (cased) reference text.

    Returns tuples of (lowercased_word, start_char, end_char).
    """
    spans = []
    # Keep apostrophes and hyphens inside words; stop on punctuation/space/JP.
    for m in _re.finditer(r"[A-Za-z][A-Za-z0-9'’\-\.]*", original):
        spans.append((m.group(0).lower(), m.start(), m.end()))
    return spans


def _match_english_words_to_original(norm_words: list[str], original: str) -> list[tuple[int, int] | None]:
    """Align normalized English word tokens to original char spans in order.

    Returns a list the same length as `norm_words`; each element is either
    (start, end) char offsets in `original`, or None if no span matched.
    """
    orig_spans = _english_spans_in_original(original)
    mapping: list[tuple[int, int] | None] = []
    oi = 0
    for w in norm_words:
        # A normalized word may differ from the original (digits, full-width,
        # punctuation removed). Fall back to the next English span whose
        # lowercased form starts with the same letters, then advance.
        found = None
        while oi < len(orig_spans):
            ow, s, e = orig_spans[oi]
            if ow.startswith(w) or w.startswith(ow.replace("-", "").replace("'", "")):
                found = (s, e)
                oi += 1
                break
            oi += 1
        mapping.append(found)
    return mapping


def _is_at_sentence_boundary(span: tuple[int, int] | None, original: str, sentences: list[tuple[int, int]]) -> bool:
    """True if the English span starts or ends a sentence."""
    if span is None or not sentences:
        return False
    s, e = span
    # Sentence start = previous non-space char is a sentence-final punctuation or start of text.
    prev_ch = original[s - 1] if s > 0 else ""
    at_start = s == 0 or prev_ch in "。！？.!?"
    next_ch = original[e] if e < len(original) else ""
    at_end = e == len(original) or next_ch in "。！？.!?"
    return at_start or at_end


def _build_distance_buckets(max_dist: int = 3) -> dict:
    keys = [str(d) for d in range(max_dist)] + [f">={max_dist}"]
    return {k: {"tokens": 0, "correct": 0} for k in keys}


def _bucket(d: int | None, max_dist: int) -> str:
    if d is None:
        return ">=0"  # no switches in utterance
    if d < max_dist:
        return str(d)
    return f">={max_dist}"


def analyze_switch_points(reference: str, hypothesis: str, original: str,
                          remove_fillers: bool = True) -> dict:
    """Stratified analysis of errors around EN/JP switch points.

    Requires the *original* reference text (before normalization) so that
    sentence boundaries can be detected for inter/intra-sentential typing.
    """
    rn = normalize(reference, remove_fillers)
    hn = normalize(hypothesis, remove_fillers)
    ref_toks = mixed_tokens(rn)
    hyp_toks = mixed_tokens(hn)
    ops = align(ref_toks, hyp_toks)

    points = switch_points(ref_toks)
    max_dist = 3
    all_by_dist = _build_distance_buckets(max_dist)
    en_by_dist = _build_distance_buckets(max_dist)
    ja_by_dist = _build_distance_buckets(max_dist)

    # Token-level correctness from alignment.
    for i, (kind, r, h) in enumerate(ops):
        if r is None:
            continue
        tok = r
        lang = _token_lang(tok)
        dist = _nearest_switch_distance(i, points)
        b = _bucket(dist, max_dist)
        correct = kind == "eq"
        all_by_dist[b]["tokens"] += 1
        if correct:
            all_by_dist[b]["correct"] += 1
        if lang == "en":
            en_by_dist[b]["tokens"] += 1
            if correct:
                en_by_dist[b]["correct"] += 1
        elif lang == "ja":
            ja_by_dist[b]["tokens"] += 1
            if correct:
                ja_by_dist[b]["correct"] += 1

    # Switch-type and span-length for English-script accuracy.
    sentences = _sentence_spans(original)
    en_word_indices = [i for i, t in enumerate(ref_toks) if is_english_word(t)]
    orig_spans = _match_english_words_to_original([ref_toks[i] for i in en_word_indices], original)

    script_detail = script_counts_detailed(ref_toks, hyp_toks)
    type_counts = {"intra": {"en_ref": 0, "latin": 0},
                   "inter": {"en_ref": 0, "latin": 0},
                   "unknown": {"en_ref": 0, "latin": 0}}
    span_counts = {"single": {"en_ref": 0, "latin": 0},
                   "phrase": {"en_ref": 0, "latin": 0}}

    # Compute run lengths of consecutive English tokens.
    run_lengths = {}
    i = 0
    while i < len(ref_toks):
        if is_english_word(ref_toks[i]):
            j = i
            while j < len(ref_toks) and is_english_word(ref_toks[j]):
                j += 1
            for k in range(i, j):
                run_lengths[k] = j - i
            i = j
        else:
            i += 1

    for detail, idx, span in zip(script_detail, en_word_indices, orig_spans):
        inter = _is_at_sentence_boundary(span, original, sentences)
        stype = "inter" if inter else "intra"
        type_counts[stype]["en_ref"] += 1
        if detail["category"] == "latin":
            type_counts[stype]["latin"] += 1

        run = run_lengths.get(idx, 1)
        slen = "single" if run == 1 else "phrase"
        span_counts[slen]["en_ref"] += 1
        if detail["category"] == "latin":
            span_counts[slen]["latin"] += 1

    def _acc(d: dict) -> dict:
        return {k: round(v["correct"] / v["tokens"], 4) if v["tokens"] else None
                for k, v in d.items()}

    def _script_acc(d: dict) -> dict:
        return {k: round(v["latin"] / v["en_ref"], 4) if v["en_ref"] else None
                for k, v in d.items()}

    # Downstream effect of a katakana substitution.
    after_kat = {"tokens": 0, "errors": 0}
    after_lat = {"tokens": 0, "errors": 0}
    for i, det in enumerate(script_detail):
        idx = en_word_indices[i]
        if det["category"] != "katakana" and det["category"] != "latin":
            continue
        # Look at the next 5 tokens in the *reference* mixed stream.
        for offset in range(1, 6):
            nxt = idx + offset
            if nxt >= len(ref_toks):
                break
            # Find if this downstream token matched.
            op = next((op for j, op in enumerate(ops) if j == nxt), None)
            if op is None:
                continue
            _, rt, _ = op
            if rt is None:
                continue
            kind = op[0]
            entry = after_kat if det["category"] == "katakana" else after_lat
            entry["tokens"] += 1
            if kind != "eq":
                entry["errors"] += 1

    return {
        "distance": all_by_dist,
        "en_distance": en_by_dist,
        "ja_distance": ja_by_dist,
        "switch_type": type_counts,
        "span_length": span_counts,
        "downstream": {"after_katakana": after_kat, "after_correct_latin": after_lat},
    }


def _acc_from_counts(counts: dict) -> dict:
    return {k: round(v["correct"] / v["tokens"], 4) if v["tokens"] else None
            for k, v in counts.items()}


def _scriptacc_from_counts(counts: dict) -> dict:
    return {k: round(v["latin"] / v["en_ref"], 4) if v["en_ref"] else None
            for k, v in counts.items()}


def _rate_from_counts(entry: dict) -> float | None:
    return round(entry["errors"] / entry["tokens"], 4) if entry["tokens"] else None


def switch_report_from_counts(total: dict) -> dict:
    """Turn aggregated raw counts into the published switch-point report."""
    downstream = total["downstream"]
    return {
        "token_accuracy_by_distance": _acc_from_counts(total["distance"]),
        "en_token_accuracy_by_distance": _acc_from_counts(total["en_distance"]),
        "ja_token_accuracy_by_distance": _acc_from_counts(total["ja_distance"]),
        "script_accuracy_by_switch_type": _scriptacc_from_counts(total["switch_type"]),
        "script_accuracy_by_span_length": _scriptacc_from_counts(total["span_length"]),
        "downstream_error_after_katakana": _rate_from_counts(downstream["after_katakana"]),
        "downstream_error_after_correct_latin": _rate_from_counts(downstream["after_correct_latin"]),
    }


def aggregate_switch_analysis(records: Iterable[dict], remove_fillers: bool = True) -> dict:
    """Aggregate switch-point counts over many utterances and return the report."""
    max_dist = 3
    total_dist = _build_distance_buckets(max_dist)
    total_en_dist = _build_distance_buckets(max_dist)
    total_ja_dist = _build_distance_buckets(max_dist)
    total_type = {"intra": {"en_ref": 0, "latin": 0},
                  "inter": {"en_ref": 0, "latin": 0},
                  "unknown": {"en_ref": 0, "latin": 0}}
    total_span = {"single": {"en_ref": 0, "latin": 0},
                  "phrase": {"en_ref": 0, "latin": 0}}
    total_down = {"after_katakana": {"tokens": 0, "errors": 0},
                  "after_correct_latin": {"tokens": 0, "errors": 0}}

    for rec in records:
        counts = analyze_switch_points(rec["reference"], rec["hypothesis"],
                                       rec["reference"], remove_fillers)
        for k, v in counts["distance"].items():
            total_dist[k]["tokens"] += v["tokens"]
            total_dist[k]["correct"] += v["correct"]
        for k, v in counts["en_distance"].items():
            total_en_dist[k]["tokens"] += v["tokens"]
            total_en_dist[k]["correct"] += v["correct"]
        for k, v in counts["ja_distance"].items():
            total_ja_dist[k]["tokens"] += v["tokens"]
            total_ja_dist[k]["correct"] += v["correct"]
        for k, v in counts["switch_type"].items():
            total_type[k]["en_ref"] += v["en_ref"]
            total_type[k]["latin"] += v["latin"]
        for k, v in counts["span_length"].items():
            total_span[k]["en_ref"] += v["en_ref"]
            total_span[k]["latin"] += v["latin"]
        for k, v in counts["downstream"].items():
            total_down[k]["tokens"] += v["tokens"]
            total_down[k]["errors"] += v["errors"]

    return switch_report_from_counts({
        "distance": total_dist,
        "en_distance": total_en_dist,
        "ja_distance": total_ja_dist,
        "switch_type": total_type,
        "span_length": total_span,
        "downstream": total_down,
    })


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


def _int_counts(parts: dict) -> dict:
    """Flatten one utterance's EditCounts/ScriptCounts to plain ints."""
    def _ec(ec: EditCounts) -> dict:
        return {"sub": ec.sub, "del": ec.dele, "ins": ec.ins, "ref": ec.ref_len,
                "err": ec.errors}
    return {
        "mer": _ec(parts["mer"]), "ja_cer": _ec(parts["ja"]),
        "ja_wer": _ec(parts["jaw"]), "en_wer": _ec(parts["en"]),
        "script": {
            "en_ref": parts["script"].en_ref, "latin": parts["script"].latin,
            "katakana": parts["script"].katakana, "other_jp": parts["script"].other_jp,
            "dropped": parts["script"].dropped,
        },
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


def per_utterance_counts(records: Iterable[dict], remove_fillers: bool = True) -> list[dict]:
    """Per-utterance integer counts; needed for bootstrap resampling."""
    out = []
    for rec in records:
        out.append(_int_counts(score_pair(rec.get("reference", ""),
                                          rec.get("hypothesis", ""), remove_fillers)))
    return out


def _aggregate_rates(counts: Sequence[dict]) -> dict:
    """Pool a list of per-utterance int-count dicts into metric rates."""
    def _rate(err: int, ref: int) -> float:
        return err / ref if ref else float("nan")

    mer_err = mer_ref = 0
    ja_err = ja_ref = 0
    jaw_err = jaw_ref = 0
    en_err = en_ref = 0
    latin = en_ref_words = 0
    for c in counts:
        mer_err += c["mer"]["err"]; mer_ref += c["mer"]["ref"]
        ja_err += c["ja_cer"]["err"]; ja_ref += c["ja_cer"]["ref"]
        jaw_err += c["ja_wer"]["err"]; jaw_ref += c["ja_wer"]["ref"]
        en_err += c["en_wer"]["err"]; en_ref += c["en_wer"]["ref"]
        latin += c["script"]["latin"]; en_ref_words += c["script"]["en_ref"]
    return {
        "MER": _rate(mer_err, mer_ref),
        "JA_CER": _rate(ja_err, ja_ref),
        "JA_WER": _rate(jaw_err, jaw_ref),
        "EN_WER": _rate(en_err, en_ref),
        "ScriptAcc": _rate(latin, en_ref_words),
    }


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    """Nearest-rank percentile on an already-sorted sequence (0..100)."""
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def bootstrap_ci(per_utts: Sequence[dict], n_samples: int, ci: float,
                 rng: random.Random) -> dict:
    """Bootstrap percentile CIs for every rate over `per_utts`.

    Returns a dict of metric -> {"mean": ..., "lo": ..., "hi": ...} at the
    requested confidence level.
    """
    n = len(per_utts)
    dists = {k: [] for k in ("MER", "JA_CER", "JA_WER", "EN_WER", "ScriptAcc")}
    for _ in range(n_samples):
        sample = [per_utts[rng.randrange(n)] for _ in range(n)]
        rates = _aggregate_rates(sample)
        for k, v in rates.items():
            dists[k].append(v)
    lo_p = (100.0 - ci) / 2.0
    hi_p = 100.0 - lo_p
    out = {}
    for k, vals in dists.items():
        s = sorted(vals)
        m = sum(s) / len(s)
        out[k] = {"mean": round(m, 4), "lo": round(_percentile(s, lo_p), 4),
                  "hi": round(_percentile(s, hi_p), 4)}
    return out


def paired_bootstrap(per_utts_a: Sequence[dict], per_utts_b: Sequence[dict],
                     n_samples: int, ci: float, rng: random.Random) -> dict:
    """Paired bootstrap of (model_a - model_b) over matched utterances."""
    if len(per_utts_a) != len(per_utts_b):
        raise ValueError("paired bootstrap requires the same number of utterances")
    n = len(per_utts_a)
    dists = {k: [] for k in ("MER", "JA_CER", "JA_WER", "EN_WER", "ScriptAcc")}
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        sample_a = [per_utts_a[i] for i in idx]
        sample_b = [per_utts_b[i] for i in idx]
        rates_a = _aggregate_rates(sample_a)
        rates_b = _aggregate_rates(sample_b)
        for k in rates_a:
            dists[k].append(rates_a[k] - rates_b[k])
    lo_p = (100.0 - ci) / 2.0
    hi_p = 100.0 - lo_p
    out = {}
    for k, vals in dists.items():
        s = sorted(vals)
        m = sum(s) / len(s)
        out[k] = {"mean": round(m, 4), "lo": round(_percentile(s, lo_p), 4),
                  "hi": round(_percentile(s, hi_p), 4)}
    return out


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


def _format_ci(ci: dict) -> str:
    return f"{ci['mean']*100:5.1f}% [{ci['lo']*100:5.1f}%, {ci['hi']*100:5.1f}%]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Code-switching ASR scoring harness")
    ap.add_argument("inputs", nargs="+", help="One or more label:path (or path) JSONL prediction files")
    ap.add_argument("--no-fillers", action="store_true",
                    help="Keep English fillers (default removes um/uh/...)")
    ap.add_argument("--json", metavar="OUT", help="Also write metrics rows as JSON to OUT")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N",
                    help="Bootstrap N resamples over utterances and report CIs")
    ap.add_argument("--ci", type=float, default=95, metavar="P",
                    help="Confidence level for bootstrap intervals (default 95)")
    ap.add_argument("--bootstrap-seed", type=int, default=42, metavar="S",
                    help="Random seed for reproducible bootstrap resampling")
    ap.add_argument("--paired", action="store_true",
                    help="Paired bootstrap between the first two models (same resamples)")
    ap.add_argument("--switch-report", metavar="OUT",
                    help="Write switch-point stratified analysis JSON for the first model")
    ap.add_argument("--audit", metavar="OUT",
                    help="Write per-English-word script audit JSON for the first model")
    args = ap.parse_args(argv)

    if args.paired and len(args.inputs) != 2:
        raise SystemExit("--paired requires exactly two input files")
    if args.bootstrap < 0:
        raise SystemExit("--bootstrap must be non-negative")

    remove_fillers = not args.no_fillers
    models: list[ModelMetrics] = []
    per_model_counts: list[list[dict]] = []
    all_rows: list[list[dict]] = []
    for spec in args.inputs:
        label, path = parse_spec(spec)
        rows = load_jsonl(path)
        all_rows.append(rows)
        models.append(score_records(rows, label, remove_fillers))
        per_model_counts.append(per_utterance_counts(rows, remove_fillers))

    print(render_table(models))

    bootstrap_results: dict = {}
    if args.bootstrap:
        rng = random.Random(args.bootstrap_seed)
        print(f"\nBootstrap CIs ({args.bootstrap} resamples, {args.ci}% level, seed={args.bootstrap_seed}):")
        for mm, counts in zip(models, per_model_counts):
            cis = bootstrap_ci(counts, args.bootstrap, args.ci, rng)
            bootstrap_results[mm.label] = cis
            print(f"  {mm.label}")
            for metric, ci in cis.items():
                print(f"    {metric:9s} {_format_ci(ci)}")

        if args.paired:
            paired = paired_bootstrap(per_model_counts[0], per_model_counts[1],
                                      args.bootstrap, args.ci, rng)
            bootstrap_results["paired_diff"] = paired
            print(f"\nPaired bootstrap ({models[0].label} - {models[1].label}):")
            for metric, ci in paired.items():
                sign = "includes 0" if ci["lo"] <= 0 <= ci["hi"] else "excludes 0"
                print(f"    {metric:9s} Δ {_format_ci(ci)} ({sign})")

    if args.switch_report or args.audit:
        if len(args.inputs) != 1:
            raise SystemExit("--switch-report and --audit require exactly one input file")
        rows = all_rows[0]
        label = models[0].label

        if args.switch_report:
            payload = {"model": label,
                       "analysis": aggregate_switch_analysis(rows, remove_fillers)}
            with open(args.switch_report, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"\nWrote switch-point report -> {args.switch_report}")

        if args.audit:
            audit_records = []
            for rec in rows:
                rn = normalize(rec["reference"], remove_fillers)
                hn = normalize(rec["hypothesis"], remove_fillers)
                for det in script_counts_detailed(mixed_tokens(rn), mixed_tokens(hn)):
                    h = det["hyp"]
                    latin_mismatch = (det["category"] == "latin" and
                                      h is not None and h != det["ref"])
                    audit_records.append({
                        "id": rec["id"], "ref_word": det["ref"],
                        "hyp_word": h, "category": det["category"],
                        "latin_mismatch": latin_mismatch,
                    })
            counts = {"en_ref": 0, "latin": 0, "katakana": 0, "other_jp": 0, "dropped": 0,
                      "latin_mismatch": 0}
            for r in audit_records:
                counts["en_ref"] += 1
                counts[r["category"]] += 1
                if r["latin_mismatch"]:
                    counts["latin_mismatch"] += 1
            payload = {"model": label, "counts": counts, "records": audit_records}
            with open(args.audit, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"\nWrote script audit -> {args.audit}")

    if args.json:
        payload = [m.as_row() for m in models]
        if bootstrap_results:
            payload = {"models": payload, "bootstrap": bootstrap_results}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
