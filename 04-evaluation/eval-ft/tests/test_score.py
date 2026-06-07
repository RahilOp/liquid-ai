#!/usr/bin/env python3
"""Hand-made unit tests for the scoring harness.

Run directly:   python tests/test_score.py
Or with pytest: pytest tests/test_score.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))

import score  # noqa: E402

# A realistic code-switching utterance:
#   "tomorrow I'll merge the pull request"  rendered the way a bilingual JP dev
#   would actually say/write it.
REF = "明日 pull request を merge します"


def test_normalize_folds_fullwidth():
    # Full-width Latin + digits should fold to half-width via NFKC.
    assert score.normalize("ＡＢＣ１２３") == "abc123"
    # Casing normalized; ALL punctuation (incl. hyphens) -> space, applied
    # identically to ref & hyp, so "pull-request" becomes two tokens.
    assert score.normalize("Pull-Request, merge!") == "pull request merge"


def test_tokenizers():
    n = score.normalize(REF)
    assert score.english_words(n) == ["pull", "request", "merge"]
    assert score.japanese_chars(n) == ["明", "日", "を", "し", "ま", "す"]
    # Mixed stream: 1 token per JP char, 1 per English word.
    assert score.mixed_tokens(n) == ["明", "日", "pull", "request", "を", "merge", "し", "ま", "す"]


def test_perfect_hypothesis_is_zero_error():
    parts = score.score_pair(REF, REF)
    assert parts["mer"].rate == 0.0
    assert parts["ja"].rate == 0.0
    assert parts["en"].rate == 0.0
    assert parts["script"].en_ref == 3
    assert parts["script"].latin == 3
    assert parts["script"].accuracy == 1.0


def test_katakana_forcing_tanks_script_accuracy():
    # The classic cloud-ASR failure: English insertions forced into katakana.
    hyp = "明日 プルリクエスト を マージ します"
    parts = score.score_pair(REF, hyp)
    sc = parts["script"]
    assert sc.en_ref == 3
    assert sc.latin == 0
    assert sc.katakana == 3          # all three English words katakana-ized
    assert sc.accuracy == 0.0
    # English words vanished from the Latin stream -> 100% EN-WER (all deletions).
    assert parts["en"].rate == 1.0


def test_dropping_minority_language_is_penalized():
    # Model drops the English entirely.
    hyp = "明日 を します"
    parts = score.score_pair(REF, hyp)
    sc = parts["script"]
    assert sc.en_ref == 3
    assert sc.dropped == 3
    assert sc.accuracy == 0.0


def test_corpus_aggregation_and_row():
    records = [
        {"id": "a", "reference": REF, "hypothesis": REF},
        {"id": "b", "reference": REF, "hypothesis": "明日 プルリクエスト を マージ します"},
    ]
    mm = score.score_records(records, label="demo")
    row = mm.as_row()
    assert row["n"] == 2
    assert row["en_ref_words"] == 6          # 3 + 3
    assert row["script_breakdown"]["latin"] == 3
    assert row["script_breakdown"]["katakana"] == 3
    # Pooled script accuracy = 3 latin / 6 english ref words = 0.5
    assert row["ScriptAcc"] == 0.5


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
