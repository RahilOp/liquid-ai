# Paper: Synthetic Speech Overstates Code-Switching ASR

LaTeX source for the Japanese–English code-switching ASR paper, built from the
dissertation in `../code-switching/` and the real-speech benchmark in `../`.

## Build

```bash
./build.sh          # regenerates figures, then compiles main.pdf
```

The build uses [Tectonic](https://tectonic-typesetting.github.io/) (self-contained,
fetches LaTeX packages on demand). It is vendored in `.tools/` and git-ignored;
re-fetch with:

```bash
mkdir -p .tools && cd .tools
curl -sL https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz | tar xz
```

Requires a CJK font — the document asks for `Noto Serif CJK JP` / `Noto Sans CJK JP`
(`fonts-noto-cjk` on Debian/Ubuntu). Mixed Japanese–English strings appear in the
prose, so this is not optional.

Overleaf works too: upload `main.tex`, `refs.bib`, and `figures/`, and set the
compiler to **XeLaTeX** (required by `xeCJK`).

## Layout

| Path | What |
|---|---|
| `main.tex` | the manuscript |
| `refs.bib` | bibliography |
| `OUTLINE.md` | paper story, section outline, **claim–evidence map**, open items |
| `figures/make_figures.py` | generates the two vector figures from the numbers in the text |
| `figures/*.pdf` | generated — do not edit by hand |
| `figures/*.png` | inherited from the dissertation drafts |
| `sources/extract_docx.py` | dumps a `.docx` to text |
| `sources/*.txt` | extracted dissertation + spec report, used as source material |

## Before submitting

`OUTLINE.md` tracks these; the two marked in the PDF with a red `[TODO: …]` block
must be closed:

1. **Recover the training recipe for the best LFM run** (rank-32 encoder LoRA +
   FLEURS mix). It is the paper's strongest number and `docs/asr-ft-training-log.md`
   stops at the preceding run — currently a measurement with no method.
2. **Score the merged Whisper checkpoint on the FLEURS monolingual controls** to
   fill the three `–` cells in Table 5.
3. **Bootstrap confidence intervals** over the 196 code-switching utterances.
4. Separate LoRA rank from data mixture in the final ablation step (61.8 → 80.8
   changes both).

## Citation corrections carried over from the dissertation

- **Hsu et al. (2023)** (arXiv 2310.12477) was cited in the dissertation as the
  authority for "q/v projections, r=8 optimal for Whisper LoRA". That paper is
  *In-Context Learning of Textless Speech LM for Speech Classification* — not LoRA,
  not Whisper, not ASR. Removed; the hyperparameter choices are re-grounded on this
  work's own rank ablation and on Hu et al. (2021).
- **"Indra, W.G. et al. (2020)"** → **Winata, G.I. et al. (2020)**. The given name
  had been parsed as the surname.
