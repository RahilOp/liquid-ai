#!/usr/bin/env python3
"""Generate the two vector figures for the paper.

    python make_figures.py

Writes fig_real_benchmark.pdf and fig_gap.pdf next to this script.

Design notes (see paper/OUTLINE.md):
  * ScriptAcc and MER differ in scale AND direction, so they never share an axis.
  * CS-WER (synthetic set) and MER (real set) are different metrics on different
    test sets. Figure 2 gives each its own panel, axis, and explicit n, so the
    two numbers are never read as points on one scale.
  * Every bar is direct-labelled; the palette carries no information alone.
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent

BLUE = "#2a78d6"   # categorical slot 1 - prior / baseline systems
ORANGE = "#eb6834"  # categorical slot 2 - this work
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def _style(ax, xmax, xlabel):
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_visible(False)


def _bars(ax, labels, values, colors, xmax, value_labels=None):
    y = range(len(labels))
    ax.barh(y, values, height=0.62, color=colors, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.invert_yaxis()
    texts = value_labels or [f"{v:.1f}" for v in values]
    for yi, v, t in zip(y, values, texts):
        ax.text(v + xmax * 0.015, yi, t, va="center", ha="left",
                fontsize=8, color=INK)


# --------------------------------------------------------------------------
# Figure 1 - the real-speech scoreboard (CS-FLEURS JA-EN read/test, n=196)
# --------------------------------------------------------------------------
systems = [
    "LFM2.5-Audio-1.5B-JP\n(zero-shot)",
    "Whisper large-v3\n(zero-shot)",
    "Whisper large-v3 + LoRA\n(this work, synthetic-trained)",
    "LFM2.5-Audio + LoRA\n(this work, best)",
]
script_acc = [37.4, 64.6, 73.0, 80.8]
mer = [67.3, 40.9, 36.3, 27.0]
colors = [BLUE, BLUE, ORANGE, ORANGE]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
_bars(axes[0], systems, script_acc, colors, 100)
_style(axes[0], 100, "Script Accuracy (%) \u2191")
axes[0].set_title("(a) English words kept in Latin script", fontsize=9, color=INK, loc="left", pad=8)

_bars(axes[1], systems, mer, colors, 80)
_style(axes[1], 80, "MER (%) \u2193")
axes[1].set_title("(b) Mixed error rate", fontsize=9, color=INK, loc="left", pad=8)

fig.tight_layout()
fig.savefig(HERE / "fig_real_benchmark.pdf", bbox_inches="tight")
print("wrote", HERE / "fig_real_benchmark.pdf")

# --------------------------------------------------------------------------
# Figure 2 - the synthetic-to-real gap for ONE checkpoint
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.95))

pair = ["Whisper large-v3\n(zero-shot)", "+ LoRA\n(same checkpoint)"]

_bars(axes[0], pair, [111.57, 2.48], [BLUE, ORANGE], 130, ["111.57", "2.48"])
_style(axes[0], 130, "CS-WER (%) \u2193")
axes[0].set_title("(a) Synthetic TTS test set (n = 105)", fontsize=9, color=INK, loc="left", pad=8)
axes[0].annotate("\u221297.8% relative", xy=(0.55, 0.14), xycoords="axes fraction",
                 fontsize=8, color=ORANGE, ha="left")

_bars(axes[1], pair, [40.9, 36.3], [BLUE, ORANGE], 130)
_style(axes[1], 130, "MER (%) \u2193")
axes[1].set_title("(b) Real read speech, CS-FLEURS JA-EN (n = 196)",
                  fontsize=9, color=INK, loc="left", pad=8)
axes[1].annotate("\u221211.2% relative", xy=(0.55, 0.14), xycoords="axes fraction",
                 fontsize=8, color=ORANGE, ha="left")

fig.tight_layout()
fig.savefig(HERE / "fig_gap.pdf", bbox_inches="tight")
print("wrote", HERE / "fig_gap.pdf")
