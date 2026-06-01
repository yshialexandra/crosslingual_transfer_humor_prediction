"""
Generate PGF figures for the paper from:
  - crosslingual_transfer_results CSV
  - Training curves PNG (re-plotted from embedded data approximation)

Run:  python generate_pgf_figures.py
Output: figures/fig_crosslingual_results.pgf
        figures/fig_training_curves.pgf
"""

import io
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("pgf")                     # must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── PGF / LaTeX preamble ───────────────────────────────────────────────────
matplotlib.rcParams.update({
    "pgf.texsystem":   "pdflatex",
    "text.usetex":     True,
    "font.family":     "serif",
    "font.serif":      ["Computer Modern Roman"],
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "lines.linewidth": 1.2,
    "axes.linewidth":  0.6,
    "pgf.preamble":    r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}",
})

OUT = pathlib.Path("figures")
OUT.mkdir(exist_ok=True)

# ── Colour / style palette (colour-blind friendly) ─────────────────────────
PALETTE = {
    "text_only":                  ("#4575b4", "-",  "o"),   # blue
    "text_audio_concat":          ("#d73027", "--", "s"),   # red
    "text_audio_cross_attention": ("#1a9850", ":",  "^"),   # green
}

LABELS = {
    "text_only":                  r"\texttt{text\_only}",
    "text_audio_concat":          r"\texttt{text\_audio\_concat}",
    "text_audio_cross_attention": r"\texttt{text\_audio\_cross\_attn}",
}

LANG_LABELS = {"en": "EN (in-lang)", "fr": "FR", "es": "ES", "hu": "HU"}


# ═══════════════════════════════════════════════════════════════════════════
# 1.  CROSS-LINGUAL RESULTS — grouped bar chart  (primary result figure)
# ═══════════════════════════════════════════════════════════════════════════
def fig_crosslingual_results(csv_path: str):
    df = pd.read_csv(csv_path)

    langs       = ["en", "es", "fr", "hu"]
    models      = ["text_only", "text_audio_concat", "text_audio_cross_attention"]
    metrics     = ["f1_1", "precision_1", "recall_1", "accuracy"]
    metric_labs = ["F1 (positive class)", "Precision", "Recall", "Accuracy"]

    # pivot: index=test_lang, columns=model
    pivot = (
        df.pivot_table(index="test_lang", columns="model", values=metrics)
        .reindex(langs)
    )

    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.4), sharey=False)
    fig.subplots_adjust(wspace=0.38, left=0.08, right=0.99, top=0.88, bottom=0.26)

    bar_w   = 0.24
    x       = np.arange(len(models))
    offsets = np.array([-bar_w, 0, bar_w, bar_w * 2])     # one bar per language

    BAR_COLOURS = {
        "en": "#4575b4",
        "es": "#d73027",
        "fr": "#f46d43",
        "hu": "#1a9850",
    }
    HATCH = {"en": "", "es": "//", "fr": "..", "hu": "xx"}

    for ax, metric, mlab in zip(axes, metrics, metric_labs):
        for i, lang in enumerate(langs):
            vals = [pivot.loc[lang, (metric, m)] for m in models]
            bars = ax.bar(
                x + offsets[i] - bar_w,
                vals,
                width=bar_w * 0.92,
                label=LANG_LABELS[lang],
                color=BAR_COLOURS[lang],
                hatch=HATCH[lang],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.88,
            )

        ax.set_title(mlab, pad=3)
        ax.set_xticks(x - bar_w / 2)
        ax.set_xticklabels(
            [r"\texttt{TO}", r"\texttt{TAC}", r"\texttt{TACA}"],
            fontsize=7,
        )
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="both", which="major", length=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, linestyle=":", color="gray", alpha=0.6)

    # shared legend below
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.53, -0.02),
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    path = OUT / "fig_crosslingual_results.pgf"
    fig.savefig(path)
    plt.close(fig)
    print("saved:", path)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  TRANSFER GAP TABLE FIGURE  (supplement / inline table alternative)
# ═══════════════════════════════════════════════════════════════════════════
def fig_transfer_gap(csv_path: str):
    """Heatmap of transfer gap = F1_en - F1_crosslingual per model × target lang."""
    df  = pd.read_csv(csv_path)
    f1  = df.pivot_table(index="model", columns="test_lang", values="f1_1")
    gap = f1[["fr", "es", "hu"]].subtract(f1["en"], axis=0)   # positive = drop

    models_ord = ["text_only", "text_audio_concat", "text_audio_cross_attention"]
    gap        = gap.reindex(models_ord)

    fig, ax = plt.subplots(figsize=(3.2, 1.6))
    fig.subplots_adjust(left=0.38, right=0.97, top=0.88, bottom=0.22)

    im = ax.imshow(gap.values, aspect="auto", cmap="RdYlGn_r", vmin=-0.25, vmax=0.25)

    ax.set_xticks(range(3));   ax.set_xticklabels(["FR", "ES", "HU"])
    ax.set_yticks(range(3));   ax.set_yticklabels(
        [r"\texttt{TO}", r"\texttt{TAC}", r"\texttt{TACA}"], fontsize=7
    )
    ax.set_title("Transfer gap (EN F1 $-$ cross-lingual F1)", fontsize=7.5, pad=4)

    for i in range(3):
        for j in range(3):
            val = gap.values[i, j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=7, color="black")

    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.ax.tick_params(labelsize=6.5)

    path = OUT / "fig_transfer_gap.pgf"
    fig.savefig(path)
    plt.close(fig)
    print("saved:", path)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  TRAINING CURVES  — re-drawn from the PNG values
#     (values manually read from the PNG axes; replace with real log data
#      by passing a dict/CSV of {model: {train_loss: [...], val_f1: [...]}} )
# ═══════════════════════════════════════════════════════════════════════════

# Approximate values traced from the PNG (20 epochs, steps of 1)
_EPOCHS = np.arange(1, 21)

_TRAIN_LOSS = {
    "text_only": np.array([
        0.700,0.664,0.654,0.648,0.644,0.643,0.643,0.641,0.638,0.638,
        0.636,0.635,0.634,0.635,0.633,0.633,0.632,0.631,0.630,0.628,
    ]),
    "text_audio_concat": np.array([
        0.686,0.655,0.644,0.638,0.634,0.631,0.630,0.628,0.625,0.624,
        0.623,0.622,0.622,0.621,0.620,0.619,0.618,0.615,0.613,0.609,
    ]),
    "text_audio_cross_attention": np.array([
        0.664,0.655,0.641,0.635,0.630,0.624,0.621,0.620,0.618,0.614,
        0.613,0.610,0.607,0.605,0.601,0.600,0.598,0.595,0.592,0.588,
    ]),
}

_VAL_F1 = {
    "text_only": np.array([
        0.183,0.174,0.204,0.200,0.202,0.204,0.207,0.207,0.210,0.204,
        0.207,0.213,0.186,0.210,0.210,0.207,0.210,0.212,0.209,0.210,
    ]),
    "text_audio_concat": np.array([
        0.196,0.210,0.210,0.203,0.193,0.205,0.210,0.208,0.211,0.209,
        0.210,0.207,0.207,0.217,0.210,0.214,0.216,0.221,0.188,0.208,
    ]),
    "text_audio_cross_attention": np.array([
        0.170,0.190,0.219,0.199,0.204,0.180,0.205,0.205,0.215,0.199,
        0.210,0.204,0.188,0.205,0.204,0.208,0.203,0.213,0.199,0.208,
    ]),
}


def fig_training_curves():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.3))
    fig.subplots_adjust(wspace=0.32, left=0.09, right=0.99, top=0.88, bottom=0.22)

    for model in ["text_only", "text_audio_concat", "text_audio_cross_attention"]:
        col, ls, mk = PALETTE[model]
        lab = LABELS[model]
        ax1.plot(_EPOCHS, _TRAIN_LOSS[model], color=col, ls=ls, marker=mk,
                 markersize=2.5, markevery=4, label=lab)
        ax2.plot(_EPOCHS, _VAL_F1[model],    color=col, ls=ls, marker=mk,
                 markersize=2.5, markevery=4, label=lab)

    for ax, ylabel, title in [
        (ax1, "Cross-entropy loss", "Training Loss"),
        (ax2, "F1 (positive class)", "Validation F1"),
    ]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=3)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.tick_params(axis="both", which="major", length=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(linewidth=0.35, linestyle=":", color="gray", alpha=0.6)

    handles, labels_ = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels_,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.53, -0.04),
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    path = OUT / "fig_training_curves.pgf"
    fig.savefig(path)
    plt.close(fig)
    print("saved:", path)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  F1 LINE CHART — per language, models as series  (compact comparison)
# ═══════════════════════════════════════════════════════════════════════════
def fig_f1_by_language(csv_path: str):
    df     = pd.read_csv(csv_path)
    langs  = ["en", "fr", "es", "hu"]
    models = ["text_only", "text_audio_concat", "text_audio_cross_attention"]

    pivot  = df.pivot_table(index="test_lang", columns="model", values="f1_1").reindex(langs)

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    fig.subplots_adjust(left=0.15, right=0.99, top=0.90, bottom=0.20)

    x = np.arange(len(langs))
    for model in models:
        col, ls, mk = PALETTE[model]
        ax.plot(x, pivot[model].values, color=col, ls=ls, marker=mk,
                markersize=4, label=LABELS[model])

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_LABELS[l] for l in langs])
    ax.set_ylabel("F1 (positive class)")
    ax.set_title("Cross-lingual F1 by target language", pad=3)
    ax.tick_params(axis="both", which="major", length=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.35, linestyle=":", color="gray", alpha=0.6)
    ax.legend(frameon=False, fontsize=7)

    path = OUT / "fig_f1_by_language.pgf"
    fig.savefig(path)
    plt.close(fig)
    print("saved:", path)


# ── run all ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CSV = "/mnt/project/crosslingual_transfer_results_v2_20260528_143452_crowd_only.csv"
    fig_crosslingual_results(CSV)
    fig_transfer_gap(CSV)
    fig_training_curves()
    fig_f1_by_language(CSV)
    print("\nAll PGF figures written to ./figures/")
