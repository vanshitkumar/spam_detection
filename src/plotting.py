"""Shared visual style for every figure produced by the project.

Implements a small, validated design system (light surface):

* categorical model colours use fixed slots (colour follows the entity,
  never its rank) — Logistic Regression = blue, Random Forest = orange,
  XGBoost = aqua, Hybrid Ensemble = yellow;
* class polarity uses the diverging pair blue (ham) ↔ red (spam) — the same
  convention SHAP uses, so every explanation reads consistently;
* magnitude (confusion matrices) uses a single-hue blue ramp, light→dark;
* recessive hairline grid, no top/right spines, text in ink tokens.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

MODEL_COLORS = {                     # fixed categorical slots 1–4
    "Logistic Regression": "#2a78d6",
    "Random Forest": "#eb6834",
    "XGBoost": "#1baf7a",
    "Hybrid Ensemble": "#eda100",
}

HAM_COLOR = "#2a78d6"                # diverging pair: cool pole
SPAM_COLOR = "#e34948"               # diverging pair: warm pole
NEUTRAL = "#f0efec"

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

CLASS_COLORS = {"ham": HAM_COLOR, "spam": SPAM_COLOR}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "figure.dpi": 110,
    })


def style_axes(ax, xgrid: bool = False, ygrid: bool = True) -> None:
    """Recessive grid on one axis only; hairline spines."""
    ax.grid(False)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9)


def save_fig(fig, path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
