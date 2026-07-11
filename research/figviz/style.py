"""Shared matplotlib style and palette for report figures.

Colours follow the repository data-viz palette: a fixed categorical order for the
four tasks, a single-hue blue ramp for sequential heatmaps, and a red/blue
diverging pair for signed quantities. Figures are rendered on a light surface for
embedding in the markdown reports.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots (light mode) from references/palette.md, chosen for CVD safety.
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
MAGENTA = "#e87ba4"
ORANGE = "#eb6834"

# One stable colour per task, reused across every report.
TASK_COLORS = {
    "copy": BLUE,
    "reverse": ORANGE,
    "sort": VIOLET,
    "rotate_left": GREEN,
}
TASK_LABELS = {
    "copy": "copy",
    "reverse": "reverse",
    "sort": "sort",
    "rotate_left": "rotate-left",
}
TASK_ORDER = ["copy", "reverse", "sort", "rotate_left"]

# Ink tokens.
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#ffffff"

# Sequential blue ramp (100 -> 700) for magnitude heatmaps.
_BLUE_RAMP = [
    "#f2f7fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#184f95", "#0d366b",
]
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", _BLUE_RAMP)

# Diverging red/neutral/blue for signed effects.
DIVERGING = LinearSegmentedColormap.from_list(
    "div_rb", ["#184f95", "#6da7ec", "#f2f1ec", "#f0a6a5", "#c02c2b"]
)


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": INK_2,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })
