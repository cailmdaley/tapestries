"""PTE heatmap: systematic null test pass/fail per sysmap, tracer, and bin.

Single 11x18 matshow — rows = systematic maps, columns = 3 tracers x 6 bins.
Colored by PTE from NaMaster chi2 null test (C^{fS} = 0).

Colormap: vlag (diverging, 0-1) with TwoSlopeNorm. PTE < 0.05 and > 0.95
saturate to extreme colors. Cells annotated with PTE values.

CAVEAT: Error bars are not yet validated — PTE pass/fail may reflect error bar
mis-estimation rather than true contamination. This is a preliminary diagnostic.
"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.plot_utils import (
    FW, save_evidence, setup_theme, LABELS, SYSMAP_LABELS, TOM_BINS,
)

import json

snakemake = snakemake  # type: ignore # noqa: F821

output_png = Path(snakemake.output.png)
evidence_path = Path(snakemake.output.evidence)
sys_dir = Path(snakemake.input.sys_dir)
methods = list(snakemake.params.methods)
sysmap_keys = list(snakemake.params.sys_map_keys)
tom_bins = list(snakemake.params.tom_bins)

snakemake_log(snakemake, f"PTE heatmap: {len(methods)} methods, {len(sysmap_keys)} sysmaps")

# -- Ordered sysmap display -----------------------------------------------
sysmap_order = [k for k in [
    "stellar_density", "galactic_extinction", "exposures", "zodiacal_light",
    "saa", "psf", "noise", "persistence", "star_brightness_mean",
    "galaxy_number_density", "phz_quality",
] if k in sysmap_keys]

n_maps = len(sysmap_order)
n_bins = len(tom_bins)
n_methods = len(methods)
n_cols = n_methods * n_bins  # 18


def load_pte(sysmap, method, bin_id):
    """Load PTE from evidence JSON. Returns PTE or NaN."""
    path = sys_dir / f"{sysmap}_x_{method}_bin{bin_id}_evidence.json"
    if not path.exists():
        return np.nan
    with open(path) as f:
        ev = json.load(f)
    evidence = ev.get("evidence", {})
    pte = evidence.get("pte")
    if pte is not None:
        return float(pte)
    chi2_val = evidence.get("chi2")
    dof = evidence.get("dof")
    if chi2_val is not None and dof is not None:
        from scipy.stats import chi2 as chi2_dist
        return float(chi2_dist.sf(chi2_val, dof))
    return np.nan


# -- Build 11 x 18 PTE matrix --------------------------------------------
# Column order: [density_bin1..6, lensmc_bin1..6, metacal_bin1..6]
mat = np.full((n_maps, n_cols), np.nan)
col_labels = []
col_method_group = []  # track which method each column belongs to

for m_idx, method in enumerate(methods):
    for b_idx, bin_id in enumerate(tom_bins):
        col = m_idx * n_bins + b_idx
        col_labels.append(f"{bin_id}")
        col_method_group.append(method)
        for row, sysmap in enumerate(sysmap_order):
            mat[row, col] = load_pte(sysmap, method, bin_id)

# -- Colormap: vlag with saturated tails -----------------------------------
# PTE < 0.05 → extreme red (flat), PTE > 0.95 → extreme blue (flat).
# Interior [0.05, 0.95] maps linearly to the full colormap range.
vlag_cmap = sns.color_palette("vlag", as_cmap=True).copy()
vlag_cmap.set_bad(color="lightgray")


class SaturatedTailNorm(mcolors.Normalize):
    """Piecewise linear norm: tails [0,lo] and [hi,1] clamp to extremes."""

    def __init__(self, lo=0.05, hi=0.95):
        super().__init__(vmin=0, vmax=1)
        self.lo = lo
        self.hi = hi

    def __call__(self, value, clip=None):
        x = np.asarray(value, dtype=float)
        # Give tails a thin but non-degenerate extent so colorbar ticks
        # at 0, 0.05, 0.95, 1.0 don't overlap.
        result = np.interp(x, [0, self.lo, self.hi, 1], [0.0, 0.04, 0.96, 1.0])
        return np.ma.array(result, mask=np.ma.getmask(value))

    def inverse(self, value):
        return np.interp(np.asarray(value), [0, 1], [self.lo, self.hi])


norm = SaturatedTailNorm(lo=0.05, hi=0.95)

# -- Plot -----------------------------------------------------------------
setup_theme(style="white")

row_labels = [SYSMAP_LABELS.get(k, k) for k in sysmap_order]
method_labels = [LABELS.get(m, m) for m in methods]

# Dimensions: wide enough for 18 columns to be readable
cell_w = 0.48
cell_h = 0.46
margin_left = 1.5
margin_right = 0.7
margin_top = 1.0
margin_bottom = 0.15

fig_width = margin_left + n_cols * cell_w + margin_right
fig_height = margin_top + n_maps * cell_h + margin_bottom

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
fig.subplots_adjust(
    left=margin_left / fig_width,
    right=1 - margin_right / fig_width,
    top=1 - margin_top / fig_height,
    bottom=margin_bottom / fig_height,
)

im = ax.imshow(mat, aspect="auto", cmap=vlag_cmap, norm=norm,
               interpolation="nearest")

# Annotate cells
for i in range(n_maps):
    for j in range(n_cols):
        val = mat[i, j]
        if np.isfinite(val):
            text_color = "white" if val < 0.05 or val > 0.95 else "black"
            weight = "bold" if val < 0.05 else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5.5, color=text_color, fontweight=weight)

# Row labels (sysmaps)
ax.set_yticks(np.arange(n_maps))
ax.set_yticklabels(row_labels, fontsize=8)

# Column labels (bin numbers) at top
ax.set_xticks(np.arange(n_cols))
ax.set_xticklabels(col_labels, fontsize=7)
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")

# Method group labels above bin numbers
for m_idx, method in enumerate(methods):
    center = m_idx * n_bins + (n_bins - 1) / 2
    ax.text(center, 1.08, method_labels[m_idx], ha="center", va="bottom",
            fontsize=8, fontweight="bold", transform=ax.get_xaxis_transform())

# Vertical separators between method groups
for m_idx in range(1, n_methods):
    x = m_idx * n_bins - 0.5
    ax.axvline(x, color="black", lw=1.5)

# Grid lines between cells
ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(n_maps + 1) - 0.5, minor=True)
ax.grid(which="minor", color="white", linewidth=0.5)
ax.tick_params(which="minor", bottom=False, left=False)

# Colorbar with clean threshold ticks (no overlap at saturated tails)
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03, label="PTE")
cbar.set_ticks([0.05, 0.25, 0.5, 0.75, 0.95])
cbar.set_ticklabels([r"$\leq\!0.05$", "0.25", "0.50", "0.75", r"$\geq\!0.95$"])
cbar.ax.axhline(0.05, color="black", lw=0.8, ls="--")
cbar.ax.axhline(0.95, color="black", lw=0.8, ls="--")

output_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_png, bbox_inches="tight", dpi=150)
plt.close(fig)
snakemake_log(snakemake, f"Saved: {output_png}")

# -- Evidence --------------------------------------------------------------
evidence_summary = {}
for m_idx, method in enumerate(methods):
    cols = slice(m_idx * n_bins, (m_idx + 1) * n_bins)
    sub = mat[:, cols]
    finite = np.isfinite(sub)
    n_total = int(np.sum(finite))
    n_fail_low = int(np.sum(sub[finite] < 0.05))
    n_fail_high = int(np.sum(sub[finite] > 0.95))
    evidence_summary[method] = {
        "n_total": n_total,
        "n_fail_005": n_fail_low,
        "n_fail_095": n_fail_high,
        "n_pass": n_total - n_fail_low - n_fail_high,
        "pass_rate": round((n_total - n_fail_low - n_fail_high) / max(n_total, 1), 3),
    }

evidence_doc = {
    "id": "plot_pte_heatmap",
    "generated": datetime.now(timezone.utc).isoformat(),
    "output": {"png": output_png.name},
    "evidence": {
        "n_sysmaps": n_maps,
        "n_bins": n_bins,
        "n_methods": n_methods,
        "methods": methods,
        "caveat": "Error bars not validated; PTE may reflect error bar mis-estimation",
        "per_method": evidence_summary,
    },
}

evidence_path.parent.mkdir(parents=True, exist_ok=True)
save_evidence(evidence_doc, evidence_path, snakemake)
snakemake_log(snakemake, f"Saved: {evidence_path}")
