"""Systematic uncertainty budget: bias significance heatmap.

For each (systematic, bin) pair, computes the bias significance:
  S_bias = sqrt( Σ_ℓ X(ℓ)² / σ(ℓ)² )

where X(ℓ) = C^{κS}_ℓ · C^{fS}_ℓ / C^{SS}_ℓ is the contamination metric
and σ(ℓ) is the Knox uncertainty on C^{Eκ}_ℓ.

S_bias < 1 means the systematic bias is sub-noise (negligible).
S_bias > 1 means the systematic contributes > 1σ of bias.

Produces a heatmap: rows = systematics, columns = z-bins, panels = CMB experiments.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.plot_utils import FW, FH, save_evidence, setup_theme, LABELS, SYSMAP_LABELS, TOM_BINS


snakemake = snakemake  # type: ignore # noqa: F821

method = snakemake.wildcards.method
output_png = Path(snakemake.output.png)
evidence_path = Path(snakemake.output.evidence)
sys_dir = Path(snakemake.params.sys_dir)
signal_dir = Path(snakemake.params.signal_dir)
cmbk_sys_dir = Path(snakemake.params.cmbk_sys_dir)
sysmaps = snakemake.params.sysmaps
cmbk_experiments = snakemake.params.cmbk_experiments

snakemake_log(snakemake, f"Systematic budget: {method}, {len(sysmaps)} systematics × {len(cmbk_experiments)} CMB")


n_bins = 6
n_sys = len(sysmaps)
n_cmbk = len(cmbk_experiments)

# Compute bias significance for each (systematic, bin, cmbk) combination
bias_sig = np.full((n_cmbk, n_sys, n_bins), np.nan)

for ic, cmbk in enumerate(cmbk_experiments):
    for isys, sysmap in enumerate(sysmaps):
        # Load C^{κS} and C^{SS}
        cmbk_sys_path = cmbk_sys_dir / f"{sysmap}_x_{cmbk}_cls.npz"
        if not cmbk_sys_path.exists():
            snakemake_log(snakemake, f"  Missing: {cmbk_sys_path.name}")
            continue
        cmbk_sys = np.load(str(cmbk_sys_path), allow_pickle=True)
        cl_kS = cmbk_sys["cls"][0]
        cl_SS = cmbk_sys.get("cl_sys_auto", None)
        if cl_SS is None:
            snakemake_log(snakemake, f"  No cl_sys_auto in {cmbk_sys_path.name}, skipping")
            continue

        for ibin, bin_id in enumerate(TOM_BINS):
            # Load C^{fS} (shear × systematic)
            fS_path = sys_dir / f"{sysmap}_x_{method}_bin{bin_id}_cls.npz"
            if not fS_path.exists():
                continue
            fS_data = np.load(str(fS_path), allow_pickle=True)
            cl_fS = fS_data["cls"][0]  # E-mode
            ells = fS_data["ells"]

            # Compute X = C^{κS} * C^{fS} / C^{SS}
            safe = np.abs(cl_SS) > 1e-30
            X = np.zeros_like(cl_kS)
            X[safe] = cl_kS[safe] * cl_fS[safe] / cl_SS[safe]

            # Load signal Knox variance
            sig_path = signal_dir / f"{method}_bin{bin_id}_x_{cmbk}_cls.npz"
            if not sig_path.exists():
                continue
            sig_data = np.load(str(sig_path), allow_pickle=True)
            knox_var = sig_data["knox_var"]

            # Compute bias significance: sqrt(Σ X²/σ²)
            valid = (knox_var > 0) & safe[:len(knox_var)]
            if valid.sum() == 0:
                continue
            bias_chi2 = np.sum(X[:len(knox_var)][valid] ** 2 / knox_var[valid])
            bias_sig[ic, isys, ibin] = np.sqrt(bias_chi2)

snakemake_log(snakemake, f"  Computed bias significance: {np.nansum(~np.isnan(bias_sig))}/{bias_sig.size} entries")

# --- Plot ---
setup_theme(style="white")

fig, axes = plt.subplots(1, n_cmbk, figsize=(FW * n_cmbk + 1, FH), squeeze=False)

bin_labels = [f"z-bin {b}" for b in TOM_BINS]
sys_labels_ordered = [SYSMAP_LABELS.get(s, s) for s in sysmaps]

for ic, cmbk in enumerate(cmbk_experiments):
    ax = axes[0, ic]
    data = bias_sig[ic]  # (n_sys, n_bins)

    # Cap display at vmax for color scale, but annotate actual values
    vmax = max(3.0, np.nanmax(data) * 0.8) if np.any(~np.isnan(data)) else 3.0

    im = ax.imshow(
        data, aspect="auto", cmap="RdYlGn_r",
        vmin=0, vmax=vmax,
        interpolation="nearest",
    )

    # Annotate cells
    for isys in range(n_sys):
        for ibin in range(n_bins):
            val = data[isys, ibin]
            if np.isnan(val):
                ax.text(ibin, isys, "---", ha="center", va="center", fontsize=8, color="0.5")
            else:
                color = "white" if val > vmax * 0.6 else "black"
                ax.text(ibin, isys, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold" if val > 1 else "normal", color=color)

    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_yticks(range(n_sys))
    ax.set_yticklabels(sys_labels_ordered if ic == 0 else [], fontsize=9)
    ax.set_title(LABELS.get(cmbk, cmbk), fontsize=12, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r"Bias significance ($\sigma$)", fontsize=9)

    # Footnote for dashes — place below figure, not on axis
    if np.any(np.isnan(data)) and ic == 0:
        fig.text(
            0.5, -0.02,
            r"— = $C^{SS}_\ell \approx 0$ (no systematic auto-power; contamination negligible by construction)",
            fontsize=8, color="0.4", ha="center",
        )

fig.suptitle(
    f"Systematic uncertainty budget ({method})\n"
    r"$S_{\rm bias} = \sqrt{\sum_\ell X(\ell)^2 / \sigma(\ell)^2}$"
    "  — values > 1 indicate significant contamination",
    fontsize=11, y=1.08,
)

plt.tight_layout()

output_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_png, bbox_inches="tight")
plt.close(fig)
snakemake_log(snakemake, f"Saved: {output_png}")

# --- Evidence ---
# Per-systematic summary: max bias significance across bins
per_sys_summary = {}
for isys, sysmap in enumerate(sysmaps):
    per_cmbk = {}
    for ic, cmbk in enumerate(cmbk_experiments):
        vals = bias_sig[ic, isys, :]
        if np.all(np.isnan(vals)):
            continue
        per_cmbk[cmbk] = {
            "max_bias_sigma": round(float(np.nanmax(vals)), 2),
            "mean_bias_sigma": round(float(np.nanmean(vals)), 2),
            "n_bins_above_1sigma": int(np.nansum(vals > 1)),
        }
    per_sys_summary[sysmap] = per_cmbk

# Overall summary
all_vals = bias_sig[~np.isnan(bias_sig)]
n_above_1 = int(np.sum(all_vals > 1)) if len(all_vals) > 0 else 0

evidence_doc = {
    "id": "plot_systematic_budget",
    "generated": datetime.now(timezone.utc).isoformat(),
    "artifacts": {"png": output_png.name},
    "evidence": {
        "method": method,
        "n_systematics": n_sys,
        "n_cmbk": n_cmbk,
        "n_bins": n_bins,
        "total_entries": int(np.sum(~np.isnan(bias_sig))),
        "n_above_1sigma": n_above_1,
        "n_above_3sigma": int(np.sum(all_vals > 3)) if len(all_vals) > 0 else 0,
        "max_bias_sigma": round(float(np.nanmax(all_vals)), 2) if len(all_vals) > 0 else None,
        "per_systematic": per_sys_summary,
    },
}

save_evidence(evidence_doc, evidence_path, snakemake)
snakemake_log(snakemake, f"Saved: {evidence_path}")
snakemake_log(snakemake, "Done!")
