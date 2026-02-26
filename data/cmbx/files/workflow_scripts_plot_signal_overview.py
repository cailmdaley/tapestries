"""Signal overview: 6 bins with ACT + SPT cross-spectra and Planck 2018 theory.

Hero figure for the cross-correlation analysis. Data with both CMB experiments
overlaid on binned theory predictions. Two outputs: 6-panel per-bin figure +
redshift trend summary (all bins on one panel, ACT only).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.plot_utils import FW, FH, setup_theme, COLORS, LABELS, MARKERS, BIN_PALETTE, TOM_BINS, save_evidence
from dr1_notebooks.scratch.cdaley.spectrum_utils import load_cross_spectrum, compute_snr


snakemake = snakemake  # type: ignore # noqa: F821

# --- Inputs ---
theory_npz_path = Path(snakemake.input.theory_npz)
data_dir = Path(snakemake.input.data_dir)
method = snakemake.wildcards.method

output_png = Path(snakemake.output.png)
evidence_path = Path(snakemake.output.evidence)
trend_png = output_png.parent / output_png.name.replace(".png", "_trend.png")

snakemake_log(snakemake, f"Building signal overview for {method}")

# --- Load data for both CMB experiments ---
bins = TOM_BINS
cmbk_keys = ["act", "spt_winter_gmv"]

bin_results = {}
for bid in bins:
    bin_results[bid] = {}
    for cmbk in cmbk_keys:
        npz_path = data_dir / f"{method}_bin{bid}_x_{cmbk}_cls.npz"
        if not npz_path.exists():
            snakemake_log(snakemake, f"  WARNING: missing {npz_path.name}")
            continue
        spec = load_cross_spectrum(npz_path)
        # Excess S/N: (chi2-dof)/sqrt(2*dof), zero-centered for noise
        snr = compute_snr(spec["cl_e"], spec["knox_var"]) or 0.0
        spec["snr"] = snr
        bin_results[bid][cmbk] = spec

    snakemake_log(
        snakemake,
        f"  bin {bid}: ACT S/N={bin_results[bid].get('act', {}).get('snr', 0):.1f}, "
        f"SPT S/N={bin_results[bid].get('spt_winter_gmv', {}).get('snr', 0):.1f}",
    )

# --- Load binned theory ---
theory_data = np.load(str(theory_npz_path), allow_pickle=True)
theory_binned = {}
for bid in bins + ["all"]:
    key_ells = f"ells_binned_bin{bid}"
    key_cl = f"cl_binned_bin{bid}"
    if key_ells in theory_data and key_cl in theory_data:
        theory_binned[str(bid)] = {
            "ells": theory_data[key_ells],
            "cl": theory_data[key_cl],
        }
snakemake_log(snakemake, f"Loaded theory for {len(theory_binned)} bins")


# ===========================================================================
# FIGURE 1: 6-panel per-bin grid with theory overlay
# ===========================================================================
setup_theme("whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(2.5*FW, 2*FH), sharex=True)

per_bin_evidence = []

for i, bid in enumerate(bins):
    row, col = divmod(i, 3)
    ax = axes[row, col]

    # Theory curve (behind data)
    if bid in theory_binned:
        th = theory_binned[bid]
        ax.plot(
            th["ells"], th["ells"] * th["cl"],
            color="0.3", ls="-", lw=1.2, zorder=1,
            label="Planck 2018" if i == 0 else None,
        )

    bin_ev = {"bin": bid}
    for cmbk in cmbk_keys:
        if cmbk not in bin_results[bid]:
            continue
        d = bin_results[bid][cmbk]
        ells = d["ells"]
        cl_e = d["cl_e"]
        err = d["err"]

        # Slight horizontal offset to avoid overlap
        offset = 0.97 if cmbk == "act" else 1.03
        ax.errorbar(
            ells * offset,
            ells * cl_e,
            yerr=ells * err if err is not None else None,
            fmt=MARKERS[cmbk],
            color=COLORS[cmbk],
            markersize=5,
            capsize=2,
            alpha=0.85,
            label=LABELS[cmbk] if i == 0 else None,
        )

        bin_ev[f"{cmbk}_snr"] = round(d["snr"], 1)

    per_bin_evidence.append(bin_ev)

    # Panel formatting
    panel_label = chr(ord("a") + i)
    act_snr = bin_results[bid].get("act", {}).get("snr", 0)
    spt_snr = bin_results[bid].get("spt_winter_gmv", {}).get("snr", 0)
    ax.set_title(
        f"({panel_label}) bin {bid} (ACT {act_snr:.1f}$\\sigma$, SPT {spt_snr:.1f}$\\sigma$)",
        fontweight="bold",
        fontsize=10,
    )
    ax.axhline(0, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlim(30, 4000)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3),
                        useMathText=True)
    if row == 1:
        ax.set_xlabel(r"$\ell$")
    if col == 0:
        ylabel = r"$\ell \, C_\ell^{\kappa\delta}$" if method == "density" else r"$\ell \, C_\ell^{\kappa E}$"
        ax.set_ylabel(ylabel)

# Legend in first panel
axes[0, 0].legend(fontsize=8, loc="upper right")

method_label = LABELS.get(method, method)
fig.suptitle(
    rf"{method_label} $\times$ CMB-$\kappa$: cross-spectra",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_png, bbox_inches="tight")
plt.close(fig)
snakemake_log(snakemake, f"Saved: {output_png}")


# ===========================================================================
# FIGURE 2: Redshift trend — all bins on one panel with theory
# ===========================================================================
setup_theme("ticks")
fig2, ax2 = plt.subplots(1, 1, figsize=(FW, FH))

# Use ACT as representative (cleaner coverage)
for i, bid in enumerate(bins):
    if "act" not in bin_results[bid]:
        continue
    d = bin_results[bid]["act"]
    ells = d["ells"]
    y = ells * d["cl_e"]
    yerr = ells * d["err"] if d["err"] is not None else None

    label = f"bin {bid}"
    if d["snr"]:
        label += f" (S/N={d['snr']:.1f})"

    # Theory line per bin (same color, thin)
    if bid in theory_binned:
        th = theory_binned[bid]
        ax2.plot(
            th["ells"], th["ells"] * th["cl"],
            color=BIN_PALETTE[i], ls="-", lw=1.0,
            alpha=0.6, zorder=1,
        )

    ax2.errorbar(
        ells, y, yerr=yerr,
        fmt="o", color=BIN_PALETTE[i], markersize=5,
        capsize=2, elinewidth=1, label=label, zorder=2,
    )

ax2.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)

method_label = LABELS.get(method, method)
ax2.set_title(
    rf"{method_label} $\times$ ACT: signal vs redshift bin",
    fontsize=13,
)
ylabel = r"$\ell\, C_\ell^{\kappa\delta}$" if method == "density" else r"$\ell\, C_\ell^{\kappa E}$"
ax2.set_ylabel(ylabel)
ax2.set_xlabel(r"Multipole $\ell$")
ax2.set_xscale("log")
ax2.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3),
                     useMathText=True)
ax2.legend(loc="best", fontsize=9, ncol=2)

fig2.tight_layout()
fig2.savefig(trend_png, bbox_inches="tight")
plt.close(fig2)
snakemake_log(snakemake, f"Saved: {trend_png}")


# ===========================================================================
# EVIDENCE
# ===========================================================================
evidence = {
    "id": f"signal_overview_{method}",
    "generated": datetime.now(timezone.utc).isoformat(),
    "output": {"png": output_png.name, "trend_png": trend_png.name},
    "evidence": {
        "method": method,
        "n_bins": 6,
        "n_cmbk": 2,
        "per_bin": per_bin_evidence,
    },
}

save_evidence(evidence, evidence_path, snakemake)

snakemake_log(snakemake, f"Saved: {evidence_path}")
snakemake_log(snakemake, "Done!")
