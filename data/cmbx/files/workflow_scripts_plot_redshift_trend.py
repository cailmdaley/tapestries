"""Plot C^{Eκ}_ℓ for all tomographic bins on one panel.

Signal should increase with bin number as the lensing kernel overlap grows.
One plot per method × CMB experiment.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.plot_utils import FW, FH, setup_theme, LABELS, BIN_PALETTE, TOM_BINS, save_evidence
from dr1_notebooks.scratch.cdaley.spectrum_utils import load_cross_spectrum, load_evidence


snakemake = snakemake  # type: ignore # noqa: F821

method = snakemake.wildcards.method
cmbk = snakemake.wildcards.cmbk
output_png = Path(snakemake.output.png)
evidence_path = Path(snakemake.output.evidence)

snakemake_log(snakemake, f"Plotting redshift trend: {method} x {cmbk}")

# --- Load theory ---
theory_npz_path = Path(snakemake.input.theory_npz)
theory_data = np.load(str(theory_npz_path), allow_pickle=True)

# --- Load all bins ---
palette = BIN_PALETTE

bin_data = {}
for npz_path in snakemake.input.npz_files:
    spec = load_cross_spectrum(npz_path)
    bid = spec["metadata"]["bin"]
    spec["snr"] = load_evidence(npz_path).get("snr")
    bin_data[bid] = spec

# --- Plot ---
setup_theme("ticks")
fig, ax = plt.subplots(1, 1, figsize=(FW, FH))

for i, bid in enumerate(TOM_BINS):
    if bid not in bin_data:
        continue
    spec = bin_data[bid]
    ells = spec["ells"]
    prefactor = ells
    y = prefactor * spec["cl_e"]
    yerr = prefactor * spec["err"] if spec["err"] is not None else None

    label = f"bin {bid}"
    if spec["snr"] is not None:
        label += f" (S/N={spec['snr']:.1f})"

    ax.errorbar(
        ells, y, yerr=yerr,
        fmt="o", color=palette[i], markersize=5,
        capsize=2, elinewidth=1, label=label,
    )

    # Overlay binned theory as a thin line
    key_ells = f"ells_binned_bin{bid}"
    key_cl = f"cl_binned_bin{bid}"
    key_ia_cl = f"cl_ia_binned_bin{bid}"
    if key_ells in theory_data and key_cl in theory_data:
        ells_th = theory_data[key_ells]
        cl_th = theory_data[key_cl]
        ax.plot(ells_th, ells_th * cl_th, "-", color=palette[i], lw=1.0, alpha=0.3)
    # Overlay IA theory (dashed, more prominent)
    if key_ells in theory_data and key_ia_cl in theory_data:
        ells_th = theory_data[key_ells]
        cl_ia = theory_data[key_ia_cl]
        ax.plot(ells_th, ells_th * cl_ia, "--", color=palette[i], lw=1.2, alpha=0.6)

ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)

# Theory legend entries (one per line style, not per bin)
has_ia_theory = any(f"cl_ia_binned_bin{b}" in theory_data for b in TOM_BINS)
ax.plot([], [], "-", color="gray", lw=1.0, alpha=0.4, label="Planck 2018")
if has_ia_theory:
    ax.plot([], [], "--", color="gray", lw=1.2, alpha=0.6, label="+ NLA IA")

method_label = LABELS.get(method, method)
cmbk_label = LABELS.get(cmbk, cmbk.upper())
ax.set_title(f"{method_label} $\\times$ {cmbk_label}: redshift trend", fontsize=13)
ylabel = r"$\ell\, C_\ell^{\kappa\delta}$" if method == "density" else r"$\ell\, C_\ell^{\kappa E}$"
ax.set_ylabel(ylabel)
ax.set_xlabel(r"Multipole $\ell$")
ax.set_xscale("log")
ax.legend(loc="best", fontsize=9, ncol=2)

import seaborn as sns; sns.despine()
fig.tight_layout()

output_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_png, bbox_inches="tight")
plt.close(fig)

snakemake_log(snakemake, f"Saved: {output_png}")

# --- Evidence: redshift trend ---
from datetime import datetime, timezone

evidence_doc = {
    "id": f"redshift_trend_{method}_x_{cmbk}",
    "generated": datetime.now(timezone.utc).isoformat(),
    "artifacts": {"png": output_png.name},
}

# Compute mean amplitude per bin (mean of ℓCℓ)
bin_amplitudes = []
for bid in TOM_BINS:
    if bid in bin_data:
        spec = bin_data[bid]
        mean_amp = float(np.mean(spec["ells"] * spec["cl_e"]))
        snr = spec["snr"]
        bin_amplitudes.append({
            "bin": bid,
            "mean_ell_cl": float(f"{mean_amp:.4e}"),
            "snr": round(snr, 1) if snr is not None else None,
        })

if bin_amplitudes:
    snrs = [b["snr"] for b in bin_amplitudes if b["snr"] is not None]
    # Monotonicity of SNR: fraction of consecutive pairs where SNR increases
    n_increasing = sum(1 for j in range(len(snrs)-1) if snrs[j+1] > snrs[j])
    evidence_doc["evidence"] = {
        "per_bin": bin_amplitudes,
        "n_snr_increasing": n_increasing,
        "n_pairs": len(snrs) - 1,
        "snr_monotonic": n_increasing == len(snrs) - 1,
        "method": method,
        "cmbk": cmbk,
    }
    snakemake_log(
        snakemake,
        f"  Redshift trend: {n_increasing}/{len(snrs)-1} SNR increasing, monotonic={n_increasing == len(snrs)-1}",
    )

evidence_path.parent.mkdir(parents=True, exist_ok=True)
save_evidence(evidence_doc, evidence_path, snakemake)
snakemake_log(snakemake, f"Saved: {evidence_path}")
