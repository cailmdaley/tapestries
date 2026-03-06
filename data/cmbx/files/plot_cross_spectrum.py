"""Plot an individual cross-spectrum with error bars from covariance.

Two-panel layout for shear (spin-2 x spin-0):
  - Top: E-mode cross-correlation (signal)
  - Bottom: B-mode cross-correlation (null check)

Single panel for scalar (spin-0 x spin-0).

Uses ell * C_ell scaling to make the signal visible at ell ~ few hundred.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.plot_utils import FW, FH, setup_theme, LABELS
from dr1_notebooks.scratch.cdaley.spectrum_utils import load_cross_spectrum, load_evidence


# Snakemake always provides this
snakemake = snakemake  # type: ignore # noqa: F821


# --- Load data ---
npz_path = Path(snakemake.input.npz)
output_png = Path(snakemake.output.png)

snakemake_log(snakemake, f"Plotting cross-spectrum: {npz_path.stem}")

spec = load_cross_spectrum(npz_path)
ells = spec["ells"]
cl_e = spec["cl_e"]
cl_b = spec["cl_b"]
err_e = spec["err"]
cov = spec["cov"]
meta = spec["metadata"]
nbins = len(ells)

method = meta["method"]
bin_id = meta["bin"]
cmbk = meta["cmbk"]
tracer_type = meta.get("tracer_type", "shear")
is_shear = tracer_type == "shear"

# Evidence sidecar — Knox SNR, PTE, etc.
evidence = load_evidence(npz_path)

# B-mode errors from NaMaster Gaussian covariance (Knox doesn't apply to B)
err_b = None
if is_shear and cov is not None and cov.shape[0] >= 2 * nbins:
    # NaMaster interleaves components: reshape to extract Bκ diagonal (component 1)
    cov_4d = cov.reshape(nbins, 2, nbins, 2)
    var_b = np.diag(cov_4d[:, 1, :, 1])
    err_b = np.sqrt(np.maximum(var_b, 0))


# --- Plot ---
setup_theme("ticks")

if is_shear and cl_b is not None:
    fig, (ax_e, ax_b) = plt.subplots(
        2, 1, figsize=(FW, 4/3*FH), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.05},
    )
else:
    fig, ax_e = plt.subplots(1, 1, figsize=(FW, FH))
    ax_b = None

# ell * C_ell scaling — makes signal visible at ell ~ few hundred
prefactor = ells

# --- E-mode cross (main signal) ---
y_e = prefactor * cl_e
yerr_e = prefactor * err_e if err_e is not None else None

ax_e.errorbar(
    ells, y_e, yerr=yerr_e, fmt="o", color="C0", markersize=5,
    capsize=2, elinewidth=1,
    label=r"$C_\ell^{\kappa E}$" if is_shear else r"$C_\ell^{\kappa\kappa}$",
)
ax_e.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)

# Title and annotation
cmbk_label = LABELS.get(cmbk, cmbk.upper())
method_label = LABELS.get(method, method)
bin_label = "all bins" if bin_id == "all" else f"z-bin {bin_id}"

ax_e.set_title(f"{method_label} {bin_label} \u00d7 {cmbk_label}", fontsize=13)

info_parts = []
if "snr" in evidence:
    info_parts.append(f"S/N = {evidence['snr']:.1f}")
if "pte_null" in evidence:
    info_parts.append(f"PTE = {evidence['pte_null']:.3f}")
if info_parts:
    ax_e.text(
        0.97, 0.95, ", ".join(info_parts), transform=ax_e.transAxes,
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

ax_e.set_ylabel(r"$\ell\, C_\ell$")
if ax_b is None:
    ax_e.set_xlabel(r"Multipole $\ell$")
    ax_e.set_xscale("log")

# --- B-mode cross (null check) ---
if ax_b is not None:
    y_b = prefactor * cl_b
    yerr_b = prefactor * err_b if err_b is not None else None

    ax_b.errorbar(
        ells, y_b, yerr=yerr_b, fmt="s", color="C3", markersize=4,
        capsize=2, elinewidth=1, label=r"$C_\ell^{\kappa B}$",
    )
    ax_b.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax_b.set_ylabel(r"$\ell\, C_\ell^{\kappa B}$")
    ax_b.set_xlabel(r"Multipole $\ell$")
    ax_b.set_xscale("log")
    ax_b.legend(loc="upper right", fontsize=9)

sns.despine()
fig.tight_layout()

output_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_png, bbox_inches="tight")
plt.close(fig)

snakemake_log(snakemake, f"Saved: {output_png}")
