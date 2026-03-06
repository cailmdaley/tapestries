"""Compare catalog-based and map-based cross-spectra.

Overlay both cross-spectra to validate that map-making and spectrum
computation are consistent. Agreement at the percent level confirms
both pipelines are correct.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.plot_utils import FW, FH, setup_theme
from dr1_notebooks.scratch.cdaley.spectrum_utils import load_cross_spectrum


# Snakemake always provides this
snakemake = snakemake  # type: ignore # noqa: F821


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

spec_map = load_cross_spectrum(snakemake.input.map_npz)
spec_cat = load_cross_spectrum(snakemake.input.cat_npz)

ells_map = spec_map["ells"]
cl_E_map = spec_map["cl_e"]
err = spec_map["err"]

ells_cat = spec_cat["ells"]
cl_E_cat = spec_cat["cl_e"]

method = spec_map["metadata"].get("method", "?")
bin_id = spec_map["metadata"].get("bin", "?")
cmbk = spec_map["metadata"].get("cmbk", "?")
n_gal = spec_cat["metadata"].get("n_galaxies", "?")

snakemake_log(snakemake, f"Comparing {method} bin{bin_id} × {cmbk}: map vs catalog ({n_gal} galaxies)")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

setup_theme()
fig, axes = plt.subplots(
    2, 1, figsize=(FW, 4/3*FH),
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=True,
)

# ℓCℓ scaling
ell_cl_map = ells_map * cl_E_map
ell_cl_cat = ells_cat * cl_E_cat
if err is not None:
    ell_err = ells_map * err

# --- Upper panel: overlay ---
ax = axes[0]
if err is not None:
    ax.errorbar(
        ells_map, ell_cl_map, yerr=ell_err,
        fmt="o", label="Map-based", capsize=2, ms=4, alpha=0.8,
    )
else:
    ax.plot(ells_map, ell_cl_map, "o", label="Map-based", ms=4, alpha=0.8)

ax.plot(ells_cat, ell_cl_cat, "s", label="Catalog-based", ms=4, alpha=0.8)
ax.axhline(0, color="gray", ls="--", alpha=0.5)
ax.set_ylabel(r"$\ell \, C_\ell^{E\kappa}$")
ax.set_xscale("log")
ax.set_title(f"{method} bin{bin_id} $\\times$ {cmbk}: map vs catalog ({n_gal:,} galaxies)" if isinstance(n_gal, int) else f"{method} bin{bin_id} $\\times$ {cmbk}: map vs catalog")
ax.legend()

# --- Lower panel: fractional difference ---
ax = axes[1]
good = np.abs(cl_E_map) > 0
frac_diff = np.full_like(cl_E_map, np.nan)
frac_diff[good] = (cl_E_cat[good] - cl_E_map[good]) / cl_E_map[good]

ax.plot(ells_map, frac_diff, "ko", ms=4)
ax.axhline(0, color="gray", ls="--", alpha=0.5)
ax.axhspan(-0.1, 0.1, color="gray", alpha=0.1, label=r"$\pm$10\%")
ax.set_xlabel(r"$\ell$")
ax.set_ylabel("Fractional difference\n(cat $-$ map) / map")
ax.set_xscale("log")
ax.set_ylim(-0.5, 0.5)
ax.legend(fontsize=8)

# Summary statistics
if np.any(good):
    rms_frac = float(np.sqrt(np.nanmean(frac_diff[good] ** 2)))
    max_frac = float(np.nanmax(np.abs(frac_diff[good])))
    snakemake_log(snakemake, f"  RMS fractional difference: {rms_frac:.4f}")
    snakemake_log(snakemake, f"  Max fractional difference: {max_frac:.4f}")
    ax.text(
        0.98, 0.95,
        f"RMS: {rms_frac*100:.1f}\\%\nMax: {max_frac*100:.1f}\\%",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

plt.tight_layout()
plt.savefig(snakemake.output.png, bbox_inches="tight")
plt.close()

snakemake_log(snakemake, f"Saved: {snakemake.output.png}")
snakemake_log(snakemake, "Done!")
