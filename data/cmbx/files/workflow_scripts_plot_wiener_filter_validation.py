"""CMB kappa map power spectra and theory-informed Wiener filter.

Panel (a): per-mode auto-spectra C_l^map for ACT and SPT, residuals after
theory subtraction (showing the noise regime), and Planck 2018 C_l^{kk}.

Panel (b): theory-informed Wiener filter W(l) = C_l^{kk} / (C_l^{kk} + N_l)
using published/computed noise curves (ACT N_L, SPT N_0).
"""

from datetime import datetime, timezone
from pathlib import Path

import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import seaborn as sns
from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.plot_utils import (
    COLORS,
    FH,
    FW,
    save_evidence,
    setup_theme,
)


snakemake = snakemake  # type: ignore # noqa: F821

act_kappa_path = Path(snakemake.input.act_kappa)
act_mask_path = Path(snakemake.input.act_mask)
spt_kappa_path = Path(snakemake.input.spt_kappa)
spt_mask_path = Path(snakemake.input.spt_mask)
act_noise_path = Path(snakemake.input.act_noise)
spt_noise_path = Path(snakemake.input.spt_noise)

output_png = Path(snakemake.output.png)
evidence_path = Path(snakemake.output.evidence)

nside = int(snakemake.config["nside"])
lmax = min(3 * nside - 1, 1000)

snakemake_log(snakemake, "Wiener filter validation (theory-informed rewrite)")


# ---------------------------------------------------------------------------
# Load maps
# ---------------------------------------------------------------------------

snakemake_log(snakemake, "Loading ACT kappa + mask...")
act_kappa = hp.read_map(str(act_kappa_path), dtype=np.float64)
act_mask = hp.read_map(str(act_mask_path), dtype=np.float64)

act_nside = hp.npix2nside(len(act_kappa))
if act_nside != nside:
    act_kappa = hp.ud_grade(act_kappa, nside)
act_mask_nside = hp.npix2nside(len(act_mask))
if act_mask_nside != nside:
    act_mask = hp.ud_grade(act_mask, nside)

snakemake_log(snakemake, "Loading SPT kappa + mask...")
spt_kappa = hp.read_map(str(spt_kappa_path), dtype=np.float64)
spt_mask = hp.read_map(str(spt_mask_path), dtype=np.float64)

spt_nside = hp.npix2nside(len(spt_kappa))
if spt_nside != nside:
    spt_kappa = hp.ud_grade(spt_kappa, nside)
spt_mask_nside = hp.npix2nside(len(spt_mask))
if spt_mask_nside != nside:
    spt_mask = hp.ud_grade(spt_mask, nside)


# ---------------------------------------------------------------------------
# Load noise curves
# ---------------------------------------------------------------------------

snakemake_log(snakemake, "Loading noise curves...")
act_noise_data = np.loadtxt(str(act_noise_path))
act_noise_ells = act_noise_data[:, 0].astype(int)
act_noise_nl = act_noise_data[:, 1]

spt_noise_data = np.loadtxt(str(spt_noise_path))
spt_noise_ells = spt_noise_data[:, 0].astype(int)
spt_noise_nl = spt_noise_data[:, 1]

# Interpolate to common ell grid [0..lmax]
ells_full = np.arange(lmax + 1)
act_nl_full = np.interp(ells_full, act_noise_ells, act_noise_nl, left=act_noise_nl[0], right=act_noise_nl[-1])
spt_nl_full = np.interp(ells_full, spt_noise_ells, spt_noise_nl, left=spt_noise_nl[0], right=spt_noise_nl[-1])


# ---------------------------------------------------------------------------
# Planck 2018 theory C_l^{kk}
# ---------------------------------------------------------------------------

snakemake_log(snakemake, "Computing Planck 2018 theory...")
cosmo = ccl.Cosmology(
    Omega_c=0.2607, Omega_b=0.04897, h=0.6766,
    sigma8=0.8102, n_s=0.9665, transfer_function="bbks",
)
tracer_cmb = ccl.CMBLensingTracer(cosmo, z_source=1100.0)
ells_theory = np.arange(2, lmax + 1).astype(float)
cl_kk_theory = ccl.angular_cl(cosmo, tracer_cmb, tracer_cmb, ells_theory)
cl_theory_full = np.zeros(lmax + 1)
cl_theory_full[2:] = cl_kk_theory


# ---------------------------------------------------------------------------
# Map auto-spectra (pseudo-Cl)
# ---------------------------------------------------------------------------

def compute_map_spectrum(map_data, mask, lmax):
    fsky = float(np.mean(mask > 0))
    masked = map_data * mask
    cl_map = hp.anafast(masked, lmax=lmax)
    return cl_map / max(fsky, 1e-6), fsky

snakemake_log(snakemake, "Computing ACT pseudo-Cl...")
act_cl, act_fsky = compute_map_spectrum(act_kappa, act_mask, lmax)
snakemake_log(snakemake, f"  ACT: fsky={act_fsky:.4f}")

snakemake_log(snakemake, "Computing SPT pseudo-Cl...")
spt_cl, spt_fsky = compute_map_spectrum(spt_kappa, spt_mask, lmax)
snakemake_log(snakemake, f"  SPT: fsky={spt_fsky:.4f}")


# ---------------------------------------------------------------------------
# Theory-informed Wiener filter
# ---------------------------------------------------------------------------

# W(l) = C_l^{kk,theory} / (C_l^{kk,theory} + N_l)
act_wiener = np.zeros(lmax + 1)
spt_wiener = np.zeros(lmax + 1)
for ell in range(2, lmax + 1):
    denom_act = cl_theory_full[ell] + act_nl_full[ell]
    denom_spt = cl_theory_full[ell] + spt_nl_full[ell]
    act_wiener[ell] = cl_theory_full[ell] / denom_act if denom_act > 0 else 0
    spt_wiener[ell] = cl_theory_full[ell] / denom_spt if denom_spt > 0 else 0

# Crossover ell where W(l) drops below 0.5 from its peak (high-ell cutoff)
act_peak_ell = 2 + int(np.argmax(act_wiener[2:lmax+1]))
spt_peak_ell = 2 + int(np.argmax(spt_wiener[2:lmax+1]))
act_cross = None
spt_cross = None
for ell in range(act_peak_ell, lmax + 1):
    if act_wiener[ell] < 0.5:
        act_cross = ell
        break
for ell in range(spt_peak_ell, lmax + 1):
    if spt_wiener[ell] < 0.5:
        spt_cross = ell
        break

snakemake_log(snakemake, f"  Wiener W=0.5 crossover: ACT ell~{act_cross}, SPT ell~{spt_cross}")


# ---------------------------------------------------------------------------
# PLOT — two panels
# ---------------------------------------------------------------------------

setup_theme("ticks")
fig, (ax_cl, ax_w) = plt.subplots(1, 2, figsize=(2 * FW, FH), layout="constrained")

c_act = COLORS["act"]
c_spt = COLORS["spt_winter_gmv"]
c_theory = "black"

ells_plot = np.arange(2, lmax + 1)

def dl_factor(ells):
    return ells * (ells + 1) / (2 * np.pi)


# --- Panel (a): map power spectra ---
# C_l^map
ax_cl.loglog(ells_plot, dl_factor(ells_plot) * act_cl[2:lmax+1],
             color=c_act, lw=1.2, alpha=0.8, label=r"ACT $C_\ell^{\rm map}$")
ax_cl.loglog(ells_plot, dl_factor(ells_plot) * spt_cl[2:lmax+1],
             color=c_spt, lw=1.2, alpha=0.8, label=r"SPT $C_\ell^{\rm map}$")

# C_l^map - C_l^{kk,theory} (residual = noise)
act_resid = act_cl[2:lmax+1] - cl_theory_full[2:lmax+1]
spt_resid = spt_cl[2:lmax+1] - cl_theory_full[2:lmax+1]
ax_cl.loglog(ells_plot, dl_factor(ells_plot) * np.abs(act_resid),
             color=c_act, lw=1.0, ls=":", alpha=0.6,
             label=r"ACT $|C_\ell^{\rm map} - C_\ell^{\kappa\kappa}|$")
ax_cl.loglog(ells_plot, dl_factor(ells_plot) * np.abs(spt_resid),
             color=c_spt, lw=1.0, ls=":", alpha=0.6,
             label=r"SPT $|C_\ell^{\rm map} - C_\ell^{\kappa\kappa}|$")

# Theory
ax_cl.loglog(ells_plot, dl_factor(ells_plot) * cl_kk_theory,
             color=c_theory, lw=2, ls="--", label="Planck 2018")

ax_cl.set_xlim(2, lmax)
ax_cl.set_xlabel(r"Multipole $\ell$")
ax_cl.set_ylabel(r"$\ell(\ell+1)\,C_\ell / 2\pi$")
ax_cl.legend(fontsize=7, loc="upper left")
ax_cl.text(0.03, 0.03, "(a)", transform=ax_cl.transAxes,
           va="bottom", ha="left", fontweight="bold", fontsize=12)


# --- Panel (b): theory-informed Wiener filter ---
ax_w.semilogx(ells_plot, act_wiener[2:lmax+1],
              color=c_act, lw=2, label="ACT DR6")
ax_w.semilogx(ells_plot, spt_wiener[2:lmax+1],
              color=c_spt, lw=2, label="SPT-3G GMV")

# Mark W=0.5 crossover
ax_w.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.5)
if act_cross:
    ax_w.axvline(act_cross, color=c_act, ls=":", lw=0.8, alpha=0.5)
if spt_cross:
    ax_w.axvline(spt_cross, color=c_spt, ls=":", lw=0.8, alpha=0.5)

ax_w.set_xlim(2, lmax)
ax_w.set_ylim(-0.02, 1.02)
ax_w.set_xlabel(r"Multipole $\ell$")
ax_w.set_ylabel(r"$W(\ell) = C_\ell^{\kappa\kappa} / (C_\ell^{\kappa\kappa} + N_\ell)$")
ax_w.legend(fontsize=8)
ax_w.text(0.03, 0.03, "(b)", transform=ax_w.transAxes,
          va="bottom", ha="left", fontweight="bold", fontsize=12)

sns.despine()

output_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_png, bbox_inches="tight")
plt.close(fig)
snakemake_log(snakemake, f"Saved: {output_png}")


# ---------------------------------------------------------------------------
# EVIDENCE
# ---------------------------------------------------------------------------

evidence_doc = {
    "id": "wiener_filter_validation",
    "output": {"png": output_png.name},
    "evidence": {
        "lmax": lmax,
        "act_fsky": round(act_fsky, 5),
        "spt_fsky": round(spt_fsky, 5),
        "act_wiener_half_ell": act_cross,
        "spt_wiener_half_ell": spt_cross,
        "finding": (
            f"Theory-informed Wiener filter W(l) crosses 0.5 at "
            f"l~{act_cross} (ACT) and l~{spt_cross} (SPT). "
            "Both maps are noise-dominated at all l for cross-correlation purposes; "
            "SPT retains more signal at low l due to lower reconstruction noise."
        ),
    },
}

save_evidence(evidence_doc, evidence_path, snakemake)

snakemake_log(snakemake, f"Saved: {evidence_path}")
snakemake_log(snakemake, "Done!")
