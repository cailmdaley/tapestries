"""Compute cross-spectrum between a CMB κ map and a VMPZ-ID systematic map.

Needed for the DES×SPT-style contamination metric (2203.12440 §A.4):
  X^f_S(ℓ) = C^{κS}_ℓ · C^{fS}_ℓ / C^{SS}_ℓ

This computes C^{κS}_ℓ. Combined with existing C^{fS}_ℓ from
compute_systematic_cross_spectrum.py, gives the full metric.

Systematic maps: sparse HEALPix at NSIDE 16384 (PIXEL/WEIGHT).
CMB κ: full-sky HEALPix at native resolution.

Follows same NaMaster conventions as compute_cross_spectrum.py.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import healpy as hp
import numpy as np
from nx2pt.namaster_tools import get_workspace
from scipy.stats import chi2 as chi2_dist

from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.nmt_utils import (
    compute_knox_variance, make_cmb_mask, make_scalar_field,
    make_bins, load_systematic_map, apodize_mask, compute_coupled_cell,
)


snakemake = snakemake  # type: ignore # noqa: F821


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

sysmap_name = snakemake.wildcards.sysmap
cmbk = snakemake.wildcards.cmbk

sysmap_path = Path(snakemake.input.sysmap)
cmbk_map_path = Path(snakemake.input.cmbk_map)
cmbk_mask_path = Path(snakemake.input.cmbk_mask)

output_npz = Path(snakemake.output.npz)
evidence_path = Path(snakemake.output.evidence)

nside = int(snakemake.params.nside)
lmin = int(snakemake.params.lmin)
lmax = int(snakemake.params.lmax)
nells = int(snakemake.params.nells)
aposcale = float(snakemake.params.aposcale)
n_iter = int(snakemake.params.n_iter)
wksp_cache = Path(snakemake.params.wksp_cache) if snakemake.params.wksp_cache else None

spec_id = f"{sysmap_name}_x_{cmbk}"

snakemake_log(snakemake, f"CMBk x systematic cross-spectrum: {spec_id}")
snakemake_log(snakemake, f"  Sysmap: {sysmap_path}")
snakemake_log(snakemake, f"  CMBk map: {cmbk_map_path}")
snakemake_log(snakemake, f"  CMBk mask: {cmbk_mask_path}")

# --- Load systematic map ---
snakemake_log(snakemake, f"  Loading systematic map (target nside={nside})...")
sysmap = load_systematic_map(sysmap_path, nside,
                             log_fn=lambda msg: snakemake_log(snakemake, msg))

# --- Load CMB κ map and mask ---
snakemake_log(snakemake, "  Loading CMB kappa map and mask...")
kappa_map = hp.read_map(str(cmbk_map_path))
mask_map = hp.read_map(str(cmbk_mask_path))
cmbk_mask = make_cmb_mask(mask_map, aposcale)
snakemake_log(snakemake, f"  CMBk mask fsky: {np.mean(cmbk_mask > 0):.4f}")

# CMB κ field (spin-0)
cmbk_field = make_scalar_field(kappa_map, cmbk_mask, lmax=lmax, n_iter=n_iter)

# --- Create sysmap field ---
# Mask: intersection of sysmap coverage and CMBk footprint
cmbk_footprint = (cmbk_mask > 0).astype(np.float64)
sysmap_footprint = (sysmap != 0).astype(np.float64)
common_footprint = cmbk_footprint * sysmap_footprint

sysmap_mask = apodize_mask(common_footprint, aposcale, "C2")
snakemake_log(snakemake, f"  Common mask fsky: {np.mean(sysmap_mask > 0):.4f}")

sysmap_field = make_scalar_field(sysmap, sysmap_mask, lmax=lmax, n_iter=n_iter)

# --- Binning ---
bins, bpw_edges = make_bins(lmin, lmax, nells)
snakemake_log(snakemake, f"  Binning: {bins.get_n_bands()} bins")

# --- Cross-spectrum ---
snakemake_log(snakemake, "Computing workspace...")
wksp = get_workspace(
    sysmap_field, cmbk_field, bins,
    wksp_cache=str(wksp_cache) if wksp_cache else None,
)

snakemake_log(snakemake, "Computing coupled C_l...")
pcl = compute_coupled_cell(sysmap_field, cmbk_field)

snakemake_log(snakemake, "Decoupling...")
cl = wksp.decouple_cell(pcl)

ell_eff = bins.get_effective_ells()
nbins_eff = bins.get_n_bands()

snakemake_log(snakemake, f"  ell range: {ell_eff[0]:.0f} - {ell_eff[-1]:.0f}")
snakemake_log(snakemake, f"  cl shape: {cl.shape}")

# --- Covariance (Knox formula — primary for systematic null tests) ---
knox_result = compute_knox_variance(sysmap_field, cmbk_field, bins)
knox_var = knox_result["knox_var"]
fsky_cross = knox_result["fsky_cross"]
cl_sys_binned = knox_result["cl_auto1_binned"]
snakemake_log(
    snakemake,
    f"  Knox fsky={fsky_cross:.4f}, median(n_modes)={np.median(knox_result['n_modes']):.0f}",
)

# --- Save NPZ ---
output_npz.parent.mkdir(parents=True, exist_ok=True)

save_dict = {
    "ells": ell_eff,
    "cls": cl,
    "knox_var": knox_var,
    "cl_sys_auto": cl_sys_binned[0],  # C^{SS}_ℓ — systematic auto-spectrum (coupled/fsky)
    "metadata": {
        "sysmap": sysmap_name,
        "cmbk": cmbk,
        "tracer_type": "cmbk_x_systematic",
        "nside": nside,
        "lmin": lmin,
        "lmax": lmax,
        "aposcale": aposcale,
        "n_iter": n_iter,
        "fsky_cross": fsky_cross,
    },
}

np.savez(output_npz, **save_dict)
snakemake_log(snakemake, f"Saved: {output_npz}")

# --- Evidence (null test) ---
cl_signal = cl[0]

evidence = {
    "id": spec_id,
    "generated": datetime.now(timezone.utc).isoformat(),
    "artifacts": {"npz": output_npz.name},
}

good_knox = knox_var > 0
if np.any(good_knox):
    dof_knox = int(np.sum(good_knox))
    chi2_knox = float(np.sum(cl_signal[good_knox] ** 2 / knox_var[good_knox]))
    pte_knox = float(1.0 - chi2_dist.cdf(chi2_knox, dof_knox))
    evidence["evidence_knox"] = {
        "chi2": round(chi2_knox, 1),
        "dof": dof_knox,
        "pte": round(pte_knox, 4),
    }
    snakemake_log(
        snakemake,
        f"  Knox: chi2={chi2_knox:.1f} (dof={dof_knox}, PTE={pte_knox:.4f})",
    )

evidence_path.parent.mkdir(parents=True, exist_ok=True)
with open(evidence_path, "w") as f:
    json.dump(evidence, f, indent=2)

snakemake_log(snakemake, f"Saved: {evidence_path}")
snakemake_log(snakemake, "Done!")
