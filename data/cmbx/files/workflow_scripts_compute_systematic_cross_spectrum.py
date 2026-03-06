"""Compute cross-spectrum between a VMPZ-ID systematic map and an Euclid tracer.

Null test: the cross-spectrum should be consistent with zero if the
systematic does not contaminate the tracer.

Systematic maps are sparse HEALPix at NSIDE 16384 (PIXEL/WEIGHT columns).
They are downgraded to the analysis NSIDE and cross-correlated with the
Euclid shear (spin-2) or convergence (spin-0) field.

Follows the same NaMaster conventions as compute_cross_spectrum.py.
Shear sign: pol_conv="EUCLID" (library handles [-e1, +e2] conversion).
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist

import pymaster as nmt
from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.nmt_utils import read_healpix_map, load_systematic_map
from dr1_cmbx.eDR1data.spectra import (
    make_mask, make_field, make_bins, compute_knox_variance, extract_covariance_block,
    get_workspace,
)


# Snakemake always provides this
snakemake = snakemake  # type: ignore # noqa: F821


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

sysmap_name = snakemake.wildcards.sysmap
method = snakemake.wildcards.method
bin_id = snakemake.wildcards.bin

sysmap_path = Path(snakemake.input.sysmap)
maps_pkl_path = Path(snakemake.input.maps_pkl)
bin_idx = int(snakemake.params.bin_idx)

output_npz = Path(snakemake.output.npz)
evidence_path = Path(snakemake.output.evidence)

nside = int(snakemake.params.nside)
lmin = int(snakemake.params.lmin)
lmax = int(snakemake.params.lmax)
nells = int(snakemake.params.nells)
aposcale = float(snakemake.params.aposcale)
n_iter = int(snakemake.params.n_iter)
wksp_cache = Path(snakemake.params.wksp_cache) if snakemake.params.wksp_cache else None

is_shear = method in ["lensmc", "metacal"]
euclid_spin = 2 if is_shear else 0
sys_spin = 0
spec_id = f"{sysmap_name}_x_{method}_bin{bin_id}"

snakemake_log(snakemake, f"Systematic cross-spectrum: {spec_id}")
snakemake_log(snakemake, f"  Sysmap: {sysmap_path}")
snakemake_log(snakemake, f"  Euclid pkl: {maps_pkl_path} (bin_idx={bin_idx})")

# --- Load systematic map (gctools reader handles sparse FITS + downgrade) ---
snakemake_log(snakemake, f"  Loading systematic map (target nside={nside})...")
sysmap = load_systematic_map(sysmap_path, nside,
                             log_fn=lambda msg: snakemake_log(snakemake, msg))

# --- Load Euclid tracer from pickle ---
snakemake_log(snakemake, f"  Loading Euclid maps from pickle...")
with open(maps_pkl_path, "rb") as f:
    data = pickle.load(f)
bin_key = next(k for k in data if k.endswith("bin"))
bd = data[bin_key][f"bin{bin_idx}"]
del data

npix = 12 * nside ** 2

if is_shear:
    e1_sparse = bd["e1"]
    e2_sparse = bd["e2"]
    sw_sparse = bd["sum_weights"]
    ipix = bd["ipix"]
    e1 = np.zeros(npix)
    e2 = np.zeros(npix)
    weight_map = np.zeros(npix)
    e1[ipix] = e1_sparse
    e2[ipix] = e2_sparse
    weight_map[ipix] = sw_sparse
    euclid_mask = make_mask(weight_map, aposcale=aposcale, weights=[weight_map])
    euclid_field = make_field([e1, e2], euclid_mask, spin=2, pol_conv="EUCLID", lmax=lmax, n_iter=n_iter)
    snakemake_log(snakemake, f"  Shear mask fsky: {np.mean(euclid_mask > 0):.4f}")
else:
    # density (and any other spin-0 tracer stored in pickle)
    delta_sparse = bd["map"]
    mask_sparse = bd["mask"]
    ipix = bd["ipix"]
    euclid_map = np.zeros(npix)
    euclid_mask_raw = np.zeros(npix)
    euclid_map[ipix] = delta_sparse
    euclid_mask_raw[ipix] = mask_sparse
    euclid_mask = make_mask(euclid_mask_raw, aposcale=aposcale, weights=[euclid_mask_raw])
    euclid_field = make_field(euclid_map, euclid_mask, spin=0, lmax=lmax, n_iter=n_iter)

# --- Create sysmap field ---
# Mask: intersection of sysmap coverage and survey footprint (from Euclid mask)
survey_footprint = (euclid_field.get_mask() > 0).astype(np.float64)
sysmap_footprint = (sysmap != 0).astype(np.float64)
common_footprint = survey_footprint * sysmap_footprint

# Apodize the common mask
sysmap_mask = nmt.mask_apodization(common_footprint, aposcale, "C2")
snakemake_log(snakemake, f"  Common mask fsky: {np.mean(sysmap_mask > 0):.4f}")

sysmap_field = make_field(sysmap, sysmap_mask, spin=0, lmax=lmax, n_iter=n_iter)

# --- Binning ---
bins, bpw_edges = make_bins(lmin, lmax, nells)
snakemake_log(snakemake, f"  Binning: {bins.get_n_bands()} bins")

# --- Cross-spectrum ---
snakemake_log(snakemake, "Computing workspace...")
wksp = get_workspace(
    sysmap_field, euclid_field, bins,
    wksp_cache=str(wksp_cache) if wksp_cache else None,
)

snakemake_log(snakemake, "Computing coupled C_l...")
pcl = nmt.compute_coupled_cell(sysmap_field, euclid_field)

snakemake_log(snakemake, "Decoupling...")
cl = wksp.decouple_cell(pcl)

ell_eff = bins.get_effective_ells()

snakemake_log(snakemake, f"  ell range: {ell_eff[0]:.0f} - {ell_eff[-1]:.0f}")
snakemake_log(snakemake, f"  cl shape: {cl.shape}")

# --- Covariance ---
snakemake_log(snakemake, "Computing covariance...")
from dr1_cmbx.eDR1data.spectra import get_cov_workspace

# Knox variance (shared utility)
knox_result = compute_knox_variance(sysmap_field, euclid_field, bins)
knox_var = knox_result["knox_var"]
fsky_cross = knox_result["fsky_cross"]
cl_sys_binned = knox_result["cl_auto1_binned"]
snakemake_log(
    snakemake,
    f"  Knox fsky={fsky_cross:.4f}, median(n_modes)={np.median(knox_result['n_modes']):.0f}",
)

# NaMaster Gaussian covariance (iNKA approximation, gctools pattern)
m_sys = sysmap_field.get_mask()
m_euclid = euclid_field.get_mask()
pcl_11 = knox_result["pcl_11"]
pcl_22 = knox_result["pcl_22"]
pcl_12 = nmt.compute_coupled_cell(sysmap_field, euclid_field) / np.mean(m_sys * m_euclid)

cov_wksp = get_cov_workspace(
    sysmap_field, euclid_field, sysmap_field, euclid_field,
    wksp_cache=str(wksp_cache) if wksp_cache else None,
)

cov = nmt.gaussian_covariance(
    cov_wksp, sys_spin, euclid_spin, sys_spin, euclid_spin,
    pcl_11, pcl_12, pcl_12, pcl_22, wksp, wb=wksp,
)

snakemake_log(snakemake, f"  NaMaster covariance shape: {cov.shape}")

# --- Save NPZ ---
output_npz.parent.mkdir(parents=True, exist_ok=True)

save_dict = {
    "ells": ell_eff,
    "cls": cl,
    "cov": cov,
    "knox_var": knox_var,
    "cl_sys_auto": cl_sys_binned[0],  # C^{SS}_ℓ — systematic auto-spectrum (coupled/fsky)
    "metadata": {
        "sysmap": sysmap_name,
        "method": method,
        "bin": bin_id,
        "tracer_type": "shear" if is_shear else ("density" if method == "density" else "mass"),
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
nbins = len(ell_eff)
cl_signal = cl[0]  # First component (E-cross for spin-2, only for spin-0)

# NaMaster covariance variance for the first component.
# extract_covariance_block returns the component covariance block (nbins, nbins).
var_nmt = np.diag(extract_covariance_block(cov, nbins))

evidence = {
    "id": spec_id,
    "generated": datetime.now(timezone.utc).isoformat(),
    "artifacts": {"npz": output_npz.name},
}

# NaMaster PTE
good_nmt = var_nmt > 0
if np.any(good_nmt):
    dof_nmt = int(np.sum(good_nmt))
    chi2_nmt = float(np.sum(cl_signal[good_nmt] ** 2 / var_nmt[good_nmt]))
    pte_nmt = float(1.0 - chi2_dist.cdf(chi2_nmt, dof_nmt))
    evidence["evidence"] = {
        "chi2": round(chi2_nmt, 1),
        "dof": dof_nmt,
        "pte": round(pte_nmt, 4),
    }
    snakemake_log(
        snakemake,
        f"  NaMaster: chi2={chi2_nmt:.1f} (dof={dof_nmt}, PTE={pte_nmt:.4f})",
    )

# Knox formula PTE
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
        f"  Knox:     chi2={chi2_knox:.1f} (dof={dof_knox}, PTE={pte_knox:.4f})",
    )

    # Log ratio of NaMaster to Knox variance
    both_good = good_nmt & good_knox
    if np.any(both_good):
        ratio = np.median(var_nmt[both_good] / knox_var[both_good])
        snakemake_log(snakemake, f"  NaMaster/Knox variance ratio: {ratio:.2f}")
else:
    snakemake_log(snakemake, "  WARNING: No valid Knox variance elements")

evidence_path.parent.mkdir(parents=True, exist_ok=True)
with open(evidence_path, "w") as f:
    json.dump(evidence, f, indent=2)

snakemake_log(snakemake, f"Saved: {evidence_path}")
snakemake_log(snakemake, "Done!")
