"""Compute a single cross-spectrum between Euclid and CMB lensing maps.

This script computes one cross-spectrum (e.g., lensmc bin1 x ACT) and saves
the results to an individual NPZ file with accompanying evidence.json.

Conventions:
  - Weight-based mask: apodize(weight > 0, aposcale, "C2") * weight
  - CMB mask: apodize(cmb_mask, aposcale, "C2")  [gctools pattern]
  - Shear map/catalog values are handled in a shared IAU convention path
  - Explicit spin=2 for shear fields
  - n_iter=1, lmax=3000, lmax_mask=lmax
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pymaster as nmt
from nx2pt.maps import read_healpix_map
from nx2pt.namaster_tools import get_workspace, get_cov_workspace
from scipy.stats import chi2 as chi2_dist

from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_cmbx.eDR1data.spectra import (
    make_mask, make_field, make_catalog_field, make_bins,
    compute_knox_variance, extract_covariance_block,
)

import polars as pl


# Snakemake always provides this
snakemake = snakemake  # type: ignore # noqa: F821


# ---------------------------------------------------------------------------
# Local helpers (thin glue over dr1_cmbx.eDR1data.spectra)
# ---------------------------------------------------------------------------
def make_shear_mask(weight_map, aposcale):
    return make_mask(weight_map, aposcale=aposcale, weights=[weight_map])


def make_cmb_mask(mask_map, aposcale):
    return nmt.mask_apodization(mask_map, aposcale, "C2")


def make_convergence_mask(mask_map, aposcale):
    return make_mask(mask_map, aposcale=aposcale, weights=[mask_map])


def make_shear_field(e1, e2, mask, lmax=3000, lmax_mask=None, n_iter=1):
    return make_field([e1, e2], mask, spin=2, pol_conv="IAU",
                      lmax=lmax, lmax_mask=lmax_mask, n_iter=n_iter)


def make_scalar_field(map_data, mask, lmax=3000, lmax_mask=None, n_iter=1):
    return make_field(map_data, mask, spin=0,
                      lmax=lmax, lmax_mask=lmax_mask, n_iter=n_iter)


def make_shear_catalog_field(lon_deg, lat_deg, weights, e1, e2, lmax):
    pos = np.array([lon_deg, lat_deg])
    return make_catalog_field(pos, weights, [e1, e2], lmax, spin=2, pol_conv="IAU")


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

# --- Extract parameters from snakemake ---
method = snakemake.wildcards.method
bin_id = snakemake.wildcards.bin
cmbk = snakemake.wildcards.cmbk

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
compute_cov = bool(snakemake.params.compute_cov)
compute_knox = bool(getattr(snakemake.params, "compute_knox", compute_cov))
estimator = str(getattr(snakemake.params, "estimator", "map"))
bad_tiles = list(getattr(snakemake.params, "bad_tiles", []))

is_shear = method in ["lensmc", "metacal"]
euclid_spin = 2 if is_shear else 0
cmb_spin = 0

# Optional patch mask for spatial variation test
patch_mask_path = getattr(snakemake.input, "patch_mask", None)
euclid_apod_mask_path = getattr(snakemake.input, "euclid_apod_mask", None)
cmbk_apod_mask_path = getattr(snakemake.input, "cmbk_apod_mask", None)
patch_mask = None
if patch_mask_path is not None:
    from nx2pt.maps import read_healpix_map as read_map
    patch_mask = read_map(str(patch_mask_path))

# Optional theory+noise for improved Gaussian covariance guess spectra.
# Paths passed as params (not inputs) to avoid circular DAG dependency with
# compute_theory_cls. Script checks existence at runtime.
# See namaster-covariance-bug-double-a597421d for the double-coupling bug.
_theory_npz_str = getattr(snakemake.params, "theory_npz", None)
_cmb_noise_str = getattr(snakemake.params, "cmb_noise_curve", None)
theory_npz_path = Path(_theory_npz_str) if _theory_npz_str and Path(_theory_npz_str).exists() else None
cmb_noise_path = Path(_cmb_noise_str) if _cmb_noise_str and Path(_cmb_noise_str).exists() else None
use_theory_guess = theory_npz_path is not None
patch_label = getattr(snakemake.wildcards, "patch", None)
spec_id = f"{method}_bin{bin_id}_x_{cmbk}"
if patch_label:
    spec_id = f"{patch_label}_{spec_id}"

snakemake_log(snakemake, f"Cross-spectrum: {spec_id}")
snakemake_log(snakemake, f"  CMB-k:    {cmbk_map_path}")
snakemake_log(snakemake, f"  Type:     {'shear (spin-2)' if is_shear else 'scalar (spin-0)'}")
snakemake_log(snakemake, f"  Estimator: {estimator}")

# --- Load and create Euclid field ---
# Catalog metadata (populated for catalog-based shear)
_n_gal = None
_c1_sub = None
_c2_sub = None
_cat_sum_w = None
_cat_sum_w2 = None
_cat_sigma2_e1 = None
_cat_sigma2_e2 = None

if method == "sksp":
    # SKSP: kappa in FITS extensions, mask in same file — always map-based
    euclid_map_path = Path(snakemake.input.map)
    euclid_mask_path = Path(snakemake.input.mask)
    snakemake_log(snakemake, f"  Euclid map: {euclid_map_path}")
    kappa_ext = f"KAPPA_ZBIN_{int(bin_id) - 1}"
    euclid_map = read_healpix_map(str(euclid_map_path), field=kappa_ext)
    if patch_mask_path is None and euclid_apod_mask_path is not None:
        euclid_mask = read_healpix_map(str(euclid_apod_mask_path))
        snakemake_log(snakemake, f"  Using official apodized mask: {euclid_apod_mask_path}")
    else:
        euclid_mask_raw = read_healpix_map(str(euclid_mask_path), field="MASK")
        euclid_mask = make_convergence_mask(euclid_mask_raw, aposcale)
    euclid_field = make_scalar_field(euclid_map, euclid_mask, lmax=lmax, n_iter=n_iter)

elif is_shear and estimator == "catalog":
    # Catalog-based shear: NmtFieldCatalog bypasses map-making entirely
    catalog_path = Path(snakemake.input.catalog)
    snakemake_log(snakemake, f"  Catalog: {catalog_path}")

    lf = pl.scan_parquet(catalog_path)
    if bad_tiles:
        lf = lf.filter(~pl.col("tile_index").is_in(bad_tiles))
        snakemake_log(snakemake, f"  Excluding {len(bad_tiles)} bad tiles")
    if bin_id != "all":
        lf = lf.filter(pl.col("tom_bin_id") == int(bin_id))

    e1_col = f"she_{method}_e1_corrected"
    e2_col = f"she_{method}_e2_corrected"
    w_col = f"she_{method}_weight"
    columns = ["right_ascension", "declination", e1_col, e2_col, w_col]
    table = lf.filter(pl.col(w_col) > 0).select(columns).drop_nulls().collect()

    if table.height == 0:
        raise RuntimeError(f"No galaxies for method={method} bin={bin_id} after filtering")

    ra = table["right_ascension"].to_numpy()
    dec = table["declination"].to_numpy()
    e1 = table[e1_col].to_numpy()
    e2 = table[e2_col].to_numpy()
    w = table[w_col].to_numpy()
    del table

    # Subtract weighted-mean additive bias (same as build_maps.py)
    _c1_sub = float(np.average(e1, weights=w))
    _c2_sub = float(np.average(e2, weights=w))
    e1 = e1 - _c1_sub
    e2 = e2 - _c2_sub
    _n_gal = len(ra)
    snakemake_log(snakemake, f"  Loaded {_n_gal:,} galaxies")
    snakemake_log(snakemake, f"  Additive bias removed: c1={_c1_sub:.6e}, c2={_c2_sub:.6e}")

    # Save catalog noise statistics for Knox proxy (before freeing memory).
    # Knox proxy: N^EE = σ²_e / n_eff_sr  (flat shape noise power)
    # n_eff = (Σw)² / Σw²  (effective number per solid angle, requires catalog area Ω)
    _cat_sum_w = float(np.sum(w))
    _cat_sum_w2 = float(np.sum(w**2))
    _cat_sigma2_e1 = float(np.average(e1**2, weights=w))  # weighted variance per component
    _cat_sigma2_e2 = float(np.average(e2**2, weights=w))

    pos = np.array([ra, dec])
    del ra, dec

    # Shear convention handling is centralized in make_shear_catalog_field.
    euclid_field = make_shear_catalog_field(pos[0], pos[1], w, e1, e2, lmax)
    del pos, e1, e2, w
    snakemake_log(snakemake, "  NmtFieldCatalog (spin-2) created")

    # Build auxiliary map-based field for NaMaster Gaussian covariance.
    # NmtFieldCatalog.get_mask() raises ValueError, so covariance workspace
    # requires a map-based NmtField. The covariance is valid because it depends
    # on mask geometry and auto-spectra, not the spectrum measurement method.
    euclid_map_path = Path(snakemake.input.map)
    euclid_mask_path = Path(snakemake.input.mask)
    shear_maps_cov = np.atleast_2d(read_healpix_map(str(euclid_map_path)))
    if patch_mask_path is None and euclid_apod_mask_path is not None:
        euclid_mask = read_healpix_map(str(euclid_apod_mask_path))
        snakemake_log(snakemake, f"  Using official apodized mask: {euclid_apod_mask_path}")
    else:
        weight_map_cov = read_healpix_map(str(euclid_mask_path))
        euclid_mask = make_shear_mask(weight_map_cov, aposcale)
        del weight_map_cov
    euclid_field_for_cov = make_shear_field(
        shear_maps_cov[0], shear_maps_cov[1], euclid_mask,
        lmax=lmax, n_iter=n_iter,
    )
    del shear_maps_cov
    snakemake_log(snakemake, f"  Map-based field for covariance: fsky={np.mean(euclid_mask > 0):.4f}")

elif is_shear:
    # Map-based shear
    euclid_map_path = Path(snakemake.input.map)
    euclid_mask_path = Path(snakemake.input.mask)
    snakemake_log(snakemake, f"  Euclid map: {euclid_map_path}")
    shear_maps = np.atleast_2d(read_healpix_map(str(euclid_map_path)))
    e1, e2 = shear_maps[0], shear_maps[1]
    if patch_mask_path is None and euclid_apod_mask_path is not None:
        euclid_mask = read_healpix_map(str(euclid_apod_mask_path))
        snakemake_log(snakemake, f"  Using official apodized mask: {euclid_apod_mask_path}")
    else:
        weight_map = read_healpix_map(str(euclid_mask_path))
        # Apply spatial patch mask if provided (spatial variation test)
        if patch_mask_path is not None:
            weight_map = weight_map * patch_mask
            snakemake_log(snakemake, f"  Applied patch mask: {patch_label}")
        euclid_mask = make_shear_mask(weight_map, aposcale)
        del weight_map
    euclid_field = make_shear_field(e1, e2, euclid_mask, lmax=lmax, n_iter=n_iter)
    snakemake_log(snakemake, f"  Shear mask fsky: {np.mean(euclid_mask > 0):.4f}")

else:
    # SKS or density: always map-based scalar maps
    euclid_map_path = Path(snakemake.input.map)
    euclid_mask_path = Path(snakemake.input.mask)
    snakemake_log(snakemake, f"  Euclid map: {euclid_map_path}")
    euclid_map = read_healpix_map(str(euclid_map_path))
    if patch_mask_path is None and euclid_apod_mask_path is not None:
        euclid_mask = read_healpix_map(str(euclid_apod_mask_path))
        snakemake_log(snakemake, f"  Using official apodized mask: {euclid_apod_mask_path}")
    else:
        euclid_mask_raw = read_healpix_map(str(euclid_mask_path))
        euclid_mask = make_convergence_mask(euclid_mask_raw, aposcale)
    euclid_field = make_scalar_field(euclid_map, euclid_mask, lmax=lmax, n_iter=n_iter)

# --- Load and create CMB field ---
cmbk_map = read_healpix_map(str(cmbk_map_path))
if patch_mask_path is None and cmbk_apod_mask_path is not None:
    cmbk_mask = read_healpix_map(str(cmbk_apod_mask_path))
    snakemake_log(snakemake, f"  Using official apodized CMB mask: {cmbk_apod_mask_path}")
else:
    cmbk_mask_raw = read_healpix_map(str(cmbk_mask_path))
    # Apply spatial patch mask to CMB mask if provided
    if patch_mask_path is not None:
        cmbk_mask_raw = cmbk_mask_raw * patch_mask
        snakemake_log(snakemake, f"  Applied patch mask to CMB mask")
    cmbk_mask = make_cmb_mask(cmbk_mask_raw, aposcale)
cmbk_field = make_scalar_field(cmbk_map, cmbk_mask, lmax=lmax, n_iter=n_iter)

snakemake_log(snakemake, f"  CMB mask fsky: {np.mean(cmbk_mask > 0):.4f}")

# --- Guard against zero-overlap masks (e.g., patch outside CMB footprint) ---
# For catalog-based fields, euclid_mask is None — check CMB mask only.
fsky_euclid = float(np.mean(euclid_mask > 0)) if euclid_mask is not None else 1.0
fsky_cmb = float(np.mean(cmbk_mask > 0))
if fsky_euclid < 1e-5 or fsky_cmb < 1e-5:
    snakemake_log(snakemake, f"  WARNING: Near-zero mask overlap (fsky_euclid={fsky_euclid:.6f}, fsky_cmb={fsky_cmb:.6f})")
    snakemake_log(snakemake, "  Saving null result and exiting.")
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, ells=np.array([]), cls=np.array([]),
             metadata={"method": method, "bin": bin_id, "cmbk": cmbk, "error": "no_overlap"})
    evidence = {"id": spec_id, "generated": datetime.now(timezone.utc).isoformat(),
                "artifacts": {"npz": output_npz.name}, "error": "no_overlap"}
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2)
    import sys
    sys.exit(0)

# --- Binning (log-spaced, matching field lmax) ---
bins, bpw_edges = make_bins(lmin, lmax, nells)
snakemake_log(snakemake, f"  Binning: {bins.get_n_bands()} bins, ell=[{bpw_edges[0]}, {bpw_edges[-1]}], lmax={lmax}")

# --- Cross-spectrum ---
snakemake_log(snakemake, "Computing workspace...")
wksp = get_workspace(
    euclid_field, cmbk_field, bins,
    wksp_cache=str(wksp_cache) if wksp_cache else None,
)

snakemake_log(snakemake, "Computing coupled C_l...")
pcl = nmt.compute_coupled_cell(euclid_field, cmbk_field)

snakemake_log(snakemake, "Decoupling...")
cl = wksp.decouple_cell(pcl)

bpws = wksp.get_bandpower_windows()
ell_eff = bins.get_effective_ells()

snakemake_log(snakemake, f"  ell range: {ell_eff[0]:.0f} - {ell_eff[-1]:.0f}")
snakemake_log(snakemake, f"  cl shape: {cl.shape}")

# --- Catalog Knox proxy (kept for diagnostic comparison) ---
_catalog_knox_proxy = None
if estimator == "catalog" and is_shear and compute_knox and _cat_sum_w is not None:
    fsky_cmb_cross = float(np.mean(cmbk_mask > 0))
    omega_cross = 4 * np.pi * fsky_cmb_cross
    n_eff_sr = _cat_sum_w**2 / _cat_sum_w2 / omega_cross
    noise_ee = (_cat_sigma2_e1 + _cat_sigma2_e2) / (2.0 * n_eff_sr)
    m2 = cmbk_field.get_mask()
    pcl_kk = nmt.compute_coupled_cell(cmbk_field, cmbk_field) / np.mean(m2**2)
    cl_kk_binned = bins.bin_cell(pcl_kk)
    nbins_eff = bins.get_n_bands()
    n_modes_knox = np.zeros(nbins_eff)
    for b in range(nbins_eff):
        ells_in_bin = bins.get_ell_list(b)
        n_modes_knox[b] = np.sum(2.0 * ells_in_bin + 1.0) * fsky_cmb_cross
    knox_var_catalog = noise_ee * np.abs(cl_kk_binned[0]) / np.maximum(n_modes_knox, 1)
    _catalog_knox_proxy = {
        "knox_var": knox_var_catalog,
        "fsky_cross": fsky_cmb_cross,
        "n_modes": n_modes_knox,
        "n_eff_sr": n_eff_sr,
        "noise_ee": noise_ee,
    }
    snakemake_log(snakemake, f"  Knox proxy: n_eff={n_eff_sr:.2e}/sr, N^EE={noise_ee:.2e}")
    compute_knox = False  # skip standard Knox (uses get_mask() internally)

# --- For catalog-based shear: use map-based field for covariance ---
# NmtFieldCatalog.get_mask() raises ValueError, so the covariance workspace
# must be built from the auxiliary map-based field. The covariance is valid
# because it depends on mask geometry and auto-spectra, not measurement method.
_cov_euclid_field = euclid_field  # default: use the same field
if estimator == "catalog" and is_shear:
    _cov_euclid_field = euclid_field_for_cov

# --- Knox formula covariance ---
cov = None
knox_var = None
knox_result = None
if _catalog_knox_proxy is not None:
    # Use pre-computed catalog Knox proxy
    knox_var = _catalog_knox_proxy["knox_var"]
    knox_result = _catalog_knox_proxy
elif compute_knox or compute_cov:
    snakemake_log(snakemake, "Computing Knox covariance...")
    knox_result = compute_knox_variance(euclid_field, cmbk_field, bins)
    knox_var = knox_result["knox_var"]
    fsky_cross = knox_result["fsky_cross"]
    snakemake_log(
        snakemake,
        f"  Knox fsky={fsky_cross:.4f}, median(n_modes)={np.median(knox_result['n_modes']):.0f}",
    )

if compute_cov:
    snakemake_log(snakemake, "Computing Gaussian covariance...")

    snakemake_log(snakemake, "  Getting covariance workspace...")
    cov_wksp = get_cov_workspace(
        _cov_euclid_field, cmbk_field, _cov_euclid_field, cmbk_field,
        wksp_cache=str(wksp_cache) if wksp_cache else None,
    )

    # Check if theory NPZ has the required auto-spectra keys
    _theory_ready = False
    if use_theory_guess:
        theory_data = np.load(theory_npz_path, allow_pickle=True)
        cl_kk_key = "cl_kk_theory"
        cl_ee_key = f"cl_ee_theory_bin{bin_id}"
        if cl_kk_key in theory_data and cl_ee_key in theory_data:
            _theory_ready = True
        else:
            snakemake_log(snakemake, f"  Theory NPZ missing auto-spectra ({cl_kk_key}, {cl_ee_key}) — falling back to legacy")

    if _theory_ready:
        # --- Theory + noise guess spectra (correct approach) ---
        # Following ACT×DES (2309.04412): pass true C_ℓ + noise to
        # gaussian_covariance(coupled=False), avoiding double-coupling.
        snakemake_log(snakemake, "  Using theory+noise guess spectra (fixed double-coupling)")
        cl_kk_theory = theory_data[cl_kk_key]
        cl_ee_theory = theory_data[cl_ee_key]
        cl_ek_theory = theory_data[f"cl_ia_theory_bin{bin_id}"]

        # CMB reconstruction noise N_L
        if cmb_noise_path is not None:
            noise_data = np.loadtxt(str(cmb_noise_path))
            nl_cmb = np.zeros(lmax + 1)
            nl_ell = noise_data[:, 0].astype(int)
            nl_val = noise_data[:, 1]
            valid = nl_ell <= lmax
            nl_cmb[nl_ell[valid]] = nl_val[valid]
            snakemake_log(snakemake, f"  CMB N_L: loaded {len(nl_ell)} modes from {Path(cmb_noise_path).name}")
        else:
            nl_cmb = np.zeros(lmax + 1)
            snakemake_log(snakemake, "  CMB N_L: none provided (theory-only for kappa auto)")

        # Shape noise from FITS header (flat per-component noise)
        import astropy.io.fits as fits
        with fits.open(str(euclid_map_path)) as hdul:
            noise_cl = float(hdul[1].header.get("NOISE_CL", 0.0))
        snakemake_log(snakemake, f"  Shape noise NOISE_CL = {noise_cl:.3e}")

        # Construct guess spectra arrays with correct shapes.
        # spin-2 auto (shear): shape (4, lmax+1) — [EE, EB, BE, BB]
        guess_11 = np.zeros((4, lmax + 1))
        guess_11[0, :len(cl_ee_theory)] = cl_ee_theory[:lmax + 1] + noise_cl  # EE + shape noise
        guess_11[3, :len(cl_ee_theory)] = noise_cl  # BB ≈ shape noise only (B-mode theory ≈ 0)

        # spin-0 auto (CMB kappa): shape (1, lmax+1) — [κκ]
        guess_22 = np.zeros((1, lmax + 1))
        guess_22[0, :len(cl_kk_theory)] = cl_kk_theory[:lmax + 1] + nl_cmb[:lmax + 1]

        # spin-2 × spin-0 cross: shape (2, lmax+1) — [Eκ, Bκ]
        guess_12 = np.zeros((2, lmax + 1))
        guess_12[0, :len(cl_ek_theory)] = cl_ek_theory[:lmax + 1]
        # Bκ = 0 by construction

        snakemake_log(snakemake, f"  guess_11 EE peak: {np.max(np.arange(lmax+1) * guess_11[0]):.3e}")
        snakemake_log(snakemake, f"  guess_22 κκ peak: {np.max(np.arange(lmax+1) * guess_22[0]):.3e}")
    else:
        # --- Legacy: data-derived pseudo-Cℓ/fsky guess spectra ---
        # WARNING: these suffer from double-coupling when passed to
        # gaussian_covariance(coupled=False). Results are 10-130× overestimated.
        if knox_result is not None and "pcl_11" in knox_result:
            snakemake_log(snakemake, "  Using data-derived guess spectra (double-coupling warning)")
            m_euclid = _cov_euclid_field.get_mask()
            m_cmb = cmbk_field.get_mask()
            guess_11 = knox_result["pcl_11"]
            guess_22 = knox_result["pcl_22"]
            guess_12 = nmt.compute_coupled_cell(_cov_euclid_field, cmbk_field) / np.mean(m_euclid * m_cmb)
        else:
            # Catalog estimator Knox proxy has no pseudo-Cl — skip NaMaster covariance.
            # Regenerate theory files (compute_theory_cls) to enable proper covariance.
            snakemake_log(snakemake, "  WARNING: No theory or pseudo-Cl for covariance — skipping NaMaster Gaussian")
            compute_cov = False

    # Mode-coupling workspace for covariance: must use the same fields
    # as the covariance workspace. For catalog mode, build from map-based field.
    if compute_cov:
        if _cov_euclid_field is not euclid_field:
            snakemake_log(snakemake, "  Building map-based workspace for covariance...")
            _cov_wksp_mc = get_workspace(
                _cov_euclid_field, cmbk_field, bins,
                wksp_cache=str(wksp_cache) if wksp_cache else None,
            )
        else:
            _cov_wksp_mc = wksp

        snakemake_log(snakemake, "  Computing Gaussian covariance...")
        cov = nmt.gaussian_covariance(
            cov_wksp, euclid_spin, cmb_spin, euclid_spin, cmb_spin,
            guess_11, guess_12, guess_12, guess_22, _cov_wksp_mc, _cov_wksp_mc,
        )

        if np.isnan(cov).any():
            snakemake_log(snakemake, "  WARNING: Covariance contains NaNs!")
        snakemake_log(snakemake, f"  Covariance shape: {cov.shape}")


# --- Save NPZ ---
output_npz.parent.mkdir(parents=True, exist_ok=True)

tracer_type = "catalog_shear" if (is_shear and estimator == "catalog") else ("shear" if is_shear else "mass")
metadata = {
    "method": method,
    "bin": bin_id,
    "cmbk": cmbk,
    "tracer_type": tracer_type,
    "estimator": estimator,
    "nside": nside,
    "lmin": lmin,
    "lmax": lmax,
    "aposcale": aposcale,
    "n_iter": n_iter,
}
if _n_gal is not None:
    metadata["n_galaxies"] = _n_gal
    metadata["c1_subtracted"] = _c1_sub
    metadata["c2_subtracted"] = _c2_sub

save_dict = {
    "ells": ell_eff,
    "cls": cl,
    "bpws": bpws,
    "metadata": metadata,
}

if cov is not None:
    save_dict["cov"] = cov
if knox_var is not None:
    save_dict["knox_var"] = knox_var
if knox_result is not None:
    if "cl_auto1_binned" in knox_result:
        save_dict["cl_auto1_binned"] = knox_result["cl_auto1_binned"]
    if "cl_auto2_binned" in knox_result:
        save_dict["cl_auto2_binned"] = knox_result["cl_auto2_binned"]
    save_dict["n_modes"] = knox_result["n_modes"]

np.savez(output_npz, **save_dict)
snakemake_log(snakemake, f"Saved: {output_npz}")


# --- Evidence ---
evidence = {
    "id": spec_id,
    "generated": datetime.now(timezone.utc).isoformat(),
    "artifacts": {
        "npz": output_npz.name,
    },
}

# For spin-2 x spin-0: cl shape is (2, nbins) -- [E-cross, B-cross]
# For spin-0 x spin-0: cl shape is (1, nbins)
# Use first component (E-mode cross for shear, only component for scalar)
nbins = len(ell_eff)
cl_signal = cl[0]

if cov is not None:
    # NaMaster Gaussian covariance for the first spectral component (Eκ).
    cov_nmt = extract_covariance_block(cov, nbins)
    var_nmt = np.diag(cov_nmt) if np.ndim(cov_nmt) == 2 else np.asarray(cov_nmt)
    good_nmt = var_nmt > 0
    if np.any(good_nmt):
        snr = float(np.sqrt(np.sum(cl_signal[good_nmt] ** 2 / var_nmt[good_nmt])))
        dof = int(np.sum(good_nmt))
        chi2_null = float(np.sum(cl_signal[good_nmt] ** 2 / var_nmt[good_nmt]))
        pte_null = float(1.0 - chi2_dist.cdf(chi2_null, dof))
        evidence["evidence"] = {
            "snr": round(snr, 2),
            "chi2_null": round(chi2_null, 1),
            "dof": dof,
            "pte_null": round(pte_null, 4),
        }
        snakemake_log(
            snakemake,
            f"  NaMaster: SNR={snr:.1f}, chi2={chi2_null:.1f} "
            f"(dof={dof}, PTE={pte_null:.4f})",
        )

# Knox formula evidence (always compute when Knox variance available)
if knox_var is not None:
    good_knox = knox_var > 0
    if np.any(good_knox):
        dof_knox = int(np.sum(good_knox))
        chi2_knox = float(np.sum(cl_signal[good_knox] ** 2 / knox_var[good_knox]))
        pte_knox = float(1.0 - chi2_dist.cdf(chi2_knox, dof_knox))
        snr_knox = float(np.sqrt(chi2_knox))
        evidence["evidence_knox"] = {
            "snr": round(snr_knox, 2),
            "chi2_null": round(chi2_knox, 1),
            "dof": dof_knox,
            "pte_null": round(pte_knox, 4),
        }
        snakemake_log(
            snakemake,
            f"  Knox:     SNR={snr_knox:.1f}, chi2={chi2_knox:.1f} "
            f"(dof={dof_knox}, PTE={pte_knox:.4f})",
        )
        if cov is not None:
            both_good = good_nmt & good_knox
            if np.any(both_good):
                ratio = np.median(var_nmt[both_good] / knox_var[both_good])
                snakemake_log(snakemake, f"  NaMaster/Knox variance ratio: {ratio:.2f}")
else:
    snakemake_log(snakemake, "  WARNING: No valid variance elements for evidence")

evidence_path.parent.mkdir(parents=True, exist_ok=True)
with open(evidence_path, "w") as f:
    json.dump(evidence, f, indent=2)

snakemake_log(snakemake, f"Saved: {evidence_path}")
snakemake_log(snakemake, "Done!")
