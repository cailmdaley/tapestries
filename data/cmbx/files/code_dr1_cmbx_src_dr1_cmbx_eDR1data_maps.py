"""HEALPix map-building for galaxy clustering (GC) and weak lensing (WL).

Terminology
-----------
**footprint** : binary map (1 = observed, 0 = not).
**mask** : what NaMaster sees for mode decoupling. Can be a binary footprint,
    a weighted mask, or an apodized mask.
**weights** : per-pixel corrections applied to the map or folded into the
    mask, correcting for observational non-uniformity (e.g. shear
    sum-of-weights, GC effective coverage × visibility).

Pickle files produced by ``gc_cat2map`` / ``wl_cat2map`` store a
``"weights"`` key per tomographic bin — not a ``"mask"`` key — because
these per-pixel values are *weights*, not the final NaMaster mask.
``map2cls`` in ``spectra.py`` reads these weights and uses them as the
NaMaster mask (with apodization).  For GC, selection weights are already
baked into the overdensity by ``process_gc_catalog`` (controlled by
``sel_applied``); for WL, the shear sum-of-weights map serves as the
NaMaster mask.

Three-layer design
------------------
**Selection** (quality cuts):
    ``select_wl``, ``select_gc``

    Apply quality cuts (weight thresholds, bad-tile filtering) to a Polars
    LazyFrame.  Driven by a default YAML config shipped with the package;
    callers can override.

**Pixel aggregation** (low-level):
    ``process_gc_catalog``, ``process_wl_catalog``

    Take a plain dict of pre-filtered arrays (ra, dec, ...) and return sparse
    per-pixel quantities.  No catalog I/O, no tomographic looping.  These are
    the independently testable units — call them directly when you have already
    sliced the catalog yourself (e.g. in a notebook).

**Catalog pipeline** (high-level):
    ``gc_cat2map``, ``wl_cat2map``

    Take a pre-selected catalog (Polars DataFrame or LazyFrame), loop over
    tomographic bins calling the low-level function, and write a pickle.
    No sample selection logic — that's the caller's job via ``select_*``.
"""

from __future__ import annotations

import os
import pickle as pkl
from importlib import resources

import healpy as hp
import numpy as np
import yaml

__all__ = [
    "select_wl",
    "select_gc",
    "process_gc_catalog",
    "gc_cat2map",
    "process_wl_catalog",
    "wl_cat2map",
    "sparse_to_dense",
]


def _load_default_config() -> dict:
    """Load the default selection config shipped with the package."""
    ref = resources.files(__package__) / "selection.yaml"
    with resources.as_file(ref) as path:
        with open(path) as f:
            return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Selection (Layer 1)
# ---------------------------------------------------------------------------


def select_wl(lf, method: str, config: dict | None = None):
    """Apply quality cuts to a WL LazyFrame.

    Parameters
    ----------
    lf : polars LazyFrame or DataFrame
        Must contain ``she_{method}_weight`` column.
    method : str
        Shear measurement method (e.g. "lensmc", "metacal").
    config : dict or None
        Override selection config.  If None, loads ``selection.yaml``.

    Returns
    -------
    polars LazyFrame
        Filtered catalog (still lazy).
    """
    import polars as pl

    if config is None:
        config = _load_default_config()

    lf = lf.lazy() if hasattr(lf, "lazy") else lf

    wl_cfg = config.get("wl", {})

    # Minimum weight cut
    min_weight = wl_cfg.get("min_weight", 0)
    w_col = f"she_{method}_weight"
    lf = lf.filter(pl.col(w_col) > min_weight)

    return lf


def select_gc(lf, config: dict | None = None):
    """Apply quality cuts to a GC LazyFrame.

    Parameters
    ----------
    lf : polars LazyFrame or DataFrame
        Must contain ``phz_mode_1`` (for zmax cut) and ``tile_index``
        (if bad_tiles filtering is active).
    config : dict or None
        Override selection config.  If None, loads ``selection.yaml``.

    Returns
    -------
    polars LazyFrame
        Filtered catalog (still lazy).
    """
    import polars as pl

    if config is None:
        config = _load_default_config()

    lf = lf.lazy() if hasattr(lf, "lazy") else lf

    gc_cfg = config.get("gc", {})

    # Redshift cut
    zmax = gc_cfg.get("zmax")
    if zmax is not None:
        lf = lf.filter(pl.col("phz_mode_1") < zmax)

    # Exclude bad tiles
    bad_tiles = gc_cfg.get("bad_tiles", [])
    if bad_tiles:
        lf = lf.filter(~pl.col("tile_index").is_in(bad_tiles))

    return lf


# ---------------------------------------------------------------------------
# Galaxy clustering (GC)
# ---------------------------------------------------------------------------


def process_gc_catalog(
    cat_dict: dict, sel: np.ndarray, nside: int, *, sel_applied: bool = True
) -> dict:
    """Pixel-level overdensity map and shot noise for a galaxy catalog.

    Low-level function: takes pre-filtered arrays, returns sparse pixel data.
    Called once per tomographic bin by ``gc_cat2map``.

    Parameters
    ----------
    cat_dict : dict
        Plain dict with at least "ra" and "dec" (degrees, 1D arrays).
    sel : array_like, shape (12*nside**2,)
        Continuous selection weight; pixels with sel > 0 are included.
    nside : int
        HEALPix NSIDE resolution.
    sel_applied : bool
        If True (default), the overdensity is computed as
        δ = n / (n̄_w · sel) − 1, where n̄_w = Σn / Σsel (weighted mean).
        If False, δ = n / n̄_simple − 1, where n̄_simple = Σn / N_occ
        (unweighted mean over occupied pixels, ignoring sel values).

    Returns
    -------
    dict with keys:
        ipix             - sparse pixel indices (sel > 0)
        map              - overdensity δ per occupied pixel
        weights          - selection weights per occupied pixel
        noise_cl         - flat Poisson shot noise level 1/n_bar (scalar)
        fsky             - observed sky fraction (occupied pixels / total pixels)
        nmean_srad       - mean galaxy density in sr⁻¹
        counts_per_pixel - raw object counts per occupied pixel
        sel_applied      - whether selection weights were used in δ
    """
    ra = cat_dict["ra"]
    dec = cat_dict["dec"]

    npix = hp.nside2npix(nside)
    ipix = hp.ang2pix(nside, ra, dec, lonlat=True)

    mask = sel.copy()
    occupied = mask > 0

    # Count objects per pixel
    nmap = np.bincount(ipix, minlength=npix)

    # Overdensity map
    delta = np.zeros(npix, dtype=float)
    if sel_applied:
        # Weighted mean density: n̄ = Σn / Σsel
        nmean = np.sum(nmap[occupied]) / np.sum(mask[occupied])
        delta[occupied] = nmap[occupied] / (nmean * mask[occupied]) - 1.0
    else:
        # Unweighted mean density: n̄ = Σn / N_occ
        n_occ = np.sum(occupied)
        nmean = np.sum(nmap[occupied]) / n_occ
        delta[occupied] = nmap[occupied] / nmean - 1.0

    # Flat Poisson shot noise: N_l = 1/n_bar (per sr)
    nmean_srad = nmean * npix / (4 * np.pi)
    fsky = np.sum(mask) / npix
    noise_cl = 1.0 / nmean_srad

    return {
        "ipix": np.where(occupied)[0],
        "map": delta[occupied],
        "weights": mask[occupied],
        "noise_cl": noise_cl,
        "fsky": fsky,
        "nmean_srad": nmean_srad,
        "counts_per_pixel": nmap[occupied],
        "sel_applied": sel_applied,
    }


def gc_cat2map(
    cat,
    bins: list[int] = (1, 2, 3, 4, 5, 6),
    combined: bool = True,
    nside: int = 2048,
    visibility: dict | None = None,
    effcov: np.ndarray | None = None,
    sel_applied: bool = True,
    outdir: str = ".",
    filename: str | None = None,
) -> str:
    """Convert a pre-selected GC catalog into tomographic overdensity maps.

    Takes a catalog already filtered by ``select_gc`` (or equivalent).
    Loops over tomographic bins, builds selection masks, and delegates
    pixel aggregation to ``process_gc_catalog``.

    Parameters
    ----------
    cat : polars LazyFrame or DataFrame
        Pre-filtered GC catalog with columns: right_ascension, declination,
        pos_tom_bin_id.
    bins : list of int
        Tomographic bin IDs to process.
    combined : bool
        If True, append an "all" bin combining all galaxies.
    nside : int
        HEALPix NSIDE resolution.
    visibility : dict[int, array_like] or None
        Per-bin visibility weights.  Requires ``effcov``.
    effcov : array_like or None
        Effective coverage map (npix,).  Defines the footprint when provided.
    sel_applied : bool
        Whether to apply selection weights in the overdensity calculation.
        Passed through to ``process_gc_catalog``.  See its docstring for
        details on True vs False behavior.
    outdir : str
        Output directory (used only when ``filename`` is None).
    filename : str or None
        Override output path.

    Returns
    -------
    str
        Path to saved pickle.
    """
    import polars as pl

    if visibility is not None and effcov is None:
        raise ValueError(
            "You must pass the effective coverage (effcov) to use the visibility."
        )

    # Collect to DataFrame once
    df = cat.collect() if hasattr(cat, "collect") else cat

    # Convert required columns to NumPy arrays
    ra = df["right_ascension"].to_numpy()
    dec = df["declination"].to_numpy()
    tom_ids = df["pos_tom_bin_id"].to_numpy()

    npix = hp.nside2npix(nside)

    # Build base footprint
    if effcov is not None:
        base_footprint = np.asarray(effcov, dtype=float).copy()
    else:
        # Global binary footprint from all galaxies
        base_footprint = np.zeros(npix, dtype=float)
        all_ipix = hp.ang2pix(nside, ra, dec, lonlat=True)
        base_footprint[np.unique(all_ipix)] = 1.0

    result = {}

    for i, tom_bin in enumerate(bins):
        print(f"Processing tomographic bin {tom_bin}/{max(bins)}...")

        # Build per-bin selection mask
        if effcov is not None and visibility is not None:
            vis_bin = visibility.get(tom_bin)
            if vis_bin is None:
                raise ValueError(
                    f"Missing visibility map for tomographic bin {tom_bin}"
                )
            selmask = effcov * (1 + vis_bin)
        else:
            selmask = base_footprint.copy()

        binmask = tom_ids == tom_bin
        cat_dict = {"ra": ra[binmask], "dec": dec[binmask]}

        bin_result = process_gc_catalog(
            cat_dict, selmask, nside, sel_applied=sel_applied
        )
        result[f"bin{tom_bin}"] = bin_result

    if combined:
        # Combined bin: base footprint only, no visibility
        selmask = base_footprint.copy()
        cat_dict = {"ra": ra, "dec": dec}
        result["combined"] = process_gc_catalog(
            cat_dict, selmask, nside, sel_applied=sel_applied
        )

    if filename is None:
        name = f"maps_gc_nside-{nside}.pkl"
        filename = os.path.join(outdir, name)

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "wb") as f:
        pkl.dump(result, f)

    print(f"Saved maps to {filename}")
    return filename


# ---------------------------------------------------------------------------
# Weak lensing (WL)
# ---------------------------------------------------------------------------


def process_wl_catalog(cat_dict: dict, nside: int, pol_conv: str = "EUCLID") -> dict:
    """Pixel-level weighted shear maps and shape noise for a WL catalog.

    Low-level function: takes pre-filtered arrays, returns sparse pixel data.
    Called once per tomographic bin by ``wl_cat2map``.

    Parameters
    ----------
    cat_dict : dict with keys "ra", "dec", "e1", "e2", "weight"
        Plain dict of pre-filtered arrays; bias subtraction is applied here.
    nside : int
        HEALPix NSIDE resolution.
    pol_conv : {"EUCLID", "IAU", "COSMO"}
        Sign convention of the stored output maps.  Euclid catalogs are
        natively in the Euclid convention (θ from West).

        EUCLID (default): stored as-is from catalog. θ from West.
        IAU: stored as-is from catalog. θ from North (same raw values
             as EUCLID for Euclid data — label only).
        COSMO: e2 is flipped at map time (θ from South).

    Returns
    -------
    dict — all map arrays are sparse (indexed by ``ipix``):
        ipix           - pixel indices
        e1, e2         - weighted mean shear per pixel
        weights        - sum(w) per pixel (shear weights)
        noise_cl       - flat shape noise (scalar)
        fsky           - observed sky fraction
        sigma_eps      - combined per-galaxy shape noise σ_ε
        sigma_eps_comp - per-component shape noise σ_ε/√2
        n_eff_global   - global effective number of galaxies
        nmean_srad     - n_eff per steradian
        c1_sub, c2_sub - additive bias values subtracted
        pol_conv       - sign convention of the stored maps
    """
    pol_conv = pol_conv.upper()
    if pol_conv not in ("EUCLID", "IAU", "COSMO"):
        raise ValueError(
            f"pol_conv must be 'EUCLID', 'IAU', or 'COSMO', got '{pol_conv}'"
        )

    ra = np.asarray(cat_dict["ra"])
    dec = np.asarray(cat_dict["dec"])
    e1 = np.asarray(cat_dict["e1"], dtype=np.float64).copy()
    e2 = np.asarray(cat_dict["e2"], dtype=np.float64).copy()
    w = np.asarray(cat_dict["weight"], dtype=np.float64)

    # COSMO requests e2 flip at map time (backward compat)
    if pol_conv == "COSMO":
        e2 *= -1

    # Additive bias subtraction
    c1 = np.average(e1, weights=w)
    c2 = np.average(e2, weights=w)
    e1 -= c1
    e2 -= c2

    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(nside, ra, dec, lonlat=True)

    # Per-pixel aggregation via bincount
    sum_w = np.bincount(pix, weights=w, minlength=npix)
    sum_w2 = np.bincount(pix, weights=w**2, minlength=npix)
    sum_we1 = np.bincount(pix, weights=w * e1, minlength=npix)
    sum_we2 = np.bincount(pix, weights=w * e2, minlength=npix)

    occupied = sum_w > 0
    ipix_sparse = np.where(occupied)[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        # Weighted mean shear per pixel
        e1_map = np.where(occupied, sum_we1 / sum_w, 0.0)
        e2_map = np.where(occupied, sum_we2 / sum_w, 0.0)

    # Global statistics
    w_sum = float(w.sum())
    w_sq_sum = float((w**2).sum())
    n_eff = w_sum**2 / w_sq_sum if w_sq_sum > 0 else 0.0
    area_sr = float(occupied.sum() * hp.nside2pixarea(nside))
    n_eff_per_sr = n_eff / area_sr if area_sr > 0 else 0.0

    # Shape noise: σ_ε² = Σw²(e1²+e2²) / Σw²  (w²-weighted; map noise scales as Σw²σ²/(Σw)²)
    sigma_eps_sq = float(((w**2) * (e1**2 + e2**2)).sum() / w_sq_sum) if w_sq_sum > 0 else 0.0
    sigma_eps = np.sqrt(sigma_eps_sq)
    sigma_eps_comp = sigma_eps / np.sqrt(2.0)

    # Flat shape noise: N_l = σ_ε² / (2 n_eff_sr) [per E or B component]
    noise_cl = sigma_eps_sq / (2.0 * n_eff_per_sr) if n_eff_per_sr > 0 else 0.0
    fsky = occupied.sum() / npix

    return {
        "ipix": ipix_sparse,
        "e1": e1_map[ipix_sparse],
        "e2": e2_map[ipix_sparse],
        "weights": sum_w[ipix_sparse],
        "noise_cl": noise_cl,
        "fsky": fsky,
        "sigma_eps": sigma_eps,
        "sigma_eps_comp": sigma_eps_comp,
        "n_eff_global": n_eff,
        "nmean_srad": n_eff_per_sr,
        "c1_sub": c1,
        "c2_sub": c2,
        "pol_conv": pol_conv,
    }


def wl_cat2map(
    cat,
    method: str = "lensmc",
    bins: list[int] = (1, 2, 3, 4, 5, 6),
    combined: bool = True,
    nside: int = 2048,
    pol_conv: str = "EUCLID",
    outdir: str = ".",
    filename: str | None = None,
) -> str:
    """Convert a pre-selected WL catalog into tomographic shear maps.

    Takes a catalog already filtered by ``select_wl`` (or equivalent).
    Loops over tomographic bins, builds per-pixel shear maps, and writes
    a pickle.

    Parameters
    ----------
    cat : polars LazyFrame or DataFrame
        Pre-filtered WL catalog with columns: right_ascension, declination,
        tom_bin_id, she_{method}_e1_corrected, she_{method}_e2_corrected,
        she_{method}_weight.
    method : str
        Shear measurement method (default "lensmc").
    bins : list of int
        Tomographic bin IDs to process.
    combined : bool
        If True, append an "all" bin combining all galaxies.
    nside : int
        HEALPix NSIDE resolution.
    pol_conv : {"EUCLID", "IAU", "COSMO"}
        Sign convention for the output shear maps.
    outdir : str
        Output directory (used only when ``filename`` is None).
    filename : str or None
        Override output path.

    Returns
    -------
    str
        Path to saved pickle.
    """
    import polars as pl

    e1_col = f"she_{method}_e1_corrected"
    e2_col = f"she_{method}_e2_corrected"
    w_col = f"she_{method}_weight"
    columns = ["right_ascension", "declination", e1_col, e2_col, w_col]

    # Collect once with all needed columns
    lf = cat.lazy() if hasattr(cat, "lazy") else cat
    full_table = lf.select(columns + ["tom_bin_id"]).drop_nulls().collect()

    result = {}

    for i, tom_bin in enumerate(bins):
        print(f"Processing WL tomographic bin {tom_bin}/{max(bins)}...")

        table = full_table.filter(pl.col("tom_bin_id") == tom_bin)

        if table.height == 0:
            raise RuntimeError(
                f"No galaxies for method={method} bin={tom_bin} after filtering"
            )

        cat_dict = {
            "ra": table["right_ascension"].to_numpy(),
            "dec": table["declination"].to_numpy(),
            "e1": table[e1_col].to_numpy(),
            "e2": table[e2_col].to_numpy(),
            "weight": table[w_col].to_numpy(),
        }

        result[f"bin{tom_bin}"] = process_wl_catalog(
            cat_dict, nside, pol_conv=pol_conv
        )

    if combined:
        print("Processing WL combined bin (all)...")
        cat_dict = {
            "ra": full_table["right_ascension"].to_numpy(),
            "dec": full_table["declination"].to_numpy(),
            "e1": full_table[e1_col].to_numpy(),
            "e2": full_table[e2_col].to_numpy(),
            "weight": full_table[w_col].to_numpy(),
        }
        result["combined"] = process_wl_catalog(
            cat_dict, nside, pol_conv=pol_conv
        )

    if filename is None:
        name = f"maps_wl_{method}_nside-{nside}.pkl"
        filename = os.path.join(outdir, name)

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "wb") as f:
        pkl.dump(result, f)

    print(f"Saved WL maps to {filename}")
    return filename


def sparse_to_dense(ipix, values, nside):
    """Reconstruct a dense HEALPix map from sparse pixel indices and values.

    Parameters
    ----------
    ipix : array_like
        Pixel indices (RING ordering).
    values : array_like
        Values at those pixels.
    nside : int

    Returns
    -------
    ndarray, shape (12*nside**2,)
        Dense map with zeros at unoccupied pixels.
    """
    m = np.zeros(hp.nside2npix(nside))
    m[ipix] = values
    return m
