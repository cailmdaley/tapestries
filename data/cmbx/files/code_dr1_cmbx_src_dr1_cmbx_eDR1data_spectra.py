"""Unified NaMaster power spectrum estimation for dr1_cmbx.

Layers:
  1. Primitives — mask, field, binning construction
  2. Single-spectrum — `get_spectrum` with workspace caching and optional covariance
  3. Covariance — Knox formula + NaMaster Gaussian
  4. High-level — batch helpers, `map2cls`

Conventions:
  - Spin-2 fields: NaMaster expects HEALPix/COSMO convention (Q along South).
  - pol_conv="COSMO": already native → [e1, e2].
  - pol_conv="IAU" (θ from North): flip e2 → [e1, -e2].
  - pol_conv="EUCLID" (θ from West): flip e1 → [-e1, e2].
  - Workspace caching uses joblib hashing of pixel-space masks + spins.
"""

import os
import time
import pickle as pkl
import warnings

import numpy as np
import healpy as hp
import pymaster as nmt
from tqdm import tqdm

import dr1_cmbx.eDR1data.reader as reader

__all__ = [
    # Primitives
    "make_mask", "make_field", "make_catalog_field", "make_bins",
    # Single-spectrum
    "get_workspace", "get_cov_workspace", "get_spectrum",
    # Covariance
    "compute_knox_variance", "extract_covariance_block", "compute_gaussian_covariance",
    # High-level
    "make_gc_mask",
    "get_spectra", "get_full_cov",
    "map2cls",
    # I/O
    "pickle_spectra",
    # Batch helpers
    "cl_iter",
]


# ===========================================================================
# Layer 1: Primitives
# ===========================================================================

def make_mask(*maps, aposcale=None, weights=None):
    """Build a common mask from one or more maps.

    Parameters
    ----------
    *maps : array_like
        One or more HEALPix maps. The intersection (product of binary > 0)
        defines the footprint.
    aposcale : float or None
        If given, apodization scale in degrees (C2). Applied after weights,
        so all boundaries (including interior weight zeros) are smooth.
        None = no apodization.
    weights : array_like or list of array_like, optional
        Weight maps to multiply in before apodization.

    Returns
    -------
    mask : ndarray
    """
    if len(maps) == 0:
        raise ValueError("At least one map is required")

    footprint = np.ones_like(maps[0], dtype=np.float64)
    for m in maps:
        footprint *= (np.asarray(m, dtype=np.float64) > 0).astype(np.float64)

    if weights is not None:
        if not isinstance(weights, (list, tuple)):
            weights = [weights]
        for w in weights:
            footprint = footprint * np.asarray(w, dtype=np.float64)

    if aposcale is not None and aposcale > 0:
        footprint = nmt.mask_apodization(footprint, aposcale, "C2")

    return footprint


def _apply_pol_conv(e1, e2, pol_conv):
    """Transform spin-2 components to NaMaster's HEALPix/COSMO convention.

    Three conventions, differing in the reference direction for θ=0:

        COSMO:  θ from South.  Right-handed (x=South, y=East).
                Already NaMaster-native → no sign change.
        IAU:    θ from North.  Left-handed (x=North, y=East).
                Flip e2 → [e1, -e2].
        EUCLID: θ from West.   Left-handed (x=West, y=North).
                Flip e1 → [-e1, e2].

    Euclid's convention (position angle from West toward North) is rotated
    90° from standard IAU (from North toward East), so (e1,e2)_Euclid =
    -(e1,e2)_IAU.  The "EUCLID" option handles this directly.
    """
    pol_conv = pol_conv.upper()
    if pol_conv == "COSMO":
        pass
    elif pol_conv == "IAU":
        e2 = -e2
    elif pol_conv == "EUCLID":
        e1 = -e1
    else:
        raise ValueError(
            f"pol_conv must be 'COSMO', 'IAU', or 'EUCLID', got '{pol_conv}'"
        )
    return e1, e2


def make_field(maps, mask, *, spin=0, pol_conv="COSMO",
               lmax=3000, lmax_mask=None, n_iter=1, beam=None):
    """Create an NmtField.

    Parameters
    ----------
    maps : array_like
        For spin=0: a single map (1D array).
        For spin=2: a (e1, e2) pair.
    mask : array_like
        Apodized mask.
    spin : int
        0 or 2.
    pol_conv : {"COSMO", "IAU", "EUCLID"}
        Sign convention of the input ellipticities.
        COSMO: HEALPix native (θ from South) → no flip.
        IAU: standard IAU (θ from North) → flip e2.
        EUCLID: Euclid catalog (θ from West) → flip e1.
    lmax : int
        Maximum multipole.
    lmax_mask : int or None
        Maximum multipole for the mask. Defaults to lmax.
    n_iter : int
        Number of SHT iterations.
    beam : array_like or None
        Beam transfer function.

    Returns
    -------
    field : nmt.NmtField
    """
    if lmax_mask is None:
        lmax_mask = lmax

    kw = dict(n_iter=n_iter, lmax=lmax, lmax_mask=lmax_mask)
    if beam is not None:
        kw["beam"] = beam

    if spin == 0:
        return nmt.NmtField(mask, [np.asarray(maps)], **kw)

    elif spin == 2:
        e1, e2 = _apply_pol_conv(maps[0], maps[1], pol_conv)
        return nmt.NmtField(mask, [e1, e2], spin=2, **kw)

    else:
        raise ValueError(f"spin must be 0 or 2, got {spin}")


def make_catalog_field(pos, weights, values, lmax, *, spin=0, pol_conv="COSMO"):
    """Create an NmtFieldCatalog.

    Parameters
    ----------
    pos : array_like, shape (2, N)
        (RA, Dec) in degrees (lonlat=True).
    weights : array_like, shape (N,)
        Per-object weights.
    values : array_like
        For spin=0: shape (N,) or (1, N).
        For spin=2: shape (2, N) — (e1, e2).
    lmax : int
        Maximum multipole.
    spin : int
        0 or 2.
    pol_conv : {"COSMO", "IAU", "EUCLID"}
        Same convention as make_field for spin=2.

    Returns
    -------
    field : nmt.NmtFieldCatalog
    """
    if spin == 0:
        return nmt.NmtFieldCatalog(pos, weights, [np.asarray(values)], lmax, spin=0, lonlat=True)

    elif spin == 2:
        e1, e2 = _apply_pol_conv(values[0], values[1], pol_conv)
        return nmt.NmtFieldCatalog(pos, weights, [e1, e2], lmax, spin=2, lonlat=True)

    else:
        raise ValueError(f"spin must be 0 or 2, got {spin}")


def make_bins(lmin, lmax, nells, *, spacing="log"):
    """Create NaMaster binning.

    Parameters
    ----------
    lmin : int
        Minimum multipole (lower edge of first bin).
    lmax : int
        Maximum multipole.
    nells : int
        Number of bins.
    spacing : {"log", "linear", "sqrt"}
        Bin spacing. Default is log (geomspace).

    Returns
    -------
    bins : nmt.NmtBin
    bpw_edges : ndarray
        Bin edges.
    """
    if spacing == "log":
        bpw_edges = np.unique(np.geomspace(lmin, lmax, nells + 1).astype(int))
    elif spacing == "linear":
        bpw_edges = np.unique(np.linspace(lmin, lmax, nells + 1).astype(int))
    elif spacing == "sqrt":
        bpw_edges = np.unique(
            (np.linspace(np.sqrt(lmin), np.sqrt(lmax), nells + 1) ** 2).astype(int)
        )
    else:
        raise ValueError(f"spacing must be 'log', 'linear', or 'sqrt', got '{spacing}'")

    ells = np.arange(lmax + 1)
    bpws = np.digitize(ells, bpw_edges) - 1
    bpws[bpws == len(bpw_edges) - 1] = -1
    bpws[:bpw_edges[0]] = -1
    bins = nmt.NmtBin(ells=ells, bpws=bpws, lmax=lmax)
    return bins, bpw_edges


# ===========================================================================
# Layer 2: Single-spectrum computation (with optional covariance)
# ===========================================================================

def _hash_fields(f1, f2):
    """Hash two NaMaster fields by their mask alms and spins.

    The key is symmetric so (f1, f2) and (f2, f1) map to the same cache file.

    Returns None for catalog fields (``NmtFieldCatalog``), which do not
    support ``get_mask()``. Callers should skip file caching in that case.

    Uses ``get_mask()`` (pixel space) rather than ``get_mask_alms()`` to avoid
    triggering an unnecessary SHT. The pixel mask uniquely determines the
    workspace.
    """
    if isinstance(f1, nmt.NmtFieldCatalog) or isinstance(f2, nmt.NmtFieldCatalog):
        return None

    import joblib

    k1 = joblib.hash([f1.get_mask(), f1.spin])
    k2 = joblib.hash([f2.get_mask(), f2.spin])
    return joblib.hash(sorted([k1, k2]))


def get_workspace(f1, f2, bins, *, wksp_cache=None):
    """Get or compute+cache an NmtWorkspace.

    Parameters
    ----------
    f1, f2 : nmt.NmtField or nmt.NmtFieldCatalog
        Fields to cross-correlate.
    bins : nmt.NmtBin
        Binning scheme.
    wksp_cache : str or Path, optional
        Directory for caching workspaces. If None, no caching.
        Caching is also skipped when either field is an NmtFieldCatalog
        (which does not support ``get_mask_alms()``); in that case the
        workspace is computed fresh on every call.

    Returns
    -------
    wksp : nmt.NmtWorkspace
    """
    if wksp_cache is None:
        return nmt.NmtWorkspace.from_fields(f1, f2, bins)

    wksp_cache = str(wksp_cache)
    hash_key = _hash_fields(f1, f2)

    # hash_key is None when either field is a catalog field — skip caching
    if hash_key is None:
        return nmt.NmtWorkspace.from_fields(f1, f2, bins)

    wksp_file = os.path.join(wksp_cache, f"wksp_cl/{hash_key}.fits")

    if os.path.isfile(wksp_file):
        wksp = nmt.NmtWorkspace.from_file(wksp_file)
        wksp.check_unbinned()
        wksp.update_beams(f1.beam, f2.beam)
        wksp.update_bins(bins)
    else:
        wksp = nmt.NmtWorkspace.from_fields(f1, f2, bins)
        os.makedirs(os.path.dirname(wksp_file), exist_ok=True)
        wksp.write_to(wksp_file)

    return wksp


def get_cov_workspace(f1a, f2a, f1b=None, f2b=None, *, wksp_cache=None, hash_masks=None):
    """Get or compute+cache an NmtCovarianceWorkspace.

    Parameters
    ----------
    f1a, f2a : nmt.NmtField
        Fields for the first spectrum.
    f1b, f2b : nmt.NmtField, optional
        Fields for the second spectrum. Defaults to (f1a, f2a).
    wksp_cache : str or Path, optional
        Directory for caching. If None, no caching.
    hash_masks : sequence of 4 array_like, optional
        Explicit masks to use for cache hashing in order (f1a, f2a, f1b, f2b).
        Use this when any field does not implement ``get_mask()`` (e.g. catalogs).

    Returns
    -------
    cov_wksp : nmt.NmtCovarianceWorkspace
    """
    if f1b is None and f2b is None:
        f1b, f2b = f1a, f2a
    elif f1b is None or f2b is None:
        raise ValueError("Must provide either 2 or 4 fields")

    if wksp_cache is None:
        return nmt.NmtCovarianceWorkspace.from_fields(f1a, f2a, f1b, f2b)

    wksp_cache = str(wksp_cache)
    if hash_masks is not None:
        if len(hash_masks) != 4:
            raise ValueError("hash_masks must contain exactly 4 masks")
        masks = [np.asarray(m) for m in hash_masks]
    else:
        masks = [f1a.get_mask(), f2a.get_mask(), f1b.get_mask(), f2b.get_mask()]

    import joblib

    hash_key = joblib.hash([
        masks[0], f1a.spin, masks[1], f2a.spin,
        masks[2], f1b.spin, masks[3], f2b.spin,
    ])

    wksp_file = os.path.join(wksp_cache, f"wksp_cov/{hash_key}.fits")

    if os.path.isfile(wksp_file):
        return nmt.NmtCovarianceWorkspace.from_file(wksp_file)

    cw = nmt.NmtCovarianceWorkspace.from_fields(f1a, f2a, f1b, f2b)
    os.makedirs(os.path.dirname(wksp_file), exist_ok=True)
    cw.write_to(wksp_file)
    return cw


def get_spectrum(
    f1, f2, bins, *,
    nl=None,
    wksp=None,
    wksp_cache=None,
    treat_bb_as_noise=False,
    compute_cov=False,
    compute_knox=False,
    field1_for_cov=None,
    field2_for_cov=None,
    fiducial_spectra=None,
):
    """Compute a single decoupled power spectrum with optional covariance.

    Works with any spin combination (0×0, 2×0, 2×2) and with NmtField or
    NmtFieldCatalog. Returns all spectral components (e.g. EE, EB, BE, BB
    for spin-2 auto) — ``pickle_spectra`` handles component extraction.

    Parameters
    ----------
    f1, f2 : nmt.NmtField or nmt.NmtFieldCatalog
        Fields to cross-correlate.
    bins : nmt.NmtBin
        Binning scheme.
    nl : array_like, optional
        Uncoupled noise power spectrum to subtract.
    wksp : nmt.NmtWorkspace, optional
        Pre-computed workspace. If None, one is computed (and optionally cached).
    wksp_cache : str or Path, optional
        Directory for workspace caching.
    treat_bb_as_noise : bool
        Deprecated. If True for a spin-2 auto-spectrum, subtracts BB from EE
        in-place (legacy behavior).
    compute_cov : bool
        Compute NaMaster Gaussian covariance.
    compute_knox : bool
        Compute Knox formula variance.
    field1_for_cov, field2_for_cov : nmt.NmtField, optional
        Auxiliary map-based fields for covariance when f1/f2 are catalog-based
        (NmtFieldCatalog cannot expose its mask).
    fiducial_spectra : dict, optional
        Theory+noise spectra for Gaussian covariance:
        {"11": C(f1,f1), "12": C(f1,f2), "22": C(f2,f2)}.
        If None, falls back to pseudo-Cl/fsky with a warning.

    Returns
    -------
    result : dict with keys:
        cl       — decoupled power spectrum (all spectral components)
        nl       — decoupled noise (zeros if nl input is None)
        bpws     — bandpower window functions
        wksp     — NmtWorkspace
        ells     — effective multipoles
        cov      — Gaussian covariance (nbins × nbins), or None
        cov_raw  — full NaMaster covariance output, or None
        knox     — Knox variance dict, or None
    """
    if wksp is None:
        wksp = get_workspace(f1, f2, bins, wksp_cache=wksp_cache)

    pcl = nmt.compute_coupled_cell(f1, f2)
    cl = wksp.decouple_cell(pcl)

    if nl is not None:
        nl_decoupled = wksp.decouple_cell(wksp.couple_cell(nl))
        cl = cl - nl_decoupled
    else:
        nl_decoupled = np.zeros_like(cl)

    if treat_bb_as_noise and cl.shape[0] == 4:
        warnings.warn(
            "treat_bb_as_noise is deprecated; use explicit BB handling in "
            "downstream code instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        cl[0] = cl[0] - cl[3]

    bpws = wksp.get_bandpower_windows()
    ells = bins.get_effective_ells()

    result = {
        "cl": cl,
        "nl": nl_decoupled,
        "bpws": bpws,
        "wksp": wksp,
        "ells": ells,
        "cov": None,
        "cov_raw": None,
        "knox": None,
    }

    if compute_knox:
        result["knox"] = compute_knox_variance(f1, f2, bins)

    if compute_cov:
        f1_cov = field1_for_cov if field1_for_cov is not None else f1
        f2_cov = field2_for_cov if field2_for_cov is not None else f2

        wksp_cov = wksp
        cov_hash_masks = None
        if f1_cov is not f1 or f2_cov is not f2:
            wksp_cov = get_workspace(f1_cov, f2_cov, bins, wksp_cache=wksp_cache)
            cov_hash_masks = [
                f1_cov.get_mask(), f2_cov.get_mask(),
                f1_cov.get_mask(), f2_cov.get_mask(),
            ]

        cov, cov_raw, _ = compute_gaussian_covariance(
            f1_cov, f2_cov, f1_cov, f2_cov, bins,
            wksp_ab=wksp_cov, wksp_cd=wksp_cov,
            fiducial_spectra=fiducial_spectra,
            wksp_cache=wksp_cache,
            cov_hash_masks=cov_hash_masks,
        )
        result["cov"] = cov
        result["cov_raw"] = cov_raw

    return result


# ===========================================================================
# Layer 3: Covariance
# ===========================================================================

def compute_knox_variance(field1, field2, bins):
    """Knox formula variance for a cross-spectrum.

    Var(C^{ab}_b) = C^{aa}_b * C^{bb}_b / n_modes_b

    where n_modes = sum(2l+1) * fsky_eff and
    fsky_eff = <w1*w2>^2 / <w1^2 * w2^2>  (Hivon+ 2002, Eq. 17).

    Auto-spectra are pseudo-Cl / fsky (coupled / mean mask product),
    computed directly from the fields without workspace deconvolution.

    Handles both NmtField and NmtFieldCatalog (catalog fields lack
    get_mask(), so fsky is approximated from the map-based field).

    Parameters
    ----------
    field1, field2 : nmt.NmtField or nmt.NmtFieldCatalog
        The two fields being cross-correlated.
    bins : nmt.NmtBin
        Bandpower binning object.

    Returns
    -------
    result : dict with keys:
        knox_var, fsky_cross, n_modes, pcl_11, pcl_22,
        cl_auto1_binned, cl_auto2_binned
    """
    # NmtFieldCatalog doesn't support get_mask()
    try:
        m1 = field1.get_mask()
        fsky_11 = float(np.mean(m1 * m1))
    except (ValueError, AttributeError):
        m1 = None
        fsky_11 = None

    try:
        m2 = field2.get_mask()
        fsky_22 = float(np.mean(m2 * m2))
    except (ValueError, AttributeError):
        m2 = None
        fsky_22 = None

    if m1 is None and m2 is None:
        # Both fields are catalog fields — cannot compute Knox variance
        warnings.warn(
            "compute_knox_variance: both fields lack get_mask() (catalog fields). "
            "Returning zeros for Knox variance.",
            stacklevel=2,
        )
        nbins_eff = bins.get_n_bands()
        zero = np.zeros(nbins_eff)
        return {
            "knox_var": zero,
            "fsky_cross": 0.0,
            "n_modes": zero,
            "pcl_11": None,
            "pcl_22": None,
            "cl_auto1_binned": None,
            "cl_auto2_binned": None,
        }

    # If one field is a catalog, fall back to the other field's mask for fsky
    if m1 is None:
        # field1 is catalog — approximate fsky from m2
        fsky_cross = float(np.mean(m2) ** 2 / np.mean(m2**2))
        pcl_11_norm = fsky_cross
        pcl_22_norm = fsky_22
    elif m2 is None:
        # field2 is catalog — approximate fsky from m1
        fsky_cross = float(np.mean(m1) ** 2 / np.mean(m1**2))
        pcl_11_norm = fsky_11
        pcl_22_norm = fsky_cross
    else:
        fsky_cross = float(np.mean(m1 * m2) ** 2 / np.mean(m1**2 * m2**2))
        pcl_11_norm = fsky_11
        pcl_22_norm = fsky_22

    pcl_11 = nmt.compute_coupled_cell(field1, field1) / pcl_11_norm
    pcl_22 = nmt.compute_coupled_cell(field2, field2) / pcl_22_norm

    cl_auto1_binned = bins.bin_cell(pcl_11)
    cl_auto2_binned = bins.bin_cell(pcl_22)

    nbins_eff = bins.get_n_bands()
    n_modes = np.zeros(nbins_eff)
    for b in range(nbins_eff):
        ells_in_bin = bins.get_ell_list(b)
        n_modes[b] = np.sum(2.0 * ells_in_bin + 1.0) * fsky_cross

    knox_var = (
        np.abs(cl_auto1_binned[0]) * np.abs(cl_auto2_binned[0])
        / np.maximum(n_modes, 1)
    )

    return {
        "knox_var": knox_var,
        "fsky_cross": fsky_cross,
        "n_modes": n_modes,
        "pcl_11": pcl_11,
        "pcl_22": pcl_22,
        "cl_auto1_binned": cl_auto1_binned,
        "cl_auto2_binned": cl_auto2_binned,
    }


def extract_covariance_block(cov_raw, nbins):
    """Extract the first spectral component from NaMaster covariance.

    NaMaster's gaussian_covariance() output has shape
    (n_cls*nbins, n_cls*nbins) with spectral components interleaved.
    Cases:
    - spin-0 x spin-0 (n_cls=1): returns as-is
    - spin-2 x spin-0 (n_cls=2): extracts Ekappa block
    - spin-2 x spin-2 (n_cls=4): extracts EE block

    Parameters
    ----------
    cov_raw : ndarray, shape (n_cls*nbins, n_cls*nbins)
        Raw NaMaster Gaussian covariance.
    nbins : int
        Number of bandpower bins.

    Returns
    -------
    cov : ndarray, shape (nbins, nbins)
        Covariance of the first spectral component.
    """
    n_total = cov_raw.shape[0]
    n_cls = n_total // nbins
    if n_cls == 1:
        return cov_raw[:nbins, :nbins]
    cov_4d = cov_raw.reshape(nbins, n_cls, nbins, n_cls)
    return cov_4d[:, 0, :, 0]


def compute_gaussian_covariance(
    f1a, f2a, f1b, f2b, bins, *,
    wksp_ab=None, wksp_cd=None, cov_wksp=None,
    fiducial_spectra=None,
    wksp_cache=None,
    cov_hash_masks=None,
):
    """NaMaster Gaussian covariance with proper spin handling.

    Parameters
    ----------
    f1a, f2a : nmt.NmtField
        Fields for spectrum A = (f1a x f2a).
    f1b, f2b : nmt.NmtField
        Fields for spectrum B = (f1b x f2b).
    bins : nmt.NmtBin
        Binning scheme.
    wksp_ab, wksp_cd : nmt.NmtWorkspace, optional
        Pre-computed workspaces for spectra A and B.
    cov_wksp : nmt.NmtCovarianceWorkspace, optional
        Pre-computed covariance workspace.
    fiducial_spectra : dict, optional
        Guess spectra for the covariance: {"11": C(f1a,f1b), "12": C(f1a,f2b),
        "22": C(f2a,f2b)}. Optionally include "21": C(f2a,f1b) for
        cross-covariance between different spectra (defaults to "12" if
        absent, which is exact for same-spectrum covariance).
        If None, falls back to pseudo-Cl/fsky with a warning.
    wksp_cache : str or Path, optional
        Directory for workspace caching.
    cov_hash_masks : sequence of 4 array_like, optional
        Explicit masks for covariance hashing/fsky normalization in order
        (f1a, f2a, f1b, f2b). Useful when fields are catalog-based.

    Returns
    -------
    cov : ndarray, shape (nbins, nbins)
        Gaussian covariance (first spectral component extracted).
    cov_raw : ndarray
        Full NaMaster covariance output.
    cov_wksp : nmt.NmtCovarianceWorkspace
    """
    if wksp_ab is None:
        wksp_ab = get_workspace(f1a, f2a, bins, wksp_cache=wksp_cache)
    if wksp_cd is None:
        wksp_cd = get_workspace(f1b, f2b, bins, wksp_cache=wksp_cache)
    if cov_wksp is None:
        cov_wksp = get_cov_workspace(
            f1a, f2a, f1b, f2b,
            wksp_cache=wksp_cache,
            hash_masks=cov_hash_masks,
        )

    spins = [f1a.spin, f2a.spin, f1b.spin, f2b.spin]

    if fiducial_spectra is not None:
        g11 = fiducial_spectra["11"]
        g12 = fiducial_spectra["12"]
        g21 = fiducial_spectra.get("21", g12)  # defaults to g12 for same-spectrum
        g22 = fiducial_spectra["22"]
    else:
        # Legacy: data-derived pseudo-Cl/fsky
        warnings.warn(
            "No fiducial_spectra provided — using pseudo-Cl/fsky. "
            "This suffers from double-coupling and overestimates covariance.",
            stacklevel=2,
        )
        fs = [f1a, f2a, f1b, f2b]
        if cov_hash_masks is not None:
            if len(cov_hash_masks) != 4:
                raise ValueError("cov_hash_masks must contain exactly 4 masks")
            ms = [np.asarray(m) for m in cov_hash_masks]
        else:
            ms = [f.get_mask() for f in fs]
        g11 = nmt.compute_coupled_cell(f1a, f1b) / np.mean(ms[0] * ms[2])
        g12 = nmt.compute_coupled_cell(f1a, f2b) / np.mean(ms[0] * ms[3])
        g21 = nmt.compute_coupled_cell(f2a, f1b) / np.mean(ms[1] * ms[2])
        g22 = nmt.compute_coupled_cell(f2a, f2b) / np.mean(ms[1] * ms[3])

    # NaMaster convention: gaussian_covariance(cw, s1, s2, s3, s4,
    #   C13, C14, C23, C24, wa, wb)
    # Spectrum A = f1a x f2a, Spectrum B = f1b x f2b
    # C13=C(f1a,f1b), C14=C(f1a,f2b), C23=C(f2a,f1b), C24=C(f2a,f2b)
    cov_raw = nmt.gaussian_covariance(
        cov_wksp, *spins,
        g11, g12, g21, g22,
        wksp_ab, wb=wksp_cd,
    )

    nbins = bins.get_n_bands()
    cov = extract_covariance_block(cov_raw, nbins)

    return cov, cov_raw, cov_wksp


# ===========================================================================
# Layer 4: High-level
# ===========================================================================


# ===========================================================================
# Batch helpers (refactored to use new primitives)
# ===========================================================================

def cl_iter(map_names, skipped_pairs):
    """Generator over unique map pairs for Cl computation.

    Parameters
    ----------
    map_names : list of str
    skipped_pairs : list of tuple of str

    Yields
    ------
    icl, i1, i2, nm1, nm2, cl_name
    """
    icl = 0
    for i1, nm1 in enumerate(map_names):
        for i2, nm2 in enumerate(map_names):
            if i2 < i1:
                continue
            if (nm1, nm2) in skipped_pairs:
                continue
            if nm1 == "kappa" and nm2 == "kappa":
                continue
            cl_name = f"{nm1}_{nm2}"
            yield icl, i1, i2, nm1, nm2, cl_name
            icl += 1


def get_spectra(maps, bins, skipped_pairs, *,
                common_mask=False, wksp_cache=None, treat_bb_as_noise=False):
    """Compute auto- and cross-power spectra for all unique pairs of fields.

    Parameters
    ----------
    maps : dict
        Dictionary of fields, each entry must contain:
        - 'field' : nmt.NmtField
        Optionally:
        - 'nl_uncoupled' : noise spectrum for auto-spectra (or omit / set None)
    bins : nmt.NmtBin
    skipped_pairs : list of tuple of str
    common_mask : bool
        If True, reuse the same coupling workspace for all spectra.
    wksp_cache : str or Path, optional
    treat_bb_as_noise : bool
        Deprecated. Passed through to ``get_spectrum``.

    Returns
    -------
    cls : dict
    """
    map_names = list(maps.keys())
    cls = {}
    shared_wksp = None

    for icl, _, _, nm1, nm2, cl_name in cl_iter(map_names, skipped_pairs):
        fg1 = maps[nm1]["field"]
        fg2 = maps[nm2]["field"]
        pnl = maps[nm1].get("nl_uncoupled") if nm1 == nm2 else None

        if common_mask and shared_wksp is not None:
            r = get_spectrum(fg1, fg2, bins, nl=pnl, wksp=shared_wksp,
                             treat_bb_as_noise=treat_bb_as_noise)
        else:
            r = get_spectrum(fg1, fg2, bins, nl=pnl, wksp_cache=wksp_cache,
                             treat_bb_as_noise=treat_bb_as_noise)
            if common_mask and icl == 0:
                shared_wksp = r["wksp"]

        cls[cl_name] = {
            "cl": r["cl"],
            "nl": r["nl"],
            "bpws": r["bpws"],
            "w": r["wksp"],
        }

    return cls


def get_full_cov(maps, cls, bins, skipped_pairs, *,
                 common_mask=False, fiducial_spectra=None, wksp_cache=None):
    """Build full covariance matrix for all auto- and cross-spectra.

    Parameters
    ----------
    maps : dict
    cls : dict
        Output from get_spectra.
    bins : nmt.NmtBin
    skipped_pairs : list
    common_mask : bool
    fiducial_spectra : dict, optional
        If provided, used for all covariance blocks (theory+noise).
    wksp_cache : str or Path, optional

    Returns
    -------
    cov : list of lists of ndarray
    cov_all : ndarray
        Full covariance, shape (n_cls*n_ell, n_cls*n_ell).
    """
    map_names = list(maps.keys())
    n_cls = len(cls.keys())
    leff = bins.get_effective_ells()
    n_ell = len(leff)

    cov = []
    for _ in cl_iter(map_names, skipped_pairs):
        cov.append([None] * n_cls)

    pairs = list(cl_iter(map_names, skipped_pairs))
    total = len(pairs) ** 2
    pbar = tqdm(total=total, desc="Computing covariance blocks", ncols=100)

    shared_cov_wksp = None

    for icl1, _, _, nm1, nm2, cl_name12 in cl_iter(map_names, skipped_pairs):
        f1 = maps[nm1]["field"]
        f2 = maps[nm2]["field"]
        w12 = cls[cl_name12]["w"]

        for icl2, _, _, nm3, nm4, cl_name34 in cl_iter(map_names, skipped_pairs):
            f3 = maps[nm3]["field"]
            f4 = maps[nm4]["field"]
            w34 = cls[cl_name34]["w"]

            if icl2 < icl1:
                cv = cov[icl2][icl1].T
            else:
                cv, _, cov_wksp_out = compute_gaussian_covariance(
                    f1, f2, f3, f4, bins,
                    wksp_ab=w12, wksp_cd=w34,
                    cov_wksp=shared_cov_wksp if common_mask else None,
                    fiducial_spectra=fiducial_spectra,
                    wksp_cache=wksp_cache,
                )

                if common_mask and shared_cov_wksp is None:
                    shared_cov_wksp = cov_wksp_out

            if icl1 == icl2:
                cls[cl_name12]["cov"] = cv
            cov[icl1][icl2] = cv
            pbar.update(1)

    pbar.close()
    cov_blocks = np.array(cov)
    cov_all = np.transpose(cov_blocks, axes=[0, 2, 1, 3]).reshape(
        [n_cls * n_ell, n_cls * n_ell]
    )
    return cov, cov_all


# ===========================================================================
# I/O
# ===========================================================================

def pickle_spectra(cls, cov, cov_all, dndz, leff, fname=""):
    """Save power spectra and covariance to pickle.

    Extracts the first spectral component for spin-2 fields.
    Strips workspace objects before serialization.

    Parameters
    ----------
    cls : dict
    cov : dict or list
    cov_all : dict or ndarray
    dndz : dict
    leff : array_like
    fname : str
    """
    data = {}
    for k_i in cls.keys():
        data[k_i] = {}
        for k_j in cls[k_i].keys():
            if k_j in ["w", "wksp", "cov"]:
                continue
            data[k_i][k_j] = cls[k_i][k_j]

    for k in data.keys():
        data[k]["leff"] = leff
        n_spec = data[k]["cl"].shape[0]

        if n_spec == 1:
            # Spin-0 x spin-0: single component
            data[k]["cl"] = data[k]["cl"][0]
            data[k]["nl"] = data[k]["nl"][0]
        elif n_spec in (2, 4):
            # Spin-2 x spin-0 (n_spec=2): E-mode cross + B-mode for null tests
            # Spin-2 x spin-2 (n_spec=4): EE + BB (noise-debiased if nl was provided)
            bb_idx = 1 if n_spec == 2 else 3
            data[k]["cl_bb"] = data[k]["cl"][bb_idx]
            data[k]["nl_bb"] = data[k]["nl"][bb_idx]
            data[k]["cl"] = data[k]["cl"][0]
            data[k]["nl"] = data[k]["nl"][0]
            data[k]["bpws"] = data[k]["bpws"][0, :, 0, :]
        else:
            raise ValueError(
                f"Unexpected number of spectral components: {n_spec}. "
                "Expected 1 (spin-0), 2 (spin-2 x spin-0), or 4 (spin-2 x spin-2)."
            )

    if fname:
        with open(fname, "wb") as f:
            pkl.dump(
                {"cov": cov, "cov_all": cov_all, "leff": leff,
                 "cls": data, "dndz": dndz},
                f,
            )


# ===========================================================================
# High-level entry points
# ===========================================================================

def make_gc_mask(field, nside, *, visibility=None, effcov=None, footprint=None):
    """Build unapodized per-bin GC masks from a loaded GC map pickle.

    Constructs the binary pixel footprint for each tomographic bin and
    optionally applies visibility weighting, effective coverage, and a
    survey footprint cutout. Returns pre-apodization arrays ready to pass
    to ``map2cls`` as the ``gc['masks']`` dict.

    Parameters
    ----------
    field : dict
        Loaded GC map pickle (output of ``gc_cat2map``).
    nside : int
        HEALPix NSIDE of the output maps.
    visibility : dict or None
        Per-bin visibility maps {bin_index: array}. If provided, effcov is required.
    effcov : array_like or None
        Effective coverage map (shape ``12*nside**2``).
    footprint : array_like or None
        Binary survey footprint to intersect with all bin masks.

    Returns
    -------
    masks : dict
        ``{f'g{i}': ndarray}`` — one unapodized mask per tomographic bin.
    """
    bin_key = next(k for k in field if k.endswith("bin"))
    if visibility is not None and effcov is None:
        raise ValueError("effcov required when visibility is provided")

    masks = {}
    for i, n in enumerate(field[bin_key].keys()):
        ipix = field[bin_key][n]["ipix"]
        m = np.zeros(hp.nside2npix(nside), dtype=np.float64)
        m[ipix] = 1.0
        if footprint is not None:
            m *= np.asarray(footprint, dtype=np.float64)
        if visibility is not None:
            m *= (1 + visibility[i + 1]) * np.asarray(effcov, dtype=np.float64)
        elif effcov is not None:
            m *= np.asarray(effcov, dtype=np.float64)
        masks[f"g{i}"] = m
    return masks


def map2cls(
    gc=None,
    wl=None,
    kappa=None,
    y=None,
    skipped_pairs=(),
    common_mask=False,
    aposcale=0.15,
    lmax=3000,
    lmax_mask=None,
    n_iter=1,
    pol_conv="COSMO",
    lmin=2,
    ell_binning=100,
    spacing="linear",
    compute_cov=False,
    treat_bb_as_noise=False,
    nside=None,
    outfile=None,
    outdir="./data/cls",
    wksp_cache=None,
):
    """Compute angular power spectra from GC, WL, CMB kappa, and/or tSZ-y maps.

    Primary high-level entry point. Galaxy clustering and weak lensing fields
    are loaded from pickle files produced by ``gc_cat2map`` / ``wl_cat2map``.

    Parameters
    ----------
    gc : None, str, or dict
        Galaxy-density source:
        - None: no GC fields.
        - str: path to a GC pkl (``gc_cat2map`` output). Binary footprint
          masks are derived from the pixel indices.
        - dict ``{'map': pkl_path, 'masks': {'g0': arr_or_path, ...}}``:
          explicit per-bin unapodized masks. If ``'masks'`` is absent or
          None, binary footprint masks are derived from the pickle.
    wl : None, str, or dict
        Weak-lensing shear source (same interface as ``gc``):
        - None: no WL fields.
        - str: path to a WL pkl (``wl_cat2map`` output).
        - dict ``{'map': pkl_path, 'masks': {'l0': arr_or_path, ...}}``.
    kappa : None or dict
        CMB lensing field: ``{'map': arr_or_path, 'mask': arr_or_path}``.
        If absent, no CMB kappa field.
    y : None or dict
        Compton-y field: ``{'map': arr_or_path, 'mask': arr_or_path,
        'beam': arr_or_None}``. If absent, no tSZ-y field.
    skipped_pairs : list of tuple of str
    common_mask : bool
        If True, build one shared mask from the intersection of all fields.
    aposcale : float
        C2 apodization scale in degrees.
    lmax : int
    lmax_mask : int or None
    n_iter : int
    pol_conv : {"COSMO", "IAU", "EUCLID"}
        Sign convention for WL shear maps.
    lmin : int
    ell_binning : int or nmt.NmtBin
    spacing : {"linear", "log", "sqrt"}
    compute_cov : bool
    treat_bb_as_noise : bool
        Deprecated. Passed to ``get_spectra``.
    nside : int or None
        HEALPix NSIDE. If None, inferred from the first mask array found.
        Must be provided explicitly when all masks are file paths.
    outfile : str, optional
        Output filename. If None, auto-generated from source names.
    outdir : str
    wksp_cache : str or Path, optional

    Returns
    -------
    cls_fname : str
        Path to the saved pickle file.
    """
    os.makedirs(outdir, exist_ok=True)
    if lmax_mask is None:
        lmax_mask = lmax

    # Binning
    if isinstance(ell_binning, int):
        b, _ = make_bins(lmin, lmax, ell_binning, spacing=spacing)
    else:
        b = ell_binning

    # Normalize GC/WL sources
    gc_src = _resolve_source(gc, "gc")
    wl_src = _resolve_source(wl, "wl")

    # --- Infer nside ---
    if nside is None:
        candidate_masks = []
        for src in (gc_src, wl_src):
            if src is not None and src.get("masks"):
                candidate_masks.append(src["masks"])
        if kappa is not None and isinstance(kappa, dict) and kappa.get("mask") is not None:
            candidate_masks.append({"_k": kappa["mask"]})
        if y is not None and isinstance(y, dict) and y.get("mask") is not None:
            candidate_masks.append({"_y": y["mask"]})
        nside = _infer_nside_from_masks(*candidate_masks)
        if nside is None:
            raise ValueError(
                "Cannot infer nside: no mask arrays found. "
                "Pass nside= explicitly or provide at least one mask as an ndarray."
            )

    npix = hp.nside2npix(nside)

    # --- Load pickle data (needed for both mask derivation and field building) ---
    gc_data = gc_bin_key = None
    wl_data = wl_bin_key = None
    if gc_src is not None:
        gc_data = np.load(gc_src["map"], allow_pickle=True)
        gc_bin_key = next(k for k in gc_data if k.endswith("bin"))
    if wl_src is not None:
        wl_data = np.load(wl_src["map"], allow_pickle=True)
        wl_bin_key = next(k for k in wl_data if k.endswith("bin"))

    # --- Collect raw (unapodized) masks ---
    raw_masks = {}
    if gc_data is not None:
        for i, n in enumerate(gc_data[gc_bin_key].keys()):
            fk = f"g{i}"
            mask_val = gc_src["masks"].get(fk) if gc_src["masks"] else None
            if mask_val is not None:
                raw_masks[fk] = _load_array(mask_val, nside)
            else:
                m = np.zeros(npix, dtype=np.float64)
                m[gc_data[gc_bin_key][n]["ipix"]] = 1.0
                raw_masks[fk] = m

    if wl_data is not None:
        for i, n in enumerate(wl_data[wl_bin_key].keys()):
            fk = f"l{i}"
            mask_val = wl_src["masks"].get(fk) if wl_src["masks"] else None
            if mask_val is not None:
                raw_masks[fk] = _load_array(mask_val, nside)
            else:
                m = np.zeros(npix, dtype=np.float64)
                m[wl_data[wl_bin_key][n]["ipix"]] = 1.0
                raw_masks[fk] = m

    if kappa is not None:
        kappa_cfg = kappa if isinstance(kappa, dict) else {"map": kappa}
        if kappa_cfg.get("mask") is not None:
            raw_masks["kappa"] = _load_array(kappa_cfg["mask"], nside)

    if y is not None:
        y_cfg = y if isinstance(y, dict) else {"map": y}
        if y_cfg.get("mask") is not None:
            raw_masks["y"] = _load_array(y_cfg["mask"], nside)

    # --- Apodize ---
    if common_mask:
        # TODO: common_mask binarizes all inputs before intersection.
        # For weighted (non-binary) masks, this discards weight information.
        # Consider whether weighted intersection is needed.
        shared_mask = make_mask(*raw_masks.values(), aposcale=aposcale)
        apo_masks = {k: shared_mask for k in raw_masks}
    else:
        apo_masks = {k: make_mask(m, aposcale=aposcale) for k, m in raw_masks.items()}

    # --- Build fields ---
    maps_dict = {}

    if kappa is not None and "kappa" in apo_masks:
        kappa_cfg = kappa if isinstance(kappa, dict) else {"map": kappa}
        maps_dict["kappa"] = _build_external_field(
            {"map": _load_array(kappa_cfg["map"], nside)},
            apo_masks["kappa"], lmax, lmax_mask, n_iter,
        )

    if y is not None and "y" in apo_masks:
        y_cfg = y if isinstance(y, dict) else {"map": y}
        y_entry = {"map": _load_array(y_cfg["map"], nside)}
        if y_cfg.get("beam") is not None:
            y_entry["beam"] = y_cfg["beam"]
        maps_dict["y"] = _build_external_field(
            y_entry, apo_masks["y"], lmax, lmax_mask, n_iter,
        )

    if gc_data is not None:
        for i, n in enumerate(gc_data[gc_bin_key].keys()):
            fk = f"g{i}"
            delta = np.zeros(npix)
            delta[gc_data[gc_bin_key][n]["ipix"]] = gc_data[gc_bin_key][n]["map"]
            maps_dict[fk] = {
                "field": make_field(delta, apo_masks[fk], spin=0,
                                    lmax=lmax, lmax_mask=lmax_mask, n_iter=n_iter),
                "nl_uncoupled": np.full((1, lmax + 1),
                                        float(gc_data[gc_bin_key][n]["noise_cl"])),
            }

    if wl_data is not None:
        for i, n in enumerate(wl_data[wl_bin_key].keys()):
            fk = f"l{i}"
            bin_data = wl_data[wl_bin_key][n]
            e1 = np.zeros(npix)
            e2 = np.zeros(npix)
            e1[bin_data["ipix"]] = bin_data["e1"]
            e2[bin_data["ipix"]] = bin_data["e2"]
            pkl_pol_conv = bin_data.get("pol_conv", pol_conv)
            noise_scalar = float(bin_data["noise_cl"])
            nl_4 = np.zeros((4, lmax + 1))
            nl_4[0] = noise_scalar  # EE
            nl_4[3] = noise_scalar  # BB (equal per component)
            maps_dict[fk] = {
                "field": make_field([e1, e2], apo_masks[fk], spin=2,
                                    pol_conv=pkl_pol_conv, lmax=lmax,
                                    lmax_mask=lmax_mask, n_iter=n_iter),
                "nl_uncoupled": nl_4,
            }

    if not maps_dict:
        raise ValueError(
            "No fields to correlate — provide at least one of gc, wl, kappa, y."
        )

    # --- Output filename ---
    if outfile is not None:
        cls_fname = os.path.join(outdir, outfile)
    else:
        parts = []
        if gc_src is not None:
            parts.append(os.path.splitext(os.path.basename(gc_src["map"]))[0])
        if wl_src is not None:
            parts.append(os.path.splitext(os.path.basename(wl_src["map"]))[0])
        if kappa is not None:
            parts.append("kappa")
        if y is not None:
            parts.append("y")
        cls_fname = os.path.join(outdir, "cls_" + "_".join(parts) + ".pkl")

    # --- Compute spectra ---
    print("Starting computation of spectra...")
    start = time.perf_counter()
    cls = get_spectra(
        maps_dict, b, skipped_pairs,
        common_mask=common_mask,
        wksp_cache=wksp_cache,
        treat_bb_as_noise=treat_bb_as_noise,
    )
    t_spectra = time.perf_counter() - start

    leff = b.get_effective_ells()
    pickle_spectra(cls, {}, {}, {}, leff, fname=cls_fname)
    print(f"Spectra computed in {t_spectra / 60:.2f} min -> {cls_fname}")

    if compute_cov:
        print("Starting covariance computation...")
        start_cov = time.perf_counter()
        cov_matrix, cov_all = get_full_cov(
            maps_dict, cls, b, skipped_pairs,
            common_mask=common_mask,
            wksp_cache=wksp_cache,
        )
        t_cov = time.perf_counter() - start_cov
        print(f"Covariance computed in {t_cov / 60:.2f} min")
        pickle_spectra(cls, cov_matrix, cov_all, {}, leff, fname=cls_fname)

    print(f"Total runtime: {(time.perf_counter() - start) / 60:.2f} min")
    return cls_fname


# ===========================================================================
# Private helpers
# ===========================================================================

def _load_array(arr_or_path, nside=None):
    """Load a HEALPix map array from a file path, or return as-is.

    Parameters
    ----------
    arr_or_path : array_like, str, or None
        An array, a file path string, or None.
    nside : int or None
        Target nside for downgrade when loading from file.

    Returns
    -------
    ndarray or None
    """
    if arr_or_path is None:
        return None
    if isinstance(arr_or_path, str):
        return reader.read_partial_map(arr_or_path, nside)
    return np.asarray(arr_or_path, dtype=np.float64)


def _resolve_source(src, name):
    """Normalise a gc/wl source argument to a dict or None.

    Parameters
    ----------
    src : None, str, or dict
        None        → field absent.
        str         → pkl path, no explicit masks (binary footprint derived
                      from pixel indices at load time).
        dict        → ``{'map': path_or_data, 'masks': per_bin_dict_or_None}``.

    Returns
    -------
    dict ``{'map': ..., 'masks': dict_or_None}`` or None
    """
    if src is None:
        return None
    if isinstance(src, str):
        return {"map": src, "masks": None}
    if isinstance(src, dict):
        if "map" not in src:
            raise ValueError(f"{name}: dict must contain a 'map' key")
        return {"map": src["map"], "masks": src.get("masks")}
    raise TypeError(f"{name} must be str, dict, or None; got {type(src).__name__}")


def _infer_nside_from_masks(*mask_dicts):
    """Infer HEALPix nside from the first non-None array value in any mask dict.

    Parameters
    ----------
    *mask_dicts : dict
        Each entry maps field keys to arrays or paths. Paths are skipped.

    Returns
    -------
    int or None
    """
    for mdict in mask_dicts:
        if mdict is None:
            continue
        for v in mdict.values():
            if v is not None and not isinstance(v, str):
                arr = np.asarray(v)
                if arr.ndim == 1 and arr.size > 0:
                    try:
                        return hp.npix2nside(arr.size)
                    except (ValueError, RuntimeError):
                        pass
    return None


def _build_external_field(cfg, mask, lmax, lmax_mask, n_iter):
    """Build a spin-0 NmtField maps-dict entry for CMB kappa or tSZ-y.

    Parameters
    ----------
    cfg : dict
        Must contain "map". Optionally "beam" (tSZ-y). If "beam" is present,
        nl_uncoupled defaults to zeros (treated as noiseless at map level).
    mask : ndarray
        Pre-built apodized mask (caller is responsible for mask construction
        so it works for both per-field and common-mask paths).
    lmax, lmax_mask, n_iter : int

    Returns
    -------
    dict with keys "field" and "nl_uncoupled", ready for get_spectra / get_full_cov.
    """
    kw = dict(spin=0, lmax=lmax, lmax_mask=lmax_mask, n_iter=n_iter)
    if "beam" in cfg:
        kw["beam"] = cfg["beam"]
    nl = np.zeros((1, lmax + 1)) if "beam" in cfg else None
    return {"field": make_field(cfg["map"], mask, **kw), "nl_uncoupled": nl}
