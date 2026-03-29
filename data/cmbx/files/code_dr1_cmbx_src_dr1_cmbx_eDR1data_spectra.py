"""Unified NaMaster power spectrum estimation for dr1_cmbx.

Layers:
  1. Primitives — mask, field, binning construction
  2. Single-spectrum — `get_spectrum` with workspace caching and optional Knox variance
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
from pathlib import Path

import numpy as np
import healpy as hp
import pymaster as nmt
from tqdm import tqdm
import joblib

import dr1_cmbx.eDR1data.reader as reader
from dr1_cmbx.eDR1data.maps import sparse_to_dense

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
        Weight maps to multiply in before apodization. When calling via
        ``map2cls``, pass the ``weights`` dict there; it is forwarded
        per-field to this function.

    Returns
    -------
    mask : ndarray
    """
    if len(maps) == 0:
        raise ValueError("At least one map is required")

    mask = np.ones_like(maps[0], dtype=np.float64)
    for m in maps:
        mask *= (np.asarray(m, dtype=np.float64) > 0).astype(np.float64)

    if weights is not None:
        if not isinstance(weights, (list, tuple)):
            weights = [weights]
        for w in weights:
            w = np.asarray(w, dtype=np.float64)
            if w.shape != mask.shape:
                raise ValueError(f"Weight shape {w.shape} does not match mask shape {mask.shape}")
            mask *= w

    if aposcale is not None and aposcale > 0:
        mask = nmt.mask_apodization(mask, aposcale, "C2")

    return mask


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
               lmax=3000, lmax_mask=None, n_iter=1,
               beam=None, templates=None, purify_e=True, purify_b=True):
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
    purify_e, purify_b : bool
        If True, enable NaMaster E/B-mode purification for spin-2 fields.

    Returns
    -------
    field : nmt.NmtField
    """
    if lmax_mask is None:
        lmax_mask = lmax

    kw = dict(n_iter=n_iter, lmax=lmax, lmax_mask=lmax_mask)
    if beam is not None:
        kw["beam"] = beam
    if templates is not None:
        kw["templates"] = templates

    if spin == 0:
        return nmt.NmtField(mask, [np.asarray(maps)], **kw)

    elif spin == 2:
        e1, e2 = _apply_pol_conv(maps[0], maps[1], pol_conv)
        kw["purify_e"] = purify_e
        kw["purify_b"] = purify_b
        return nmt.NmtField(mask, [e1, e2], spin=2, **kw)

    else:
        raise ValueError(f"spin must be 0 or 2, got {spin}")


def make_catalog_field(
    pos, weights, values, lmax, *, spin=0, pol_conv="COSMO", templates=None,
    purify_e=True, purify_b=True
):
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
    purify_e, purify_b : bool
        If True, enable NaMaster E/B-mode purification for spin-2 fields.

    Returns
    -------
    field : nmt.NmtFieldCatalog
    """
    if spin == 0:
        return nmt.NmtFieldCatalog(pos, weights, [np.asarray(values)], lmax, spin=0, lonlat=True, templates=templates)

    elif spin == 2:
        e1, e2 = _apply_pol_conv(values[0], values[1], pol_conv)
        kw = {}
        if purify_e:
            kw["purify_e"] = True
        if purify_b:
            kw["purify_b"] = True
        while True:
            try:
                return nmt.NmtFieldCatalog(
                    pos, weights, [e1, e2], lmax, spin=2, lonlat=True, templates=templates, **kw
                )
            except TypeError as exc:
                if "purify_e" in kw and "purify_e" in str(exc):
                    warnings.warn(
                        "NmtFieldCatalog does not support purify_e in this NaMaster "
                        "build; proceeding without catalog E purification.",
                        stacklevel=2,
                    )
                    kw.pop("purify_e", None)
                    continue
                if "purify_b" in kw and "purify_b" in str(exc):
                    warnings.warn(
                        "NmtFieldCatalog does not support purify_b in this NaMaster "
                        "build; proceeding without catalog B purification.",
                        stacklevel=2,
                    )
                    kw.pop("purify_b", None)
                    continue
                raise

    else:
        raise ValueError(f"spin must be 0 or 2, got {spin}")


def make_bins(lmin, lmax, nells=None, *, nlb=None, spacing="log"):
    """Create NaMaster binning.

    Parameters
    ----------
    lmin : int
        Minimum multipole (lower edge of first bin).
    lmax : int
        Maximum multipole.
    nells : int or None
        Number of bins. Required for log/sqrt spacing, and for linear
        spacing when ``nlb`` is not given.
    nlb : int or None
        Bandpower width for linear spacing via
        ``nmt.NmtBin.from_lmax_linear``. Only used when
        ``spacing="linear"``. If provided, ``nells`` is ignored.
    spacing : {"log", "linear", "sqrt"}
        Bin spacing. Default is log (geomspace).

    Returns
    -------
    bins : nmt.NmtBin
    bpw_edges : ndarray
        Bin edges.
    """
    if spacing not in ("log", "linear", "sqrt"):
        raise ValueError(f"spacing must be 'log', 'linear', or 'sqrt', got '{spacing}'")

    # Linear with nlb: use NaMaster's from_lmax_linear directly
    if spacing == "linear" and nlb is not None:
        bins = nmt.NmtBin.from_lmax_linear(lmax, nlb)
        # Reconstruct edges for the return value
        nbands = bins.get_n_bands()
        edges = np.empty(nbands + 1, dtype=int)
        for i in range(nbands):
            ell_list = bins.get_ell_list(i)
            edges[i] = ell_list[0]
        edges[-1] = bins.get_ell_list(nbands - 1)[-1]
        return bins, edges

    # All other cases require nells
    if nells is None:
        raise ValueError(
            "nells is required for log/sqrt spacing, and for linear "
            "spacing when nlb is not given."
        )

    if spacing == "log":
        bpw_edges = np.unique(np.geomspace(lmin, lmax, nells + 1).astype(int))
    elif spacing == "linear":
        bpw_edges = np.unique(np.linspace(lmin, lmax, nells + 1).astype(int))
    elif spacing == "sqrt":
        bpw_edges = np.unique(
            (np.linspace(np.sqrt(lmin), np.sqrt(lmax), nells + 1) ** 2).astype(int)
        )

    # Safety: np.unique may collapse edges (e.g. very narrow range)
    if len(bpw_edges) < 2:
        raise ValueError(
            f"Bin edges collapsed to {len(bpw_edges)} unique value(s) "
            f"for lmin={lmin}, lmax={lmax}, nells={nells}, spacing='{spacing}'. "
            "Increase the ell range or decrease nells."
        )

    # from_edges uses half-open [ell_ini, ell_end); extend last edge to include lmax.
    ell_ini = bpw_edges[:-1]
    ell_end = bpw_edges[1:].copy()
    ell_end[-1] = lmax + 1
    bins = nmt.NmtBin.from_edges(ell_ini, ell_end)
    return bins, bpw_edges


# ===========================================================================
# Layer 2: Single-spectrum computation (with optional covariance)
# ===========================================================================

def _field_template_hash(f):
    """Return a stable hash of a field's NaMaster templates (or None)."""
    try:
        t = f.templates
    except AttributeError:
        return None
    if t is None:
        return None
    return joblib.hash(t)


def _hash_fields(f1, f2):
    """Hash two NaMaster fields by their mask alms, spins and templates.

    The key is symmetric so (f1, f2) and (f2, f1) map to the same cache file.

    Returns None for catalog fields (``NmtFieldCatalog``), which do not
    support ``get_mask()``. Callers should skip file caching in that case.

    Uses ``get_mask()`` (pixel space) rather than ``get_mask_alms()`` to avoid
    triggering an unnecessary SHT. The pixel mask uniquely determines the
    workspace.
    """
    if isinstance(f1, nmt.NmtFieldCatalog) or isinstance(f2, nmt.NmtFieldCatalog):
        return None

    k1 = joblib.hash([f1.get_mask(), f1.spin, _field_template_hash(f1)])
    k2 = joblib.hash([f2.get_mask(), f2.spin, _field_template_hash(f2)])
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
    method=None,
    include_knox=False,
    field1_for_cov=None,
    field2_for_cov=None,
    fiducial_spectra=None,
):
    """Compute a single decoupled power spectrum with workspace caching 
    and optional Knox variance.

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
    method : {None, "namaster", "knox"}
        Covariance method for this spectrum's diagonal block.
        None: no covariance.
        "namaster": NaMaster Gaussian covariance.
        "knox": Knox formula variance.
    field1_for_cov, field2_for_cov : nmt.NmtField, optional
        Auxiliary map-based fields for covariance when f1/f2 are catalog-based
        (NmtFieldCatalog cannot expose its mask). Only used with
        ``method="namaster"``.
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
        w        — NmtWorkspace
        cov      — covariance matrix (nbins × nbins), or None.
                   NaMaster: full Gaussian block. Knox: diagonal matrix np.diag(knox_var).
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
            "treat_bb_as_noise is deprecated. Noise should be estimated from "
            "catalogue properties (n_eff, sigma_e) and subtracted via "
            "nl_uncoupled, not approximated from the B-mode spectrum.",
            DeprecationWarning,
            stacklevel=2,
        )
        cl[0] = cl[0] - cl[3]

    bpws = wksp.get_bandpower_windows()

    result = {
        "cl": cl,
        "nl": nl_decoupled,
        "bpws": bpws,
        "w": wksp,
        "cov": None,
        "knox": None,
    }

    if method == "knox" or include_knox:
        knox = compute_knox_variance(f1, f2, bins)
        result["knox"] = knox
        if method == "knox":
            result["cov"] = np.diag(knox["knox_var"])

    if method == "namaster":
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

        cov, _, _ = compute_gaussian_covariance(
            f1_cov, f2_cov, f1_cov, f2_cov, bins,
            wksp_ab=wksp_cov, wksp_cd=wksp_cov,
            fiducial_spectra=fiducial_spectra,
            wksp_cache=wksp_cache,
            cov_hash_masks=cov_hash_masks,
        )
        result["cov"] = cov

    elif method is not None:
        raise ValueError(f"method must be None, 'namaster', or 'knox', got '{method}'")

    return result


# ===========================================================================
# Layer 3: Covariance
# ===========================================================================

def _get_field_mask_and_fsky(field):
    """Return (mask, fsky) for *field*, or (None, None) for catalog fields.

    NmtFieldCatalog lacks get_mask(); in that case both values are None.
    fsky is defined as <w^2> (Hivon+ 2002, Eq. 17 normalisation).
    """
    try:
        m = field.get_mask()
        return m, float(np.mean(m * m))
    except (ValueError, AttributeError):
        return None, None


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
    m1, fsky_11 = _get_field_mask_and_fsky(field1)
    m2, fsky_22 = _get_field_mask_and_fsky(field2)

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

    # Knox variance is computed only for the E/EE component.
    # For spin-2×spin-2, this corresponds to the EE spectrum (index 0).
    # BB, EB, BE are intentionally ignored.
    if cl_auto1_binned.shape[0] > 1:
        warnings.warn(
            "compute_knox_variance: Knox variance is computed only for the "
            "E/EE component (index 0). Other spin-2 components (BB, EB, BE) "
            "are ignored.",
            stacklevel=2,
        )

    # Use the first spectral component:
    # - spin-0 × spin-0 → scalar Cl
    # - spin-2 × spin-0 → E-mode cross
    # - spin-2 × spin-2 → EE auto-spectrum
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


def _lookup_block_fiducial(fiducial_spectra, nm1, nm2, nm3, nm4):
    """Build the per-block fiducial dict for compute_gaussian_covariance.

    Looks up the four cross-spectra needed for Cov(C^{AB}, C^{CD}),
    accepting either key order (symmetry fallback).
    """
    def _get(a, b):
        if (a, b) in fiducial_spectra:
            return fiducial_spectra[(a, b)]
        if (b, a) in fiducial_spectra:
            return fiducial_spectra[(b, a)]
        raise KeyError(f"fiducial_spectra missing ({a!r}, {b!r}) or ({b!r}, {a!r})")

    return {
        "11": _get(nm1, nm3),  # C(A, C)
        "12": _get(nm1, nm4),  # C(A, D)
        "21": _get(nm2, nm3),  # C(B, C)
        "22": _get(nm2, nm4),  # C(B, D)
    }


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
                common_mask=False, wksp_cache=None, treat_bb_as_noise=False,
                method=None, include_knox=False, fiducial_spectra=None):
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
    method : {None, "namaster", "knox"}
        Covariance method for per-spectrum diagonal blocks.
        Passed through to ``get_spectrum``.
    fiducial_spectra : dict, optional
        Theory+noise spectra for Gaussian covariance.  Either a flat
        ``{"11": ..., "12": ..., "22": ...}`` dict (passed through as-is),
        or a ``{(name, name): array}`` dict (per-pair block built via
        ``_lookup_block_fiducial``).

    Returns
    -------
    cls : dict
    """
    map_names = list(maps.keys())
    cls = {}
    shared_wksp = None
    pair_keyed_fid = (
        fiducial_spectra is not None
        and isinstance(next(iter(fiducial_spectra)), tuple)
    )

    for _, _, _, nm1, nm2, cl_name in cl_iter(map_names, skipped_pairs):
        fg1 = maps[nm1]["field"]
        fg2 = maps[nm2]["field"]
        pnl = maps[nm1].get("nl_uncoupled") if nm1 == nm2 else None

        if fiducial_spectra is None:
            per_spectrum_fid = None
        elif pair_keyed_fid:
            per_spectrum_fid = _lookup_block_fiducial(
                fiducial_spectra, nm1, nm2, nm1, nm2)
        else:
            per_spectrum_fid = fiducial_spectra

        wksp = shared_wksp if (common_mask and shared_wksp is not None) else None
        r = get_spectrum(fg1, fg2, bins, nl=pnl,
                         wksp=wksp, wksp_cache=wksp_cache,
                         treat_bb_as_noise=treat_bb_as_noise,
                         method=method,
                         include_knox=include_knox,
                         fiducial_spectra=per_spectrum_fid)
        if common_mask and shared_wksp is None:
            shared_wksp = r["w"]

        cls[cl_name] = {
            "cl": r["cl"],
            "nl": r["nl"],
            "bpws": r["bpws"],
            "w": r["w"],
            "cov": r["cov"],
            "knox": r["knox"],
        }

    return cls


def get_full_cov(maps, cls, bins, skipped_pairs, *,
                 common_mask=False, fiducial_spectra=None, wksp_cache=None,
                 diagonal_only=False):
    """Build covariance matrix for all auto- and cross-spectra.

    Parameters
    ----------
    maps : dict
    cls : dict
        Output from get_spectra.
    bins : nmt.NmtBin
    skipped_pairs : list
    common_mask : bool
    fiducial_spectra : dict, optional
        Theory+noise spectra keyed by ``(field_name, field_name)`` tuples,
        e.g. ``{("kappa", "kappa"): cl_kk, ("l0", "kappa"): cl_ek, ...}``.
        Each off-diagonal block Cov(AB, CD) receives a per-block dict
        built via ``_lookup_block_fiducial``.
    wksp_cache : str or Path, optional
    diagonal_only : bool
        If True, compute only diagonal blocks Cov(AB, AB) and skip
        off-diagonal blocks.  Faster for error bars without the full
        covariance matrix needed for likelihood.  When True, ``cov_all``
        is returned as None.

    Returns
    -------
    cov : list of lists of ndarray
        Block covariance. Off-diagonal entries are None when diagonal_only.
    cov_all : ndarray or None
        Full covariance, shape (n_cls*n_ell, n_cls*n_ell).
        None when ``diagonal_only=True``.
    """
    map_names = list(maps.keys())
    n_cls = len(cls.keys())
    leff = bins.get_effective_ells()
    n_ell = len(leff)

    cov = []
    for _ in cl_iter(map_names, skipped_pairs):
        cov.append([None] * n_cls)

    pairs = list(cl_iter(map_names, skipped_pairs))
    if diagonal_only:
        total = len(pairs)
    else:
        total = len(pairs) ** 2
    pbar = tqdm(total=total, desc="Computing covariance blocks", ncols=100)

    shared_cov_wksp = None

    for icl1, _, _, nm1, nm2, cl_name12 in cl_iter(map_names, skipped_pairs):
        f1 = maps[nm1]["field"]
        f2 = maps[nm2]["field"]
        w12 = cls[cl_name12]["w"]

        for icl2, _, _, nm3, nm4, cl_name34 in cl_iter(map_names, skipped_pairs):
            if diagonal_only and icl1 != icl2:
                continue

            f3 = maps[nm3]["field"]
            f4 = maps[nm4]["field"]
            w34 = cls[cl_name34]["w"]

            if icl2 < icl1:
                cv = cov[icl2][icl1].T
            else:
                block_fid = (
                    _lookup_block_fiducial(fiducial_spectra, nm1, nm2, nm3, nm4)
                    if fiducial_spectra is not None
                    else None
                )

                cv, _, cov_wksp_out = compute_gaussian_covariance(
                    f1, f2, f3, f4, bins,
                    wksp_ab=w12, wksp_cd=w34,
                    cov_wksp=shared_cov_wksp if common_mask else None,
                    fiducial_spectra=block_fid,
                    wksp_cache=wksp_cache,
                )

                if common_mask and shared_cov_wksp is None:
                    shared_cov_wksp = cov_wksp_out

            if icl1 == icl2:
                cls[cl_name12]["cov"] = cv
            cov[icl1][icl2] = cv
            pbar.update(1)

    pbar.close()

    if diagonal_only:
        return cov, None

    cov_blocks = np.array(cov)
    cov_all = np.transpose(cov_blocks, axes=[0, 2, 1, 3]).reshape(
        [n_cls * n_ell, n_cls * n_ell]
    )
    return cov, cov_all


# ===========================================================================
# I/O
# ===========================================================================

def pickle_spectra(cls, cov, cov_all, dndz, leff, fname="", keep_full_components=False):
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
            if k_j in ["w"]:
                continue
            data[k_i][k_j] = cls[k_i][k_j]

    for k in data.keys():
        data[k]["leff"] = leff
        if keep_full_components:
            continue
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
            # XXX: keeping only the EE (or Eκ) bandpower window. For spin-2×spin-2
            # the full bpws has shape (4, nbins, 4, lmax+1) — revisit if needed.
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

def map2cls(
    gc=None,
    wl=None,
    kappa=None,
    y=None,
    skipped_pairs=(),
    extra_masks=None,
    extra_weights=None,
    apo_masks=None,
    common_mask=False,
    aposcale=0.15,
    n_iter=1,
    nside=None,
    lmin=2,
    lmax=3000,
    lmax_mask=None,
    ell_binning={"spacing":'linear', "delta_ell": 100},
    pol_conv="COSMO",
    covariance=None,
    diagonal_cov_only=False,
    include_knox=False,
    fiducial_spectra=None,
    treat_bb_as_noise=False,
    outfile=None,
    outdir="./data/cls",
    wksp_cache=None,
    dndz=None,
    templates=None,
    purify_e=True,
    purify_b=True,
    keep_full_components=False,
):
    """Compute angular power spectra from GC, WL, CMB kappa, and/or tSZ-y maps.

    Primary high-level entry point. Galaxy clustering and weak lensing fields
    are loaded from pickle files produced by ``gc_cat2map`` / ``wl_cat2map``.

    Parameters
    ----------
    gc : str, dict, or None
        GC source: path to a pickle (``gc_cat2map`` output), an
        already-loaded pickle dict, or None.
    wl : str, dict, or None
        WL source: path to a pickle (``wl_cat2map`` output), an
        already-loaded pickle dict, or None.
    kappa : None or dict
        CMB lensing field: ``{'map': arr_or_path, 'mask': arr_or_path}``.
        If absent, no CMB kappa field.
    y : None or dict
        Compton-y field: ``{'map': arr_or_path, 'mask': arr_or_path,
        'beam': arr_or_None}``. If absent, no tSZ-y field.
    skipped_pairs : list of tuple of str
    extra_masks : dict or None
        Optional per-field mask overrides. Keys can include:
            -"gc": dict of per-bin gc masks {"g0": arr, "g1": arr, ...},
            -"wl": dict of per-bin wl masks {"l0": arr, "l1": arr, ...},
            -"footprint": arr, binary footprint,
            -"effcov": arr, effective coverage,
            -"visibility": dict of {bin_index: map}, used with effcov for GC bins
        Rules:
            - Multiple masks for the same field are combined by intersection. 
            If only one mask is provided, it is used as the default.
            - If none of these are provided, the pickle's "mask" key is used. (see gc_cat2map).
            - Visibility maps are applied as weights to GC masks when effcov is present.
            - Per-field masks and visibility cannot be used with ``common_mask=True``
    extra_weights : dict or None
        Per-field weight maps to multiply into masks before apodization.
        Structured weights applied as:
            {"gc": {"g0": weight_g0, "g1": weight_g1, ...},
             "wl": {"l0": weight_wl0, "l1": weight_wl1, ...}}
        Weights are applied to the mask before apodization.
        Missing keys are treated as no weighting for that field.
        Cannot be combined with ``common_mask=True`` (raises ValueError).
    apo_masks : dict or None
        Prebuilt apodized masks keyed by final field name (for example
        ``{"l0": arr, "kappa": arr}``). These masks are used verbatim and
        are not re-apodized. This is useful when a pipeline materializes
        official pair masks upstream and still wants ``map2cls`` to own the
        spectrum and covariance computation.
    common_mask : bool
        If True, build one shared mask from the intersection of all fields.
    aposcale : float
        C2 apodization scale in degrees.
    n_iter : int
    nside : int or None
        HEALPix NSIDE. If None, inferred from the first mask array found.
        Must be provided explicitly when all masks are file paths.
    lmin : int
    lmax : int
    lmax_mask : int or None
    ell_binning : dict or nmt.NmtBin
        Multipole binning specification.
        - If an ``nmt.NmtBin`` object is provided, it is used directly.
        - If a dict is provided, it must contain either:
            {"nbins": int, "spacing": "linear"|"log"|"sqrt"}
            specifying the number of bins, or
            {"delta_ell": int}
            specifying a linear bin width delta_ell.
    pol_conv : {"COSMO", "IAU", "EUCLID"}
        Sign convention for WL shear maps.
    covariance : {None, "namaster", "knox"}
        Covariance estimation method.
        None: no covariance.
        "namaster": NaMaster Gaussian covariance (full block matrix, or
        diagonal only if ``diagonal_cov_only=True``).
        "knox": Knox formula variance (always per-spectrum diagonal).
    diagonal_cov_only : bool
        When ``covariance="namaster"``, compute only the diagonal blocks
        Cov(AB, AB) instead of the full block matrix. Faster for error
        bars; the full matrix is needed for likelihood inference.
        Ignored when ``covariance="knox"`` (always diagonal).
    include_knox : bool
        If True, compute and store Knox diagnostics alongside the requested
        covariance method. The diagonal Knox covariance itself remains the
        primary covariance only when ``covariance="knox"``.
    fiducial_spectra : dict, optional
        Theory+noise spectra for Gaussian covariance.  Keyed by
        ``(field_name, field_name)`` tuples for multi-field cases (used
        by both ``get_spectra`` and ``get_full_cov``), or a flat
        ``{"11", "12", "22"}`` dict for single-spectrum cases.
    treat_bb_as_noise : bool
        Deprecated. Passed to ``get_spectra``.
    outfile : str, optional
        Output filename. If None, auto-generated from source names.
    outdir : str
    wksp_cache : str or Path, optional
    dndz : dict or None
        Redshift distributions to store in the output pickle. Defaults
        to an empty dict if None.
    templates : array or None
        Per-field templates. Keys are field names ('gc', 'wl', 'kappa', 'y').
        Each value is an array or list of arrays passed to NaMaster.
        Each array containing a set of contaminant templates for each field. 
        This array should have shape [ntemp,nmap,...], where ntemp is the 
        number of templates, nmap should be 1 for spin-0 fields and 2 otherwise. 
        The other dimensions should be [npix] for HEALPix maps or [ny,nx] for 
        maps with rectangular pixels. The best-fit contribution from each contaminant 
        is automatically removed from the maps unless templates=None.
    purify_e, purify_b : bool
        If True, enable NaMaster E/B-mode purification for all spin-2 fields.
        The same purified fields are used for both spectrum estimation and
        NaMaster Gaussian covariance.
    keep_full_components : bool
        If True, keep the full spectral-component arrays in the serialized
        output instead of extracting only the first component.

    Returns
    -------
    cls_fname : str
        Path to the saved pickle file.
    """
    os.makedirs(outdir, exist_ok=True)
    if lmax_mask is None:
        lmax_mask = lmax

    # Binning
    if isinstance(ell_binning, nmt.NmtBin):
        b = ell_binning
    elif isinstance(ell_binning, dict):
        nbins = ell_binning.get("nbins")
        delta_ell = ell_binning.get("delta_ell")
        spacing = ell_binning.get("spacing")
        if (nbins is None) == (delta_ell is None):
            raise ValueError(
                "ell_binning dict must contain exactly one of 'nbins' or 'delta_ell'")
        if delta_ell is not None:
            b, _ = make_bins(lmin, lmax, nlb=delta_ell, spacing="linear")
        else:
            b, _ = make_bins(lmin, lmax, nells=nbins, spacing=spacing)
    else:
        raise TypeError(
        "ell_binning must be either an nmt.NmtBin or a dict with keys "
        "{'nbins','delta_ell','spacing'}")

    # Normalize external field args to dicts once
    kappa = kappa if kappa is None or isinstance(kappa, dict) else {"map": kappa}
    y = y if y is None or isinstance(y, dict) else {"map": y}

    # --- Load pickle data ---
    gc = np.load(gc, allow_pickle=True) if isinstance(gc, str) else gc
    wl = np.load(wl, allow_pickle=True) if isinstance(wl, str) else wl

    # --- Infer nside --- 
    if nside is None:
        nside = _infer_nside(gc, wl, extra_masks, apo_masks, kappa, y)

    npix = hp.nside2npix(nside)

    # --- Collect raw (unapodized) masks ---
    raw_masks = {}

    # --- Collect GC and WL masks ---
    for tracer, prefix, data in [("gc", "g", gc), ("wl", "l", wl)]:
        if data is None:
            continue
        
        mask_overrides = (extra_masks or {}).get(tracer, {})

        # --- Base mask (intersection of global masks) ---
        global_masks = []
        if extra_masks and tracer == "gc":
            # GC: footprint and effcov only
            for key in ["footprint", "effcov"]:
                m = extra_masks.get(key)
                if m is not None:
                    global_masks.append(np.asarray(m, dtype=np.float64))
        elif extra_masks and tracer == "wl":
            m = extra_masks.get("footprint")
            if m is not None:
                global_masks.append(np.asarray(m, dtype=np.float64))

        if global_masks:
            base_mask = np.prod(global_masks, axis=0)
        else:
            # fallback: footprint from pickle
            fp = _get_pickle_footprint(data)
            base_mask = np.asarray(fp, dtype=np.float64) if fp is not None else None

        # --- Loop over bins ---
        for i, bd in enumerate(data.values()):
            fk = f"{prefix}{i}"

            # 1) Bin-specific override or bin mask from pickle
            if fk in mask_overrides:
                mask = np.asarray(mask_overrides[fk], dtype=np.float64)
            else:
                mask = sparse_to_dense(bd["ipix"], 1.0, bd["nside"])
            # apply pickle weights if present
            if "weights" in bd:
                mask *= sparse_to_dense(bd["ipix"], bd["weights"], bd["nside"])

            # 2) Multiply by base mask
            if base_mask is not None:
                if base_mask.shape != mask.shape:
                    raise ValueError(f"Base mask shape {base_mask.shape} != bin mask {mask.shape}")
                mask *= base_mask

            # 3) Apply GC visibility if effcov is provided (per bin, not for common_mask)
            if tracer == "gc" and extra_masks and "effcov" in extra_masks and not common_mask:
                vis_dict = extra_masks.get("visibility", {})
                vis_map = vis_dict.get(i+1)
                if vis_map is not None:
                    if vis_map.shape != mask.shape:
                        raise ValueError(f"Visibility shape {vis_map.shape} != mask {mask.shape}")
                    mask *= (1.0 + vis_map)

            raw_masks[fk] = mask

    # --- Collect CMB lensing and tSZ masks ---
    if kappa is not None and kappa.get("mask") is not None:
        raw_masks["kappa"] = _load_mask(kappa["mask"], nside)

    if y is not None and y.get("mask") is not None:
        raw_masks["y"] = _load_mask(y["mask"], nside)

    # --- Apodize ---
    prebuilt_apo_masks = {}
    if apo_masks:
        if common_mask:
            raise ValueError("apo_masks cannot be combined with common_mask=True")
        for k, m in apo_masks.items():
            loaded = _load_mask(m, nside)
            if loaded is None:
                continue
            prebuilt_apo_masks[k] = np.asarray(loaded, dtype=np.float64)

    unexpected_apo = sorted(set(prebuilt_apo_masks) - set(raw_masks))
    if unexpected_apo:
        raise ValueError(
            f"apo_masks provided for unknown fields {unexpected_apo}; "
            f"available fields are {sorted(raw_masks)}"
        )

    if common_mask:
        if extra_weights:
            raise ValueError(
                "extra_weights cannot be combined with common_mask=True: per-field "
                "weighting is not defined for a shared intersection mask."
            )
        shared_mask = make_mask(*raw_masks.values(), aposcale=aposcale)
        apo_masks = {k: shared_mask for k in raw_masks}
    else:
        # continuous weights applied to mask
        apo_masks = {}
        for k, m in raw_masks.items():
            if k in prebuilt_apo_masks:
                if prebuilt_apo_masks[k].shape != np.asarray(m).shape:
                    raise ValueError(
                        f"apo_masks[{k!r}] shape {prebuilt_apo_masks[k].shape} "
                        f"!= raw mask shape {np.asarray(m).shape}"
                    )
                apo_masks[k] = prebuilt_apo_masks[k]
                continue
            if k.startswith("l"):
                field = "wl"
            elif k.startswith("g"):
                field = "gc"
            else:
                field = None  # kappa, y: no extra_weights
            w = (extra_weights or {}).get(field, {}).get(k) if field is not None else None
            apo_masks[k] = make_mask(m, aposcale=aposcale, weights=w)


    # --- Build fields ---
    maps_dict = {}

    if kappa is not None and "kappa" in apo_masks:
        maps_dict["kappa"] = _build_external_field(
            _prepare_external_field_cfg(kappa, nside),
            apo_masks["kappa"], lmax, lmax_mask, n_iter,
            purify_e=purify_e, purify_b=purify_b,
        )

    if y is not None and "y" in apo_masks:
        maps_dict["y"] = _build_external_field(
            _prepare_external_field_cfg(y, nside),
            apo_masks["y"], lmax, lmax_mask, n_iter,
            purify_e=purify_e, purify_b=purify_b,
        )

    if gc is not None:
        gc_templates = None if templates is None else templates.get("gc")
        for i, n in enumerate(gc.keys()):
            fk = f"g{i}"
            delta = np.zeros(npix)
            delta[gc[n]["ipix"]] = gc[n]["map"]
            maps_dict[fk] = {
                "field": make_field(delta, apo_masks[fk], spin=0,
                                    lmax=lmax, lmax_mask=lmax_mask, n_iter=n_iter,
                                    templates=gc_templates,
                                    purify_e=purify_e, purify_b=purify_b),
                "nl_uncoupled": np.full((1, lmax + 1),
                                        float(gc[n]["noise_cl"])),
            }

    if wl is not None:
        wl_templates = None if templates is None else templates.get("wl")
        for i, n in enumerate(wl.keys()):
            fk = f"l{i}"
            bin_data = wl[n]
            e1 = np.zeros(npix)
            e2 = np.zeros(npix)
            e1[bin_data["ipix"]] = bin_data["e1"]
            e2[bin_data["ipix"]] = bin_data["e2"]
            pkl_pol_conv = bin_data.get("pol_conv", pol_conv)
            noise_scalar = float(bin_data["noise_cl"])
            # NaMaster spin-2 x spin-2 has 4 spectral components:
            # (EE, EB, BE, BB). Shape noise is isotropic so EE=BB=noise_scalar
            # and EB=BE=0.
            nl_4 = np.zeros((4, lmax + 1))
            nl_4[0] = noise_scalar  # EE
            nl_4[3] = noise_scalar  # BB
            maps_dict[fk] = {
                "field": make_field([e1, e2], apo_masks[fk], spin=2,
                                    pol_conv=pkl_pol_conv, lmax=lmax,
                                    lmax_mask=lmax_mask, n_iter=n_iter,
                                    templates=wl_templates,
                                    purify_e=purify_e, purify_b=purify_b),
                "nl_uncoupled": nl_4,
            }

    if not maps_dict:
        raise ValueError(
            "No fields to correlate — provide at least one of gc, wl, kappa, y."
        )

    # --- Output filename ---
    if outfile is None:
        parts = [k for k, v in [("gc", gc), ("wl", wl), ("kappa", kappa), ("y", y)] if v is not None]
        outfile = "cls_" + "_".join(parts) + ".pkl"
    cls_fname = os.path.join(outdir, outfile)

    # --- Compute spectra ---
    print("Starting computation of spectra...")
    start = time.perf_counter()

    # Determine per-spectrum covariance method:
    # - Knox or diagonal NaMaster → computed inline in get_spectra
    # - Full NaMaster → computed separately in get_full_cov
    per_spectrum_method = None
    if covariance == "knox":
        per_spectrum_method = "knox"
    elif covariance == "namaster" and diagonal_cov_only:
        per_spectrum_method = "namaster"

    cls = get_spectra(
        maps_dict, b, skipped_pairs,
        common_mask=common_mask,
        wksp_cache=wksp_cache,
        treat_bb_as_noise=treat_bb_as_noise,
        method=per_spectrum_method,
        include_knox=include_knox,
        fiducial_spectra=fiducial_spectra,
    )
    t_spectra = time.perf_counter() - start

    leff = b.get_effective_ells()
    dndz_out = dndz or {}
    cov_matrix = {}
    cov_all = {}

    if covariance == "namaster" and not diagonal_cov_only:
        print(f"Spectra computed in {t_spectra / 60:.2f} min; starting covariance...")
        start_cov = time.perf_counter()
        cov_matrix, cov_all = get_full_cov(
            maps_dict, cls, b, skipped_pairs,
            common_mask=common_mask,
            fiducial_spectra=fiducial_spectra,
            wksp_cache=wksp_cache,
        )
        t_cov = time.perf_counter() - start_cov
        print(f"Covariance computed in {t_cov / 60:.2f} min")

    pickle_spectra(
        cls, cov_matrix, cov_all, dndz_out, leff,
        fname=cls_fname, keep_full_components=keep_full_components,
    )
    print(f"Total runtime: {(time.perf_counter() - start) / 60:.2f} min -> {cls_fname}")
    return cls_fname


# ===========================================================================
# Private helpers
# ===========================================================================

def _get_pickle_footprint(data):
    """Return footprint stored in a gc/wl pickle if present."""
    if data is None:
        return None

    # common pattern: stored at top-level
    if isinstance(data, dict) and "footprint" in data:
        return data["footprint"]

    # fallback: sometimes inside bins (same footprint for each bin)
    try:
        first = next(iter(data.values()))
        if isinstance(first, dict) and "footprint" in first:
            return first["footprint"]
    except Exception:
        pass

    return None

def _load_mask(mask, nside=None):
    """Load a mask from a file path or return an array as-is.

    Accepts arrays, FITS paths, or pickle/npz paths. None passes through.

    Parameters
    ----------
    mask : ndarray, str, or None
    nside : int or None
        Target NSIDE for FITS maps (downgrade if needed). Ignored for pickles.

    Returns
    -------
    ndarray or None
    """
    if mask is None or not isinstance(mask, str):
        return mask
    ext = Path(mask).suffix.lower()
    if ext == ".fits":
        return reader.read_fits(mask, nside)
    if ext in (".pkl", ".npz", ".npy"):
        return np.load(mask, allow_pickle=True)
    raise ValueError(f"Unrecognised mask file extension '{ext}': {mask}")


def _infer_nside(gc=None, wl=None, extra_masks=None, apo_masks=None, kappa=None, y=None):
    """Infer the smallest nside from all available inputs."""

    nsides = []

    # --- catalog pickles ---
    if gc:
        nsides.extend(b["nside"] for b in gc.values())
    if wl:
        nsides.extend(b["nside"] for b in wl.values())

    # --- extra masks ---
    if extra_masks:
        for key in ["footprint", "effcov"]:
            m = extra_masks.get(key)
            if isinstance(m, np.ndarray):
                nsides.append(hp.npix2nside(len(m)))

        for field in ["gc", "wl"]:
            d = extra_masks.get(field)
            if isinstance(d, dict):
                for m in d.values():
                    if isinstance(m, np.ndarray):
                        nsides.append(hp.npix2nside(len(m)))

    if apo_masks:
        for m in apo_masks.values():
            if isinstance(m, np.ndarray):
                nsides.append(hp.npix2nside(len(m)))

    # --- external maps ---
    if kappa and isinstance(kappa.get("mask"), np.ndarray):
        nsides.append(hp.npix2nside(len(kappa["mask"])))

    if y and isinstance(y.get("mask"), np.ndarray):
        nsides.append(hp.npix2nside(len(y["mask"])))

    if not nsides:
        raise ValueError("Cannot infer nside: no valid masks or catalog information found. Pass nside= explicitly")

    return min(nsides)


def _prepare_external_field_cfg(field, nside):
    """Normalise a raw kappa/y arg and load the map array.

    Parameters
    ----------
    field : str, array, or dict
        If a bare path or array, wrapped in ``{"map": field}``.
        Dicts may carry optional keys ``"mask"``, ``"beam"``, and
        ``"nl_uncoupled"``.
    nside : int

    Returns
    -------
    dict
        Contains ``"map"`` (loaded ndarray) and any optional keys copied
        from *field* (``"beam"``, ``"nl_uncoupled"``).
    """
    cfg = field if isinstance(field, dict) else {"map": field}
    raw_map = cfg["map"]
    build_cfg = {
        "map": reader.read_fits(raw_map, nside) if isinstance(raw_map, str) else raw_map
    }
    for key in ("beam", "nl_uncoupled", "templates"):
        if cfg.get(key) is not None:
            build_cfg[key] = cfg[key]
    return build_cfg


def _build_external_field(
    cfg, mask, lmax, lmax_mask, n_iter, *, purify_e=True, purify_b=True
):
    """Build a spin-0 NmtField maps-dict entry for CMB kappa or tSZ-y.

    Parameters
    ----------
    cfg : dict
        Must contain "map". Optionally "beam" (tSZ-y) and/or
        "nl_uncoupled" (noise spectrum to subtract during decoupling).
    mask : ndarray
        Pre-built apodized mask (caller is responsible for mask construction
        so it works for both per-field and common-mask paths).
    lmax, lmax_mask, n_iter : int

    Returns
    -------
    dict with keys "field" and "nl_uncoupled", ready for get_spectra / get_full_cov.
    """
    kw = dict(
        spin=0, lmax=lmax, lmax_mask=lmax_mask, n_iter=n_iter,
        purify_e=purify_e, purify_b=purify_b
    )
    if "beam" in cfg:
        kw["beam"] = cfg["beam"]
    nl = cfg.get("nl_uncoupled")
    if nl is None:
        nl = np.zeros((1, lmax + 1))
    templates = cfg.get("templates")
    return {"field": make_field(cfg["map"], mask, templates=templates, **kw), "nl_uncoupled": nl}
