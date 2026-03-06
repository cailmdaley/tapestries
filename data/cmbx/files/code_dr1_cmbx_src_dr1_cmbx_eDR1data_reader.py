"""I/O utilities for reading and writing HEALPix map products."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import healpy as hp
import numpy as np

PathLike = Union[str, Path]


def read_fits(path: PathLike, nside: int | None = None, *, field: int | None = None,
              nest: bool = False) -> np.ndarray:
    """Read a HEALPix FITS map (dense or sparse) with optional nside change.

    Detects the format automatically:

    - **Sparse** (binary table with a ``PIXEL`` column): aggregates directly
      into the target resolution without materialising the full-resolution
      array.  Downgrade only (target nside <= source nside).
    - **Dense** (standard healpy image): loaded via ``hp.read_map``.  If
      *nside* differs from the source, resampled with ``hp.ud_grade``
      (supports both downgrade and upgrade).

    Parameters
    ----------
    path : str or Path
        FITS file path.
    nside : int or None
        Target NSIDE.  If ``None`` the map is returned at its native
        resolution.
    field : int or None
        Column / field index for multi-extension dense FITS.  Ignored for
        sparse maps (which use the first value column after ``PIXEL``).
    nest : bool
        Return in NESTED ordering.  Default ``False`` (RING).

    Returns
    -------
    numpy.ndarray
        Dense HEALPix map at the requested *nside*.
    """
    path = str(path)

    # --- Try sparse first (binary table with PIXEL column) ---
    if _is_sparse_fits(path):
        return _read_sparse(path, nside, nest=nest)

    # --- Dense map ---
    kw = {"nest": nest, "dtype": np.float64}
    if field is not None:
        kw["field"] = field
    m = hp.read_map(path, **kw)

    if nside is not None:
        nside_in = hp.npix2nside(len(m))
        if nside != nside_in:
            m = hp.ud_grade(m, nside, order_in="NESTED" if nest else "RING")

    return np.asarray(m, dtype=np.float64)


def _is_sparse_fits(path: str) -> bool:
    """Return True if the FITS file contains a sparse HEALPix table."""
    import fitsio
    try:
        hdr = fitsio.read_header(path, ext=1)
        # Explicit sparse index scheme, or a PIXEL column present
        if str(hdr.get("INDXSCHM", "")).strip().upper() == "SPARSE":
            return True
        info = fitsio.FITS(path)[1].get_colnames()
        return "PIXEL" in [c.upper() for c in info]
    except Exception:
        return False


def _read_sparse(path: str, nside: int | None, *, nest: bool = False) -> np.ndarray:
    """Read a sparse FITS table and return a dense map at *nside*."""
    import fitsio

    data, header = fitsio.read(path, header=True)
    nside_in = int(header["NSIDE"])

    if nside is None:
        nside = nside_in

    if nside > nside_in:
        raise ValueError(
            f"target NSIDE={nside} exceeds source NSIDE={nside_in} in {path}; "
            "sparse maps can only be downgraded"
        )

    if nside_in % nside != 0:
        raise ValueError(
            f"target NSIDE={nside} does not evenly divide source NSIDE={nside_in}"
        )

    degrade_factor = (nside_in // nside) ** 2
    npix_out = hp.nside2npix(nside)
    out = np.zeros(npix_out, dtype=np.float64)

    ipix = np.asarray(data["PIXEL"], dtype=np.int64)

    # Use first non-PIXEL column as values
    col_names = [c for c in data.dtype.names if c.upper() != "PIXEL"]
    if not col_names:
        raise ValueError(f"Sparse map {path} has no value columns")
    weights = np.asarray(data[col_names[0]], dtype=np.float64)

    ordering = str(header.get("ORDERING", "RING")).strip().upper()
    if ordering == "RING":
        ipix = hp.ring2nest(nside_in, ipix)
    elif ordering != "NESTED":
        raise ValueError(f"Unrecognised pixel ordering '{ordering}' in {path}")

    # Aggregate source pixels into target pixels (avoids full-res array).
    ipix = ipix // degrade_factor
    if not nest:
        ipix = hp.nest2ring(nside, ipix)

    np.add.at(out, ipix, weights / degrade_factor)
    return out



def write_sparse_map(
    path: PathLike,
    *,
    nside: int,
    pixels: np.ndarray,
    columns: dict[str, np.ndarray],
    extra_header: dict | None = None,
) -> None:
    """Write a sparse HEALPix map to a FITS binary table.

    Parameters
    ----------
    path : str or Path
        Output FITS file path.
    nside : int
        HEALPix NSIDE of the map.
    pixels : array_like
        Sparse pixel indices.
    columns : dict
        Column name → values mapping (e.g. ``{"WEIGHT": weights}``).
    extra_header : dict, optional
        Additional FITS header keywords.
    """
    import fitsio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dtype = [("PIXEL", np.int64)] + [(name.upper(), np.float64) for name in columns]
    data = np.zeros(len(pixels), dtype=dtype)
    data["PIXEL"] = np.asarray(pixels, dtype=np.int64)
    for name, values in columns.items():
        data[name.upper()] = np.asarray(values, dtype=np.float64)

    header = {
        "NSIDE": int(nside),
        "ORDERING": "RING",
        "INDXSCHM": "SPARSE",
        "DATE": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra_header:
        header.update(extra_header)

    with fitsio.FITS(str(path), mode="rw", clobber=True) as hdul:
        hdul.write(data, names=[name for name, _ in dtype], header=header, extname="SPARSE_MAP")


def _load_bin_maps(data_path: str, filename_template: str, nbins: int, nside: int) -> dict:
    """Load one FITS map per tomographic bin from *data_path*.

    Parameters
    ----------
    data_path : str
        Directory containing the FITS files.
    filename_template : str
        Format string with a single positional placeholder for the bin index,
        e.g. ``"visibility_tombinid_{}.fits"``.
    nbins : int
        Number of tomographic bins.
    nside : int
        Target NSIDE resolution.

    Returns
    -------
    dict
        ``{i: map}`` for i in 1..nbins.
    """
    maps = {}
    for i in range(1, int(nbins) + 1):
        path = os.path.join(data_path, filename_template.format(i))
        maps[i] = read_fits(path, nside)
    return maps


def load_visibility_maps(nbins, nside, data_path="./data/visibility_tombinid_14_07_2025"):
    """Load visibility maps (one per tomographic bin).

    Parameters
    ----------
    nbins : int
        Number of tomographic bins.
    nside : int
        Target NSIDE resolution.
    data_path : str
        Directory containing the visibility FITS files.

    Returns
    -------
    dict
        ``{i: map}`` for i in 1..nbins.
    """
    return _load_bin_maps(data_path, "visibility_tombinid_{}.fits", nbins, nside)


def load_random_maps(nbins, nside, data_path="./data/random_tombin_14_07_2025"):
    """Load random maps (one per tomographic bin).

    Parameters
    ----------
    nbins : int
        Number of tomographic bins.
    nside : int
        Target NSIDE resolution.
    data_path : str
        Directory containing the random FITS files.

    Returns
    -------
    dict
        ``{i: map}`` for i in 1..nbins.
    """
    return _load_bin_maps(data_path, "random_tombin{}.fits", nbins, nside)


def load_effective_coverage(filename, nside, data_path="./data"):
    """Load the effective coverage map.

    Parameters
    ----------
    filename : str
        Filename of the coverage FITS file within ``data_path``.
    nside : int
        Target NSIDE resolution.
    data_path : str
        Directory containing the coverage FITS file.

    Returns
    -------
    numpy.ndarray
        Effective coverage HEALPix map.
    """
    path = os.path.join(data_path, filename)
    return read_fits(path, nside=nside)
