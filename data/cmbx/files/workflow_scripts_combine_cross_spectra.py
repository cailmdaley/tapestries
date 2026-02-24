"""Combine individual cross-spectrum NPZ files into a single SACC file.

This script takes multiple individual cross-spectra computed in parallel
and combines them into a single SACC file for downstream analysis.
"""

from pathlib import Path
import sys

import fitsio
import numpy as np
import sacc

from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log


# Snakemake always provides this
snakemake = snakemake  # type: ignore # noqa: F821


def load_npz_spectrum(npz_file: Path) -> dict:
    """Load cross-spectrum from NPZ file."""
    data = np.load(npz_file, allow_pickle=True)

    result = {
        "ells": data["ells"],
        "cls": data["cls"],
        "bpws": data["bpws"],
        "metadata": data["metadata"].item() if "metadata" in data else {},
    }
    if "cov" in data:
        result["cov"] = data["cov"]
    return result


def load_nz(nz_file: Path) -> tuple[np.ndarray, dict[int | str, np.ndarray]]:
    """Load per-bin n(z) from FITS file.

    Returns:
        z_grid: Redshift array (3000 points, 0 to 6)
        nz_per_bin: {bin_id: nz_array} for bins 1-6 and "all"
    """
    data = fitsio.read(str(nz_file), ext=1)
    z_grid = np.linspace(0, 6, data[0]["N_Z"].shape[0])

    nz_per_bin: dict[int | str, np.ndarray] = {}
    for row in data:
        nz_per_bin[int(row["BIN_ID"])] = row["N_Z"].copy()

    # "all" bin: sum per-bin n(z) (unnormalized — shape is what matters)
    nz_per_bin["all"] = sum(nz_per_bin[b] for b in sorted(nz_per_bin))

    return z_grid, nz_per_bin


def create_sacc_from_spectra(
    spectra: list[dict],
    global_metadata: dict = None,
    nz_data: tuple[np.ndarray, dict] | None = None,
) -> sacc.Sacc:
    """Create SACC file from list of cross-spectra.

    Args:
        spectra: List of spectrum dictionaries with 'ells', 'cls', 'bpws', 'metadata'
        global_metadata: Global metadata to add to SACC file
        nz_data: Optional (z_grid, nz_per_bin) for NZTracer creation

    Returns:
        SACC object with all cross-spectra
    """
    s = sacc.Sacc()

    # Add global metadata
    if global_metadata:
        for key, value in global_metadata.items():
            s.metadata[key] = value

    # Unpack n(z) if provided
    z_grid, nz_per_bin = nz_data if nz_data is not None else (None, None)

    # Track unique tracers
    tracers_added = set()

    # Add data points from each spectrum
    for spec in spectra:
        ells = spec["ells"]
        cls = spec["cls"]
        meta = spec["metadata"]

        # Parse tracer information
        method = meta["method"]
        bin_id = meta["bin"]
        cmbk = meta["cmbk"]
        tracer_type = meta.get("tracer_type", "shear")
        # compute_cross_spectrum labels all spin-0 as "mass"; override for density
        if method == "density":
            tracer_type = "density"

        # Create tracer names
        if tracer_type in ("shear", "density"):
            euclid_tracer = f"euclid_{method}_bin{bin_id}"
        else:  # mass
            euclid_tracer = f"euclid_{method}"
            if method == "sksp":
                euclid_tracer += f"_bin{bin_id}"

        cmbk_tracer = f"{cmbk}_kappa"

        # Add tracers if not already added
        if euclid_tracer not in tracers_added:
            bin_key = int(bin_id) if bin_id != "all" else "all"
            if tracer_type in ("shear", "density") and nz_per_bin is not None and bin_key in nz_per_bin:
                s.add_tracer("NZ", euclid_tracer, z=z_grid, nz=nz_per_bin[bin_key])
            else:
                quantity = "galaxy_shear" if tracer_type == "shear" else "cmb_convergence"
                s.add_tracer("Misc", euclid_tracer, quantity=quantity)
            tracers_added.add(euclid_tracer)

        if cmbk_tracer not in tracers_added:
            s.add_tracer("Misc", cmbk_tracer, quantity="cmb_convergence")
            tracers_added.add(cmbk_tracer)

        # Determine data type based on spin combination
        # Spin-2 × Spin-0 gives EB cross-correlation
        # Spin-0 × Spin-0 gives scalar correlation
        if tracer_type == "shear":
            # Shear (spin-2) × CMB-κ (spin-0)
            # cls array has shape (2, n_ells) for [E×κ, B×κ]
            if len(cls.shape) == 2 and cls.shape[0] == 2:
                # Add E-mode cross
                for i, ell in enumerate(ells):
                    s.add_data_point(
                        "cl_0e",
                        (cmbk_tracer, euclid_tracer),
                        cls[0, i],
                        ell=ell,
                    )
                # Add B-mode cross (should be consistent with zero)
                for i, ell in enumerate(ells):
                    s.add_data_point(
                        "cl_0b",
                        (cmbk_tracer, euclid_tracer),
                        cls[1, i],
                        ell=ell,
                    )
            else:
                # Fallback: treat as single spectrum
                for i, ell in enumerate(ells):
                    s.add_data_point(
                        "cl_0e",
                        (cmbk_tracer, euclid_tracer),
                        cls[i] if cls.ndim == 1 else cls[0, i],
                        ell=ell,
                    )
        else:
            # Mass (spin-0) × CMB-κ (spin-0)
            # cls array has shape (n_ells,)
            for i, ell in enumerate(ells):
                s.add_data_point(
                    "cl_00",
                    (euclid_tracer, cmbk_tracer),
                    cls[i] if cls.ndim == 1 else cls[0, i],
                    ell=ell,
                )

    return s


# Extract parameters from snakemake object
npz_files = [Path(f) for f in snakemake.input.npz_files]
output_sacc = Path(snakemake.output.sacc)

# Load n(z) if provided (shear combine rules pass this)
nz_file = getattr(snakemake.input, "nz_file", None)
nz_data = None
if nz_file is not None:
    nz_data = load_nz(Path(nz_file))

# Get metadata from params (may be empty strings)
description = snakemake.params.get("description", "")
methods = snakemake.params.get("methods", "")
bins_info = snakemake.params.get("bins", "")

snakemake_log(snakemake, f"Combining {len(npz_files)} cross-spectrum files into SACC")
snakemake_log(snakemake, f"Output: {output_sacc}")
if nz_data is not None:
    snakemake_log(snakemake, f"n(z) loaded: {len(nz_data[1])} bins from {nz_file}")

# Load all spectra
snakemake_log(snakemake, "\nLoading spectra...")
spectra = []
for npz_file in npz_files:
    try:
        spec = load_npz_spectrum(npz_file)
        spectra.append(spec)
        meta = spec["metadata"]
        snakemake_log(snakemake, f"  Loaded: {meta.get('method', '?')} bin {meta.get('bin', '?')} × {meta.get('cmbk', '?').upper()}")
    except Exception as e:
        snakemake_log(snakemake, f"  Warning: Failed to load {npz_file.name}: {e}")

if not spectra:
    snakemake_log(snakemake, "Error: No spectra could be loaded")
    sys.exit(1)

# Global metadata
global_metadata = {}
if description:
    global_metadata["description"] = description
if methods:
    global_metadata["methods"] = methods
if bins_info:
    global_metadata["bins"] = bins_info
global_metadata["n_spectra"] = len(spectra)

# Create SACC file
snakemake_log(snakemake, f"\nCreating SACC file with {len(spectra)} cross-spectra...")
s = create_sacc_from_spectra(spectra, global_metadata, nz_data=nz_data)

# Build block-diagonal covariance from per-spectrum covariances
cov_blocks = [spec["cov"] for spec in spectra if "cov" in spec]
if cov_blocks and len(cov_blocks) == len(spectra):
    from scipy.linalg import block_diag

    full_cov = block_diag(*cov_blocks)
    assert full_cov.shape[0] == len(s.data), (
        f"Covariance size mismatch: {full_cov.shape[0]} vs {len(s.data)} data points"
    )
    s.add_covariance(full_cov)
    snakemake_log(snakemake, f"Covariance: {full_cov.shape[0]}×{full_cov.shape[1]} block-diagonal ({len(cov_blocks)} blocks)")
else:
    snakemake_log(snakemake, f"Warning: covariance not added ({len(cov_blocks)}/{len(spectra)} spectra have cov)")

# Save
output_sacc.parent.mkdir(parents=True, exist_ok=True)
snakemake_log(snakemake, f"Saving to {output_sacc}")
s.save_fits(str(output_sacc), overwrite=True)

snakemake_log(snakemake, f"\nDone! SACC file contains:")
snakemake_log(snakemake, f"  Tracers: {len(s.tracers)}")
for name, t in s.tracers.items():
    snakemake_log(snakemake, f"    {name}: {type(t).__name__}")
snakemake_log(snakemake, f"  Data points: {len(s.data)}")
snakemake_log(snakemake, f"  Data types: {set(d.data_type for d in s.data)}")
snakemake_log(snakemake, f"  Has covariance: {s.covariance is not None}")
