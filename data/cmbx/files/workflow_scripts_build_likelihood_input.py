"""Build eDR1like-compatible PKL file from pipeline NPZ outputs.

Bridges the cross-correlation pipeline (NPZ + SACC) to the eDR1like likelihood
(PKL format). Reads per-bin NPZ files for one method × one CMB experiment,
extracts bandpower windows (Bbl), measured spectra, Knox covariance, and n(z)
distributions. Outputs a NumPy pickle that LikeCobayaPyccl can load directly.

PKL structure:
  cls[spectrum_name] = {cl, nl, Bbl, leff}
  cov = block-diagonal list-of-lists (Knox)
  cov_all = flattened covariance matrix
  dndz = {bin0: [z, nz], bin1: [z, nz], ...}
  leff = effective multipoles

Spectrum naming: "kappa_g{i}" for CMB κ × shear bin {i+1} (0-indexed).
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits

from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log


snakemake = snakemake  # type: ignore # noqa: F821

method = snakemake.wildcards.method
cmbk = snakemake.wildcards.cmbk
npz_paths = [Path(p) for p in snakemake.input.npz_files]
nz_path = Path(snakemake.input.nz_fits)
output_pkl = Path(snakemake.output.pkl)
evidence_path = Path(snakemake.output.evidence)

snakemake_log(snakemake, f"Building likelihood input: {method} x {cmbk}")

# --- Configuration ---
bin_ids = ["1", "2", "3", "4", "5", "6"]  # Euclid tomographic bins (1-indexed)
n_bins = len(bin_ids)

# --- Load n(z) ---
# FITS structure: BIN_INFO HDU with BIN_ID (1-6) and N_Z (3000-element array) per row.
# Z grid is np.linspace(0, 6, 3000), reconstructed (not a column).
# Following compute_theory_cls.py pattern.
snakemake_log(snakemake, f"  Loading n(z) from {nz_path}")
with fits.open(str(nz_path)) as hdul:
    nz_table = hdul["BIN_INFO"].data
    z_grid = np.linspace(0, 6, 3000)
    dndz = {}
    for i in range(len(nz_table)):
        bin_id = nz_table["BIN_ID"][i]  # 1-indexed
        nz_raw = nz_table["N_Z"][i].astype(np.float64)
        # Normalize so integral = 1
        norm = np.trapz(nz_raw, z_grid)
        if norm > 0:
            nz_raw = nz_raw / norm
        # 0-indexed for likelihood: bin0 = Euclid bin1
        dndz[f"bin{bin_id - 1}"] = np.array([z_grid, nz_raw])
        snakemake_log(snakemake, f"    bin{bin_id - 1} (Euclid bin{bin_id}): {len(z_grid)} z-points, z=[{z_grid[0]:.3f}, {z_grid[-1]:.3f}]")

# --- Load per-bin spectra ---
cls_dict = {}
leff_global = None
n_ell = None

# Build path→bin mapping from the sorted NPZ paths
for bin_idx, bin_id in enumerate(bin_ids):
    # Find the matching NPZ path
    pattern = f"{method}_bin{bin_id}_x_{cmbk}"
    matching = [p for p in npz_paths if pattern in p.stem]
    if not matching:
        raise FileNotFoundError(f"No NPZ file found for {pattern} in {[p.name for p in npz_paths]}")
    npz_path = matching[0]

    snakemake_log(snakemake, f"  Loading bin{bin_id}: {npz_path.name}")
    data = np.load(str(npz_path), allow_pickle=True)

    ell_eff = data["ells"]
    cl_full = data["cls"]  # shape (n_components, n_ell)
    bpws_full = data["bpws"]  # shape (n_spin_out, n_ell, n_spin_in, lmax+1)

    # E-mode cross-spectrum (first spin component)
    cl_ee = cl_full[0]  # shape (n_ell,)

    # E-mode bandpower window: bpws[0, :, 0, :] for spin-2 × spin-0
    bbl = bpws_full[0, :, 0, :]  # shape (n_ell, lmax+1)

    # Knox variance (diagonal noise estimate)
    knox_var = data["knox_var"] if "knox_var" in data else None

    # Store
    spectrum_name = f"kappa_g{bin_idx}"
    cls_dict[spectrum_name] = {
        "cl": cl_ee,
        "nl": np.zeros_like(cl_ee),  # noise is in the Knox variance, not subtracted
        "Bbl": bbl,
        "leff": ell_eff,
    }

    if leff_global is None:
        leff_global = ell_eff
        n_ell = len(ell_eff)

    snakemake_log(snakemake, f"    {spectrum_name}: {n_ell} bins, ell=[{ell_eff[0]:.0f}, {ell_eff[-1]:.0f}], Bbl shape {bbl.shape}")

# --- Build covariance ---
# Block-diagonal Knox covariance: only diagonal blocks (no cross-bin correlations)
snakemake_log(snakemake, "  Building block-diagonal Knox covariance")
cov_blocks = [[None for _ in range(n_bins)] for _ in range(n_bins)]
cov_all = np.zeros((n_bins * n_ell, n_bins * n_ell))

for bin_idx, bin_id in enumerate(bin_ids):
    pattern = f"{method}_bin{bin_id}_x_{cmbk}"
    matching = [p for p in npz_paths if pattern in p.stem]
    npz_path = matching[0]
    data = np.load(str(npz_path), allow_pickle=True)

    knox_var = data["knox_var"] if "knox_var" in data else np.ones(n_ell) * 1e-20

    # Diagonal block
    cov_blocks[bin_idx][bin_idx] = np.diag(knox_var)

    # Fill cov_all
    i0 = bin_idx * n_ell
    i1 = (bin_idx + 1) * n_ell
    cov_all[i0:i1, i0:i1] = np.diag(knox_var)

    # Off-diagonal blocks are zero (independent bins)
    for j in range(n_bins):
        if j != bin_idx:
            cov_blocks[bin_idx][j] = np.zeros((n_ell, n_ell))

# --- Assemble PKL ---
pkl_data = {
    "cls": cls_dict,
    "cov": cov_blocks,
    "cov_all": cov_all,
    "dndz": dndz,
    "leff": leff_global,
}

output_pkl.parent.mkdir(parents=True, exist_ok=True)
with open(str(output_pkl), "wb") as f:
    pickle.dump(pkl_data, f)

snakemake_log(snakemake, f"Saved: {output_pkl}")
snakemake_log(snakemake, f"  {len(cls_dict)} spectra, {n_bins} bins, {n_ell} ell-bins")
snakemake_log(snakemake, f"  Covariance: {cov_all.shape}, block-diagonal Knox")

# --- Evidence ---
evidence = {
    "id": "build_likelihood_input",
    "generated": datetime.now(timezone.utc).isoformat(),
    "evidence": {
        "method": method,
        "cmbk": cmbk,
        "n_spectra": len(cls_dict),
        "n_bins": n_bins,
        "n_ell": n_ell,
        "ell_range": f"{leff_global[0]:.0f}-{leff_global[-1]:.0f}",
        "bbl_shape": f"{list(bpws_full.shape)}",
        "cov_type": "block-diagonal Knox",
        "nz_bins": len(dndz),
        "nz_z_points": len(z_grid),
    },
}

evidence_path.parent.mkdir(parents=True, exist_ok=True)
with open(str(evidence_path), "w") as f:
    json.dump(evidence, f, indent=2)

snakemake_log(snakemake, f"Evidence: {evidence_path}")
