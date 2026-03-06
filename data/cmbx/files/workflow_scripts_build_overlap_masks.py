"""Build overlap masks for the ACT/SPT spatial variation test.

Identifies three regions of the shear footprint based on CMB experiment coverage:
  1. ACT-only — ACT κ coverage but no SPT
  2. Overlap  — both ACT and SPT κ coverage
  3. SPT-only — SPT κ coverage but no ACT

Saves binary masks (0/1) for each region and reports pixel counts + fsky.
The overlap region enables the strongest consistency test: same galaxies,
two independent CMB lensing reconstructions.

Output: three FITS masks + a JSON summary with pixel counts and fsky.
"""

import json
import pickle
from pathlib import Path

import healpy as hp
import numpy as np

from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.nmt_utils import read_healpix_map

snakemake = snakemake  # type: ignore # noqa: F821

nside = int(snakemake.params.nside)
act_mask_path = str(snakemake.input.act_mask)
spt_mask_path = str(snakemake.input.spt_mask)

out_act_only = str(snakemake.output.act_only)
out_overlap = str(snakemake.output.overlap)
out_spt_only = str(snakemake.output.spt_only)
out_summary = str(snakemake.output.summary)

snakemake_log(snakemake, f"Building overlap masks at nside={nside}")

# Load masks — binary threshold at > 0
act_mask = read_healpix_map(act_mask_path)
spt_mask = read_healpix_map(spt_mask_path)
with open(snakemake.input.shear_pkl, "rb") as f:
    shear_data = pickle.load(f)
bin_key = next(k for k in shear_data if k.endswith("bin"))
bd = shear_data[bin_key][f"bin{snakemake.params.bin_idx}"]
npix = hp.nside2npix(nside)
shear_weight = np.zeros(npix)
shear_weight[bd["ipix"]] = bd["sum_weights"]

# Ensure consistent resolution
npix_expected = hp.nside2npix(nside)
for name, m in [("ACT", act_mask), ("SPT", spt_mask), ("shear", shear_weight)]:
    if len(m) != npix_expected:
        raise ValueError(f"{name} mask has {len(m)} pixels, expected {npix_expected} for nside={nside}")

# Binary footprints
act_foot = act_mask > 0
spt_foot = spt_mask > 0
shear_foot = shear_weight > 0

snakemake_log(snakemake, f"  ACT footprint:   {np.sum(act_foot)} px ({np.mean(act_foot):.4f} fsky)")
snakemake_log(snakemake, f"  SPT footprint:   {np.sum(spt_foot)} px ({np.mean(spt_foot):.4f} fsky)")
snakemake_log(snakemake, f"  Shear footprint: {np.sum(shear_foot)} px ({np.mean(shear_foot):.4f} fsky)")

# Three regions (all intersected with shear footprint)
act_only = act_foot & ~spt_foot & shear_foot
overlap = act_foot & spt_foot & shear_foot
spt_only = ~act_foot & spt_foot & shear_foot

regions = {
    "act_only": act_only,
    "overlap": overlap,
    "spt_only": spt_only,
}

summary = {}
for name, region in regions.items():
    npix_region = int(np.sum(region))
    fsky = float(np.mean(region))
    snakemake_log(snakemake, f"  {name}: {npix_region} px ({fsky:.6f} fsky)")
    summary[name] = {
        "npix": npix_region,
        "fsky": round(fsky, 6),
    }

# Estimate modes per bandpower (rough, assuming ℓ_eff ~ 500, Δℓ ~ 150)
# n_modes ~ (2ℓ+1) * Δℓ * fsky
for name, info in summary.items():
    fsky = info["fsky"]
    n_modes_500 = (2 * 500 + 1) * 150 * fsky
    info["approx_modes_ell500"] = round(n_modes_500, 1)
    snakemake_log(snakemake, f"  {name}: ~{n_modes_500:.0f} modes at ℓ~500")

# Save masks
hp.write_map(out_act_only, act_only.astype(np.float64), overwrite=True, dtype=np.float64)
hp.write_map(out_overlap, overlap.astype(np.float64), overwrite=True, dtype=np.float64)
hp.write_map(out_spt_only, spt_only.astype(np.float64), overwrite=True, dtype=np.float64)

# Save summary
with open(out_summary, "w") as f:
    json.dump(summary, f, indent=2)

snakemake_log(snakemake, f"Saved overlap masks and summary to {Path(out_summary).parent}")
