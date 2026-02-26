"""Compute global color limits across tomographic bins 1-6 for consistent scaling."""

import json
import pickle
from pathlib import Path

import healpy as hp
import numpy as np

snakemake = globals().get("snakemake")  # type: ignore
if snakemake is None:
    raise RuntimeError("This script must be executed via Snakemake")

method = snakemake.wildcards.method

# Load pickle (all bins inside)
with open(snakemake.input.maps_pkl, "rb") as f:
    data = pickle.load(f)
bin_key = next(k for k in data if k.endswith("bin"))

# Collect values across bins 1-6
all_weights = []
all_e1 = []
all_e2 = []

for bin_id in range(1, 7):
    bd = data[bin_key][f"bin{bin_id - 1}"]
    nside = snakemake.params.nside
    npix = hp.nside2npix(nside)

    e1 = np.zeros(npix)
    e2 = np.zeros(npix)
    weights = np.zeros(npix)
    e1[bd["ipix"]] = bd["e1"]
    e2[bd["ipix"]] = bd["e2"]
    weights[bd["ipix"]] = bd["sum_weights"]

    mask = weights > 0
    all_weights.extend(weights[mask])
    all_e1.extend(e1[mask])
    all_e2.extend(e2[mask])

# Compute global percentile limits across all bins
all_weights = np.array(all_weights)
all_e1 = np.array(all_e1)
all_e2 = np.array(all_e2)

global_vlims = {
    "weights": (
        np.percentile(all_weights, 1),
        np.percentile(all_weights, 99),
    ),
    "e1": (
        np.percentile(all_e1, 1),
        np.percentile(all_e1, 99),
    ),
    "e2": (
        np.percentile(all_e2, 1),
        np.percentile(all_e2, 99),
    ),
}

# Save to JSON
output_path = Path(snakemake.output[0])
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w") as f:
    json.dump(global_vlims, f, indent=2)

print(f"Computed global color limits for method={method}:")
for field, (vmin, vmax) in global_vlims.items():
    print(f"  {field}: [{vmin:.6f}, {vmax:.6f}]")
