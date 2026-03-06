"""Generate diagnostic plots for shear maps produced by build_maps."""

import pickle
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dr1_notebooks.scratch.cdaley.rr2 import RR2Coverage
from dr1_notebooks.scratch.cdaley.plot_utils import setup_theme

setup_theme()

snakemake = globals().get("snakemake")  # type: ignore
if snakemake is None:
    raise RuntimeError("This script must be executed via Snakemake")

method = snakemake.wildcards.method
bin_label = snakemake.wildcards.bin
bin_idx = snakemake.params.bin_idx
nside = snakemake.params.nside

# Optional global color limits (for consistent scaling across bins)
global_vlims = snakemake.params.get("global_vlims", None)

log_fp = None
if snakemake.log:
    log_path = Path(snakemake.log[0])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("w")


def log(message: str) -> None:
    print(message)
    if log_fp is not None:
        print(message, file=log_fp)


# Load from pickle
with open(snakemake.input.maps_pkl, "rb") as f:
    data = pickle.load(f)
bin_key = next(k for k in data if k.endswith("bin"))
bd = data[bin_key][f"bin{bin_idx}"]

npix = hp.nside2npix(nside)
e1 = np.zeros(npix)
e2 = np.zeros(npix)
sum_weights_map = np.zeros(npix)
neff_map = np.zeros(npix)
e1[bd["ipix"]] = bd["e1"]
e2[bd["ipix"]] = bd["e2"]
sum_weights_map[bd["ipix"]] = bd["sum_weights"]
neff_map[bd["ipix"]] = bd["neff"]
shear_maps = np.array([e1, e2])

mask_binary = (sum_weights_map > 0).astype(float)
coverage = RR2Coverage.from_mask(nside, mask_binary, sparse=True)

log(f"Loaded shear maps for method={method}, bin={bin_label} at NSIDE={nside}")

Path(snakemake.output.sum_weights_plot).parent.mkdir(parents=True, exist_ok=True)

# Colormap helper
cubehelix = lambda start: sns.cubehelix_palette(start=start, rot=0.5, as_cmap=True)
icefire = sns.color_palette("icefire", as_cmap=True)

# Use global limits for bins 1-6, compute locally for 'all' bin
use_global_limits = global_vlims is not None and bin_label != "all"

# Plot unsmoothed sparse maps with appropriate colormaps
plot_configs = [
    (sum_weights_map, snakemake.output.sum_weights_plot, "sum_weights", cubehelix(0), "asym", (1, 99)),
    (neff_map, snakemake.output.neff_plot, "n_eff", cubehelix(2.5), (0, neff_map.max()), None),
    (shear_maps[0] * mask_binary, snakemake.output.e1_map, "e1", icefire, "sym", (1, 99)),
    (shear_maps[1] * mask_binary, snakemake.output.e2_map, "e2", icefire, "sym", (1, 99)),
]

for map_data, output_path, label, cmap, vlim, vlim_percentiles in plot_configs:
    sparse_map = coverage.sparsify(map_data)

    # Override vlim_percentiles with global limits if available and applicable
    if use_global_limits and label in global_vlims:
        fig, _ = coverage.plot(sparse_map, cmap=cmap, vlim=tuple(global_vlims[label]))
        log(f"Using global color limits for {label}: {global_vlims[label]}")
    elif vlim_percentiles is not None:
        fig, _ = coverage.plot(sparse_map, cmap=cmap, vlim=vlim, vlim_percentiles=vlim_percentiles)
    else:
        fig, _ = coverage.plot(sparse_map, cmap=cmap, vlim=vlim)

    fig.suptitle(f"{method} {label} (bin={bin_label})", y=0.85)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

log("Saved shear map diagnostic plots")

if log_fp is not None:
    log_fp.close()
