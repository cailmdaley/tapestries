"""Shared map conventions: overlap of ACT, SPT, and Euclid RR2 footprints."""

from pathlib import Path

import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skyproj
from matplotlib import colors as mcolors
from matplotlib.patches import Patch

import pickle

from dr1_notebooks.scratch.cdaley.rr2 import RR2Coverage
from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.plot_utils import save_evidence, setup_theme

snakemake = snakemake  # type: ignore # noqa: F821

act_mask_path = Path(snakemake.input.act_mask)
spt_mask_path = Path(snakemake.input.spt_mask)
output_png = Path(snakemake.output.png)
evidence_path = Path(snakemake.output.evidence)

nside = int(snakemake.config["nside"])

proj_params = [
    {"lon_0": 70, "lat_0": -33, "extent": [63.8, 77.2, -37, -26.5]},
    {"lon_0": 45, "lat_0": -64, "extent": [32, 70, -44, -63.5]},
]

snakemake_log(snakemake, "Loading footprint masks")
act_mask = hp.read_map(str(act_mask_path), dtype=np.float64)
if hp.npix2nside(len(act_mask)) != nside:
    act_mask = hp.ud_grade(act_mask, nside)

spt_mask = hp.read_map(str(spt_mask_path), dtype=np.float64)
if hp.npix2nside(len(spt_mask)) != nside:
    spt_mask = hp.ud_grade(spt_mask, nside)

with open(snakemake.input.shear_pkl, "rb") as f:
    shear_data = pickle.load(f)
bin_key = next(k for k in shear_data if k.endswith("bin"))
bd = shear_data[bin_key][f"bin{snakemake.params.all_bin_idx}"]
npix = hp.nside2npix(nside)
shear_weights = np.zeros(npix)
shear_weights[bd["ipix"]] = bd["sum_weights"]

act_bin = (act_mask > 0).astype(np.int8)
spt_bin = (spt_mask > 0).astype(np.int8)
euclid_bin = (shear_weights > 0).astype(np.int8)

# Bit-packed category code: ACT=1, SPT=2, Euclid=4
category = act_bin + (2 * spt_bin) + (4 * euclid_bin)
coverage = RR2Coverage.from_mask(nside, category > 0, sparse=True)
cat_sparse = coverage.sparsify(category.astype(float))

counts = {f"code_{i}": int(np.sum(category == i)) for i in range(8)}
fsky = {f"code_{i}": float(counts[f"code_{i}"] / category.size) for i in range(8)}
area = {f"code_{i}": float(fsky[f"code_{i}"] * 41253.0) for i in range(8)}

# Human-readable keys for evidence (ACT=bit0, SPT=bit1, Euclid=bit2)
combo_keys = {
    1: "act_only",
    2: "spt_only",
    3: "act_spt",
    4: "euclid_only",
    5: "act_euclid",
    6: "spt_euclid",
    7: "act_spt_euclid",
}

labels = {
    1: "ACT only",
    2: "SPT only",
    3: "ACT+SPT",
    4: "Euclid only",
    5: "ACT+Euclid",
    6: "SPT+Euclid",
    7: "ACT+SPT+Euclid",
}

# 0 unused in plotted region, keep white; non-zero categories are distinct.
palette = [
    "#ffffff", "#1f77b4", "#ff7f0e", "#9467bd",
    "#2ca02c", "#17becf", "#bcbd22", "#d62728",
]
cmap = mcolors.ListedColormap(palette)
bounds = np.arange(-0.5, 8.5, 1.0)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

setup_theme("whitegrid")
fig, axes = plt.subplots(
    1,
    2,
    figsize=(20, 10),
    gridspec_kw={"width_ratios": [46.5, 53.5]},
    layout="constrained",
)

patch_labels = ["North patch", "South patch"]

for col, pp in enumerate(proj_params):
    sp = skyproj.GnomonicSkyproj(axes[col], **pp)
    sp.draw_hpxpix(
        nside,
        coverage.ipix_sparse,
        cat_sparse,
        nest=coverage.nest,
        zoom=False,
        cmap=cmap,
        norm=norm,
    )
for col, label in enumerate(patch_labels):
    bbox = axes[col].get_position()
    fig.text(
        0.5 * (bbox.x0 + bbox.x1),
        bbox.y1 + 0.01,
        label,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

legend_lines = [
    f"{labels[k]}: {area[f'code_{k}']:.0f} deg$^2$"
    for k in [1, 2, 3, 4, 5, 6, 7]
    if counts[f"code_{k}"] > 0
]
fig.suptitle("Shared map conventions: ACT, SPT, and Euclid footprint overlap", fontsize=15, fontweight="bold")
fig.text(0.02, 0.02, "\n".join(legend_lines), fontsize=11, va="bottom")

legend_handles = [
    Patch(facecolor=palette[k], edgecolor="none", label=labels[k])
    for k in [1, 2, 3, 4, 5, 6, 7]
    if counts[f"code_{k}"] > 0
]
fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.98, 0.82), frameon=True, fontsize=10)

output_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_png, bbox_inches="tight")
plt.close(fig)

save_evidence(
    {
        "id": "shared_map_conventions",
        "output": {"png": output_png.name},
        "evidence": {
            "nside": nside,
            "fsky": {combo_keys[k]: round(fsky[f"code_{k}"], 6) for k in range(1, 8)},
            "area_deg2": {combo_keys[k]: round(area[f"code_{k}"], 2) for k in range(1, 8)},
        },
    },
    evidence_path,
    snakemake,
)

snakemake_log(snakemake, f"Saved {output_png}")
snakemake_log(snakemake, f"Saved {evidence_path}")
