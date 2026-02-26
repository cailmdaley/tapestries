"""Euclid RR2 map docs: shear weights + density, unmasked and apodized."""

from pathlib import Path

import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import skyproj
from matplotlib.colors import Normalize

import pickle

from dr1_notebooks.scratch.cdaley.rr2 import RR2Coverage
from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.plot_utils import save_evidence, setup_theme

snakemake = snakemake  # type: ignore # noqa: F821

act_mask_path = Path(snakemake.input.act_mask)
spt_mask_path = Path(snakemake.input.spt_mask)
shear_apod_mask_path = Path(snakemake.input.shear_apod_mask)
density_apod_mask_path = Path(snakemake.input.density_apod_mask)
support_apod_mask_path = Path(snakemake.input.support_apod_mask)
shear_x_act_apod_mask_path = Path(snakemake.input.shear_x_act_apod_mask)
shear_x_spt_apod_mask_path = Path(snakemake.input.shear_x_spt_apod_mask)
density_x_act_apod_mask_path = Path(snakemake.input.density_x_act_apod_mask)
density_x_spt_apod_mask_path = Path(snakemake.input.density_x_spt_apod_mask)
output_map = Path(snakemake.output.png_map)
output_mask = Path(snakemake.output.png_mask)
evidence_path = Path(snakemake.output.evidence)

nside = int(snakemake.config["nside"])
aposcale = float(snakemake.config["cross_correlation"]["aposcale"])

proj_params = [
    {"lon_0": 70, "lat_0": -33, "extent": [63.8, 77.2, -37, -26.5]},
    {"lon_0": 45, "lat_0": -64, "extent": [32, 70, -44, -63.5]},
]

# Euclid inputs
npix = hp.nside2npix(nside)
with open(snakemake.input.shear_pkl, "rb") as f:
    shear_data = pickle.load(f)
shear_bin_key = next(k for k in shear_data if k.endswith("bin"))
shear_bd = shear_data[shear_bin_key][f"bin{snakemake.params.all_bin_idx}"]
shear_weights = np.zeros(npix)
shear_weights[shear_bd["ipix"]] = shear_bd["sum_weights"]

with open(snakemake.input.density_pkl, "rb") as f:
    density_data = pickle.load(f)
density_bin_key = next(k for k in density_data if k.endswith("bin"))
density_bd = density_data[density_bin_key][f"bin{snakemake.params.all_bin_idx}"]
density_map = np.zeros(npix)
density_map[density_bd["ipix"]] = density_bd["map"]
density_mask_raw = np.zeros(npix)
density_mask_raw[density_bd["ipix"]] = density_bd["mask"]

# CMB masks for overlap statistics
act_mask = hp.read_map(str(act_mask_path), dtype=np.float64)
if hp.npix2nside(len(act_mask)) != nside:
    act_mask = hp.ud_grade(act_mask, nside)
spt_mask = hp.read_map(str(spt_mask_path), dtype=np.float64)
if hp.npix2nside(len(spt_mask)) != nside:
    spt_mask = hp.ud_grade(spt_mask, nside)

shear_bin = (shear_weights > 0).astype(np.float64)
density_bin = (density_mask_raw > 0).astype(np.float64)
support_raw = shear_bin * density_bin

shear_mask_apod = hp.read_map(str(shear_apod_mask_path), dtype=np.float64)
density_mask_apod = hp.read_map(str(density_apod_mask_path), dtype=np.float64)
support_apod = hp.read_map(str(support_apod_mask_path), dtype=np.float64)
shear_x_act_apod = hp.read_map(str(shear_x_act_apod_mask_path), dtype=np.float64)
shear_x_spt_apod = hp.read_map(str(shear_x_spt_apod_mask_path), dtype=np.float64)
density_x_act_apod = hp.read_map(str(density_x_act_apod_mask_path), dtype=np.float64)
density_x_spt_apod = hp.read_map(str(density_x_spt_apod_mask_path), dtype=np.float64)

coverage = RR2Coverage.from_mask(nside, (shear_bin > 0) | (density_bin > 0), sparse=True)
cmap_cubehelix = sns.cubehelix_palette(start=0, rot=0.5, as_cmap=True)

def _plot_rows(maps, norms, out_png, title):
    setup_theme("whitegrid")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(20, 16),
        gridspec_kw={"width_ratios": [46.5, 53.5]},
        layout="constrained",
    )

    patch_labels = ["North patch", "South patch"]
    for row_idx, (label, map_data) in enumerate(maps):
        sparse = coverage.sparsify(map_data)
        norm = norms[row_idx]

        for col_idx, pp in enumerate(proj_params):
            sp = skyproj.GnomonicSkyproj(axes[row_idx, col_idx], **pp)
            sp.draw_hpxpix(
                nside,
                coverage.ipix_sparse,
                sparse,
                nest=coverage.nest,
                zoom=False,
                cmap=cmap_cubehelix,
                norm=norm,
            )
            if col_idx == 1:
                sp.draw_inset_colorbar(loc="upper right")
    for row_idx, (label, _) in enumerate(maps):
        for col_idx, patch in enumerate(patch_labels):
            bbox = axes[row_idx, col_idx].get_position()
            fig.text(
                0.5 * (bbox.x0 + bbox.x1),
                bbox.y1 + 0.006,
                f"{label} - {patch}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    fig.suptitle(title, fontsize=15, fontweight="bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _build_norm(map_data, symmetric):
    sparse = coverage.sparsify(map_data)
    finite = sparse[np.isfinite(sparse) & (sparse != 0)]
    if not finite.size:
        return Normalize(vmin=-1, vmax=1) if symmetric else Normalize(vmin=0, vmax=1)
    if symmetric:
        vmax = float(np.percentile(np.abs(finite), 99))
        return Normalize(vmin=-vmax, vmax=vmax)
    vmin = float(np.percentile(finite, 1))
    vmax = float(np.percentile(finite, 99))
    return Normalize(vmin=vmin, vmax=vmax)


shared_norms = [
    _build_norm(shear_weights, symmetric=False),
    _build_norm(density_map * density_bin, symmetric=True),
]

_plot_rows(
    [
        ("(a) Shear weights (all bins)", shear_weights),
        ("(b) Density contrast (all bins)", density_map * density_bin),
    ],
    shared_norms,
    output_map,
    "Euclid RR2 maps on sky",
)

_plot_rows(
    [
        ("(a) Shear apodized mask", shear_mask_apod),
        ("(b) Density apodized mask", density_mask_apod),
    ],
    [
        _build_norm(shear_mask_apod, symmetric=False),
        Normalize(vmin=0, vmax=1),
    ],
    output_mask,
    r"Euclid RR2 apodized masks (C2, aposcale=0.15$^\circ$)",
)


def _mask_stats(raw, apod):
    fsky_raw = float(np.mean(raw > 0))
    fsky_apod = float(np.mean(apod))
    return {
        "fsky_raw": round(fsky_raw, 6),
        "fsky_apodized": round(fsky_apod, 6),
        "area_raw_deg2": round(fsky_raw * 41253.0, 2),
        "area_apodized_deg2": round(fsky_apod * 41253.0, 2),
    }

shear_stats = _mask_stats(shear_bin, support_apod)
density_stats = _mask_stats(density_bin, support_apod)

save_evidence(
    {
        "id": "euclid_rr2_maps",
        "output": {
            "png_map": output_map.name,
            "png_mask": output_mask.name,
        },
        "evidence": {
            "nside": nside,
            "masking": {
                "method": "C2",
                "aposcale_deg": round(aposcale, 3),
                "shear": shear_stats,
                "density": density_stats,
                "shared_support": _mask_stats(support_raw, support_apod),
                "effective_overlap_raw": {
                    "shear_x_act": round(float(np.mean(shear_bin * (act_mask > 0))), 6),
                    "shear_x_spt": round(float(np.mean(shear_bin * (spt_mask > 0))), 6),
                    "density_x_act": round(float(np.mean(density_bin * (act_mask > 0))), 6),
                    "density_x_spt": round(float(np.mean(density_bin * (spt_mask > 0))), 6),
                },
                "effective_overlap_apodized": {
                    "shear_x_act": round(float(np.mean(shear_x_act_apod)), 6),
                    "shear_x_spt": round(float(np.mean(shear_x_spt_apod)), 6),
                    "density_x_act": round(float(np.mean(density_x_act_apod)), 6),
                    "density_x_spt": round(float(np.mean(density_x_spt_apod)), 6),
                },
            },
        },
    },
    evidence_path,
    snakemake,
)

snakemake_log(snakemake, f"Saved {output_map}")
snakemake_log(snakemake, f"Saved {output_mask}")
snakemake_log(snakemake, f"Saved {evidence_path}")
