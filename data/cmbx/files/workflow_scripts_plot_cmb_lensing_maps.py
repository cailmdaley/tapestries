"""CMB lensing maps: ACT and SPT kappa (Wiener-filtered) on RR2 footprint."""

from pathlib import Path

import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skyproj
from matplotlib.colors import Normalize

import pickle

from dr1_notebooks.scratch.cdaley.rr2 import RR2Coverage
from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.plot_utils import save_evidence, setup_theme, wiener_filter_kappa

snakemake = snakemake  # type: ignore # noqa: F821

act_kappa_path = Path(snakemake.input.act_kappa)
act_mask_path = Path(snakemake.input.act_mask)
spt_kappa_path = Path(snakemake.input.spt_kappa)
spt_mask_path = Path(snakemake.input.spt_mask)
output_png = Path(snakemake.output.png)
evidence_path = Path(snakemake.output.evidence)

nside = int(snakemake.config["nside"])

proj_params = [
    {"lon_0": 70, "lat_0": -33, "extent": [63.8, 77.2, -37, -26.5]},
    {"lon_0": 45, "lat_0": -64, "extent": [32, 70, -44, -63.5]},
]

snakemake_log(snakemake, "Loading CMB maps and masks")
act_kappa = hp.read_map(str(act_kappa_path), dtype=np.float64)
act_mask = hp.read_map(str(act_mask_path), dtype=np.float64)
if hp.npix2nside(len(act_kappa)) != nside:
    act_kappa = hp.ud_grade(act_kappa, nside)
if hp.npix2nside(len(act_mask)) != nside:
    act_mask = hp.ud_grade(act_mask, nside)

spt_kappa = hp.read_map(str(spt_kappa_path), dtype=np.float64)
spt_mask = hp.read_map(str(spt_mask_path), dtype=np.float64)
if hp.npix2nside(len(spt_kappa)) != nside:
    spt_kappa = hp.ud_grade(spt_kappa, nside)
if hp.npix2nside(len(spt_mask)) != nside:
    spt_mask = hp.ud_grade(spt_mask, nside)

with open(snakemake.input.shear_pkl, "rb") as f:
    shear_data = pickle.load(f)
bin_key = next(k for k in shear_data if k.endswith("bin"))
bd = shear_data[bin_key][f"bin{snakemake.params.all_bin_idx}"]
npix = hp.nside2npix(nside)
shear_weights = np.zeros(npix)
shear_weights[bd["ipix"]] = bd["sum_weights"]
shear_bin = (shear_weights > 0).astype(np.float64)

snakemake_log(snakemake, "Applying Wiener filter")
act_filtered = wiener_filter_kappa(act_kappa, act_mask, nside=nside)
spt_filtered = wiener_filter_kappa(spt_kappa, spt_mask, nside=nside)

act_plot = act_filtered * (act_mask > 0).astype(np.float64) * shear_bin
spt_plot = spt_filtered * (spt_mask > 0).astype(np.float64) * shear_bin

coverage = RR2Coverage.from_mask(nside, shear_bin > 0, sparse=True)

setup_theme("whitegrid")
fig, axes = plt.subplots(
    2,
    2,
    figsize=(20, 16),
    gridspec_kw={"width_ratios": [46.5, 53.5]},
    layout="constrained",
)

rows = [
    ("ACT DR6 kappa (Wiener-filtered)", act_plot),
    ("SPT-3G GMV kappa (Wiener-filtered)", spt_plot),
]
patch_labels = ["North patch", "South patch"]

for row_idx, (label, m) in enumerate(rows):
    sparse = coverage.sparsify(m)
    finite = sparse[np.isfinite(sparse) & (sparse != 0)]
    vmax = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)

    for col_idx, pp in enumerate(proj_params):
        ax = axes[row_idx, col_idx]
        sp = skyproj.GnomonicSkyproj(ax, **pp)
        sp.draw_hpxpix(
            nside,
            coverage.ipix_sparse,
            sparse,
            nest=coverage.nest,
            zoom=False,
            cmap="RdBu_r",
            norm=norm,
        )
        if col_idx == 1:
            sp.draw_inset_colorbar(loc="upper right")

for row_idx, (label, _) in enumerate(rows):
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

fig.suptitle("CMB lensing maps on RR2 (ACT + SPT)", fontsize=15, fontweight="bold")

output_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_png, bbox_inches="tight")
plt.close(fig)

fsky_act = float(np.mean(act_mask > 0))
fsky_spt = float(np.mean(spt_mask > 0))
fsky_rr2 = float(np.mean(shear_bin > 0))
fsky_act_rr2 = float(np.mean((act_mask > 0) & (shear_bin > 0)))
fsky_spt_rr2 = float(np.mean((spt_mask > 0) & (shear_bin > 0)))

save_evidence(
    {
        "id": "cmb_lensing_maps",
        "output": {"png": output_png.name},
        "evidence": {
            "nside": nside,
            "wiener_filtered": True,
            "fsky": {
                "act_raw": round(fsky_act, 6),
                "spt_raw": round(fsky_spt, 6),
                "rr2": round(fsky_rr2, 6),
                "act_x_rr2": round(fsky_act_rr2, 6),
                "spt_x_rr2": round(fsky_spt_rr2, 6),
            },
            "area_deg2": {
                "act_x_rr2": round(fsky_act_rr2 * 41253.0, 2),
                "spt_x_rr2": round(fsky_spt_rr2 * 41253.0, 2),
            },
        },
    },
    evidence_path,
    snakemake,
)

snakemake_log(snakemake, f"Saved {output_png}")
snakemake_log(snakemake, f"Saved {evidence_path}")
