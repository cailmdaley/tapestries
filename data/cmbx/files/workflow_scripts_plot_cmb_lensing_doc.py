"""CMB lensing map docs for a single experiment: unmasked and apodized views."""

from pathlib import Path

import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skyproj
from matplotlib.colors import Normalize

import pickle

from dr1_cmbx.eDR1data.spectra import make_mask
from dr1_notebooks.scratch.cdaley.rr2 import RR2Coverage
from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.plot_utils import save_evidence, setup_theme, wiener_filter_kappa

snakemake = snakemake  # type: ignore # noqa: F821

kappa_path = Path(snakemake.input.kappa)
mask_path = Path(snakemake.input.mask)
output_map = Path(snakemake.output.png_map)
output_mask = Path(snakemake.output.png_mask)
evidence_path = Path(snakemake.output.evidence)

nside = int(snakemake.config["nside"])
aposcale = float(snakemake.config["cross_correlation"]["aposcale"])

claim_id = str(snakemake.params.claim_id)
experiment_name = str(snakemake.params.experiment_name)
experiment_short = str(snakemake.params.experiment_short)

proj_params = [
    {"lon_0": 70, "lat_0": -33, "extent": [63.8, 77.2, -37, -26.5]},
    {"lon_0": 45, "lat_0": -64, "extent": [32, 70, -44, -63.5]},
]

kappa = hp.read_map(str(kappa_path), dtype=np.float64)
mask = hp.read_map(str(mask_path), dtype=np.float64)
if hp.npix2nside(len(kappa)) != nside:
    kappa = hp.ud_grade(kappa, nside)
if hp.npix2nside(len(mask)) != nside:
    mask = hp.ud_grade(mask, nside)

with open(snakemake.input.shear_pkl, "rb") as f:
    shear_data = pickle.load(f)
bin_key = next(k for k in shear_data if k.endswith("bin"))
bd = shear_data[bin_key][f"bin{snakemake.params.all_bin_idx}"]
npix = hp.nside2npix(nside)
shear_weights = np.zeros(npix)
shear_weights[bd["ipix"]] = bd["sum_weights"]
shear_bin = (shear_weights > 0).astype(np.float64)
support_apod = make_mask(shear_bin, mask, aposcale=aposcale)

snakemake_log(snakemake, f"Applying Wiener filter to {experiment_name} kappa")
filtered = wiener_filter_kappa(kappa, mask, nside=nside)

mask_apod = make_mask(mask, aposcale=aposcale)
map_unmasked = filtered * (mask > 0).astype(np.float64)
map_apod = filtered * support_apod

coverage = RR2Coverage.from_mask(nside, (mask > 0) | (shear_bin > 0), sparse=True)


def _plot_two_patch(map_data, out_png, title, norm):
    setup_theme("whitegrid")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(20, 10),
        gridspec_kw={"width_ratios": [46.5, 53.5]},
        layout="constrained",
    )
    sparse = coverage.sparsify(map_data)
    patch_labels = ["North patch", "South patch"]
    for col_idx, pp in enumerate(proj_params):
        sp = skyproj.GnomonicSkyproj(axes[col_idx], **pp)
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
        axes[col_idx].set_title(patch_labels[col_idx], fontsize=12, fontweight="bold")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


unmasked_sparse = coverage.sparsify(map_unmasked)
finite = unmasked_sparse[np.isfinite(unmasked_sparse) & (unmasked_sparse != 0)]
vmax = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
shared_norm = Normalize(vmin=-vmax, vmax=vmax)

_plot_two_patch(
    map_unmasked,
    output_map,
    f"{experiment_name} kappa on sky (Wiener-filtered)",
    shared_norm,
)
_plot_two_patch(
    map_apod,
    output_mask,
    f"{experiment_name} kappa x apodized {experiment_short} x Euclid masks (estimator support)",
    shared_norm,
)

fsky_raw = float(np.mean(mask > 0))
fsky_apod = float(np.mean(mask_apod))
fsky_overlap_raw = float(np.mean((mask > 0).astype(np.float64) * shear_bin))
fsky_overlap_apod = float(np.mean(support_apod))

save_evidence(
    {
        "id": claim_id,
        "output": {
            "png_map": output_map.name,
            "png_mask": output_mask.name,
        },
        "evidence": {
            "nside": nside,
            "wiener_filtered": True,
            "masking": {
                "method": "C2",
                "aposcale_deg": round(aposcale, 3),
                "fsky_raw": round(fsky_raw, 6),
                "fsky_apodized": round(fsky_apod, 6),
                "area_raw_deg2": round(fsky_raw * 41253.0, 2),
                "area_apodized_deg2": round(fsky_apod * 41253.0, 2),
                "effective_overlap_raw": round(fsky_overlap_raw, 6),
                "effective_overlap_apodized": round(fsky_overlap_apod, 6),
                "effective_overlap_raw_deg2": round(fsky_overlap_raw * 41253.0, 2),
                "effective_overlap_apodized_deg2": round(fsky_overlap_apod * 41253.0, 2),
            },
        },
    },
    evidence_path,
    snakemake,
)

snakemake_log(snakemake, f"Saved {output_map}")
snakemake_log(snakemake, f"Saved {output_mask}")
snakemake_log(snakemake, f"Saved {evidence_path}")
