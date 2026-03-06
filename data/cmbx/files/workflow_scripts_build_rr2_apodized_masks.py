"""Build official RR2 apodized masks for all main tracer footprint pairs."""

import json
import pickle
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt

from dr1_cmbx.eDR1data.spectra import make_mask
from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log


snakemake = snakemake  # type: ignore # noqa: F821

shear_pkl_path = Path(snakemake.input.shear_pkl)
density_pkl_path = Path(snakemake.input.density_pkl)
act_mask_path = Path(snakemake.input.act_mask)
spt_mask_path = Path(snakemake.input.spt_mask)

aposcale = float(snakemake.params.aposcale)
nside = int(snakemake.params.nside)
npix = hp.nside2npix(nside)

out_paths = {k: Path(v) for k, v in snakemake.output.items()}

snakemake_log(snakemake, "Building official RR2 apodized masks")


# ---------------------------------------------------------------------------
# Load shear sum_weights for "all" bin from pickle → dense array
def _load_sparse_field(pkl_path, field):
    """Load one field from the 'all' bin (last bin) of a pickle, expand to dense."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    bin_key = next(k for k in data if k.endswith("bin"))
    # "all" bin is the last one
    all_idx = max(int(k.replace("bin", "")) for k in data[bin_key])
    bd = data[bin_key][f"bin{all_idx}"]
    dense = np.zeros(npix, dtype=np.float64)
    dense[bd["ipix"]] = bd[field]
    return dense


shear_raw = _load_sparse_field(shear_pkl_path, "sum_weights")
density_raw = _load_sparse_field(density_pkl_path, "mask")

snakemake_log(snakemake, f"  Shear mask from:   {shear_pkl_path}")
snakemake_log(snakemake, f"  Density mask from: {density_pkl_path}")

act_raw = np.asarray(
    hp.read_map(str(act_mask_path), dtype=np.float64), dtype=np.float64
)
spt_raw = np.asarray(
    hp.read_map(str(spt_mask_path), dtype=np.float64), dtype=np.float64
)

if hp.npix2nside(act_raw.size) != nside:
    act_raw = hp.ud_grade(act_raw, nside)
if hp.npix2nside(spt_raw.size) != nside:
    spt_raw = hp.ud_grade(spt_raw, nside)

shear_bin = (shear_raw > 0).astype(np.float64)
density_bin = (density_raw > 0).astype(np.float64)
act_bin = (act_raw > 0).astype(np.float64)
spt_bin = (spt_raw > 0).astype(np.float64)

masks = {
    "shear_apod": make_mask(shear_raw, aposcale=aposcale, weights=[shear_raw]),
    "density_apod": make_mask(density_raw, aposcale=aposcale, weights=[density_raw]),
    "act_apod": nmt.mask_apodization(act_raw, aposcale, "C2"),
    "spt_apod": nmt.mask_apodization(spt_raw, aposcale, "C2"),
    "shear_x_density_apod": nmt.mask_apodization(shear_bin * density_bin, aposcale, "C2"),
    "shear_x_act_apod": nmt.mask_apodization(shear_bin * act_bin, aposcale, "C2"),
    "shear_x_spt_apod": nmt.mask_apodization(shear_bin * spt_bin, aposcale, "C2"),
    "density_x_act_apod": nmt.mask_apodization(density_bin * act_bin, aposcale, "C2"),
    "density_x_spt_apod": nmt.mask_apodization(density_bin * spt_bin, aposcale, "C2"),
    "act_x_spt_apod": nmt.mask_apodization(act_bin * spt_bin, aposcale, "C2"),
}

for key, map_data in masks.items():
    out = out_paths[key]
    out.parent.mkdir(parents=True, exist_ok=True)
    hp.write_map(str(out), map_data, overwrite=True, dtype=np.float64)


def _fsky(mask):
    return float(np.mean(mask > 0)), float(np.mean(mask))


summary = {
    "inputs": {
        "shear_pkl": str(shear_pkl_path),
        "density_pkl": str(density_pkl_path),
        "act_mask": str(act_mask_path),
        "spt_mask": str(spt_mask_path),
    },
    "apodization": {"method": "C2", "aposcale_deg": aposcale},
    "fsky": {},
}

for key, map_data in masks.items():
    raw_fsky, apod_fsky = _fsky(map_data)
    summary["fsky"][key] = {
        "support_fraction": raw_fsky,
        "effective_fraction": apod_fsky,
    }

summary_out = out_paths["summary"]
summary_out.parent.mkdir(parents=True, exist_ok=True)
with open(summary_out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

snakemake_log(snakemake, f"Saved RR2 apodized masks under {summary_out.parent}")
