"""Build official apodized masks for a single Euclid tracer x CMB tracer pair."""

import json
import pickle
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt
from dr1_notebooks.scratch.cdaley.nmt_utils import read_healpix_map

from dr1_cmbx.eDR1data.spectra import make_mask
from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log


snakemake = snakemake  # type: ignore # noqa: F821

method = snakemake.wildcards.method
bin_id = snakemake.wildcards.bin
cmbk = snakemake.wildcards.cmbk
bin_idx = int(snakemake.params.bin_idx)

euclid_apod_path = Path(snakemake.output.euclid_apod_mask)
cmbk_apod_path = Path(snakemake.output.cmbk_apod_mask)
overlap_apod_path = Path(snakemake.output.overlap_apod_mask)
summary_path = Path(snakemake.output.mask_summary)

aposcale = float(snakemake.params.aposcale)
nside = int(snakemake.params.nside)
npix = hp.nside2npix(nside)

snakemake_log(snakemake, f"Building apodized pair masks: {method}_bin{bin_id} x {cmbk}")

# --- Load Euclid mask from pickle or FITS ---
if method in ["lensmc", "metacal", "density"]:
    pkl_path = Path(snakemake.input.maps_pkl)
    snakemake_log(snakemake, f"  Euclid pickle: {pkl_path} (bin_idx={bin_idx})")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    bin_key = next(k for k in data if k.endswith("bin"))
    bd = data[bin_key][f"bin{bin_idx}"]

    if method in ["lensmc", "metacal"]:
        # Shear: use sum_weights as continuous mask
        euclid_raw = np.zeros(npix, dtype=np.float64)
        euclid_raw[bd["ipix"]] = bd["sum_weights"]
    else:
        # Density: use binary mask
        euclid_raw = np.zeros(npix, dtype=np.float64)
        euclid_raw[bd["ipix"]] = bd["mask"]
    del data
elif method == "sksp":
    euclid_raw = read_healpix_map(str(snakemake.input.map), field="MASK")
    euclid_raw = np.asarray(euclid_raw, dtype=np.float64)
    if euclid_raw.ndim == 2:
        euclid_raw = euclid_raw[0]
else:
    euclid_raw = read_healpix_map(str(snakemake.input.map))
    euclid_raw = np.asarray(euclid_raw, dtype=np.float64)
    if euclid_raw.ndim == 2:
        euclid_raw = euclid_raw[0]

euclid_apod = make_mask(euclid_raw, aposcale=aposcale, weights=[euclid_raw])
euclid_binary = (euclid_raw > 0).astype(np.float64)

# --- CMB mask (always FITS) ---
cmbk_mask_path = Path(snakemake.input.cmbk_mask)
cmbk_raw = np.asarray(
    hp.read_map(str(cmbk_mask_path), dtype=np.float64), dtype=np.float64
)
if hp.npix2nside(cmbk_raw.size) != nside:
    cmbk_raw = hp.ud_grade(cmbk_raw, nside)

cmbk_apod = nmt.mask_apodization(cmbk_raw, aposcale, "C2")
cmbk_binary = (cmbk_raw > 0).astype(np.float64)

overlap_binary = euclid_binary * cmbk_binary
overlap_apod = nmt.mask_apodization(overlap_binary, aposcale, "C2")

for out in [euclid_apod_path, cmbk_apod_path, overlap_apod_path]:
    out.parent.mkdir(parents=True, exist_ok=True)

hp.write_map(str(euclid_apod_path), euclid_apod, overwrite=True, dtype=np.float64)
hp.write_map(str(cmbk_apod_path), cmbk_apod, overwrite=True, dtype=np.float64)
hp.write_map(str(overlap_apod_path), overlap_apod, overwrite=True, dtype=np.float64)

summary = {
    "id": f"{method}_bin{bin_id}_x_{cmbk}",
    "inputs": {
        "euclid_pkl": str(snakemake.input.maps_pkl) if hasattr(snakemake.input, "maps_pkl") else "n/a",
        "cmbk_mask": str(cmbk_mask_path),
    },
    "outputs": {
        "euclid_apod_mask": str(euclid_apod_path),
        "cmbk_apod_mask": str(cmbk_apod_path),
        "overlap_apod_mask": str(overlap_apod_path),
    },
    "apodization": {"method": "C2", "aposcale_deg": aposcale},
    "fsky": {
        "euclid_raw": float(np.mean(euclid_binary)),
        "euclid_apodized": float(np.mean(euclid_apod)),
        "cmbk_raw": float(np.mean(cmbk_binary)),
        "cmbk_apodized": float(np.mean(cmbk_apod)),
        "overlap_raw": float(np.mean(overlap_binary)),
        "overlap_apodized": float(np.mean(overlap_apod)),
    },
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

snakemake_log(snakemake, f"  Saved {euclid_apod_path}")
snakemake_log(snakemake, f"  Saved {cmbk_apod_path}")
snakemake_log(snakemake, f"  Saved {overlap_apod_path}")
