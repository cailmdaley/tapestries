"""Build per-method shear map pickles from the RR2 WL catalog.

Outputs one pickle per shear method (lensmc, metacal) containing all
tomographic bins.  Format matches dr1_cmbx.eDR1data.maps.wl_cat2map():

    {"7bin": {"bin0": {...}, "bin1": {...}, ..., "bin6": {...}}}

Bins 0-5 are tomographic bins 1-6; bin6 is the "all" (unbinned) sample.
Each bin dict is the output of process_wl_catalog() (sparse ipix, e1, e2,
sum_weights, neff, mask, noise_cl, fsky, etc.).

# Maps store catalog ellipticities in Euclid convention (θ from West).
# The sign conversion to NaMaster (Q=-e1, U=+e2) happens at spectrum
# time via make_field(..., pol_conv="EUCLID").
"""

import pickle
import sys
from pathlib import Path

import healpy as hp
import numpy as np
import polars as pl

from dr1_cmbx.eDR1data.maps import process_wl_catalog
from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log


snakemake = snakemake  # type: ignore # noqa: F821

catalog_path = Path(snakemake.input.catalog)
method = snakemake.params.method
nside = int(snakemake.config["nside"])
bad_tiles = snakemake.config.get("quality_cuts", {}).get("bad_tiles", [])
tom_bins = [int(b) for b in snakemake.config["tomography"]["bin_ids"]]
output_pkl = Path(snakemake.output.maps_pkl)

snakemake_log(
    snakemake,
    f"Building shear maps for method={method}, "
    f"bins={tom_bins}+all, nside={nside}",
)

# ---------------------------------------------------------------------------
# Load catalog, apply quality cuts once
lf = pl.scan_parquet(catalog_path)

if bad_tiles:
    n_before = lf.select(pl.len()).collect().item()
    lf = lf.filter(~pl.col("tile_index").is_in(bad_tiles))
    n_after = lf.select(pl.len()).collect().item()
    snakemake_log(
        snakemake,
        f"Filtered {n_before - n_after:,} objects from "
        f"{len(bad_tiles)} bad tiles",
    )

e1_col = f"she_{method}_e1_corrected"
e2_col = f"she_{method}_e2_corrected"
w_col = f"she_{method}_weight"
columns = ["right_ascension", "declination", "tom_bin_id", e1_col, e2_col, w_col]

# Filter weight > 0 and collect once (all bins share this)
full_table = (
    lf.filter(pl.col(w_col) > 0).select(columns).drop_nulls().collect()
)
snakemake_log(snakemake, f"Total galaxies with weight > 0: {full_table.height:,}")

# ---------------------------------------------------------------------------
# Process each tomographic bin + "all"
n_bins = len(tom_bins) + 1  # +1 for "all"
bin_key = f"{n_bins}bin"
result = {bin_key: {}}

# Tomographic bins 1-6 → bin0-bin5
for i, tom_bin in enumerate(tom_bins):
    bin_table = full_table.filter(pl.col("tom_bin_id") == tom_bin)
    if bin_table.height == 0:
        raise RuntimeError(
            f"No galaxies for method={method} tom_bin={tom_bin}"
        )

    cat_dict = {
        "ra": bin_table["right_ascension"].to_numpy(),
        "dec": bin_table["declination"].to_numpy(),
        "e1": bin_table[e1_col].to_numpy(),
        "e2": bin_table[e2_col].to_numpy(),
        "weight": bin_table[w_col].to_numpy(),
    }
    shear = process_wl_catalog(cat_dict, nside, pol_conv="EUCLID")
    result[bin_key][f"bin{i}"] = shear

    area_sr = float(len(shear["ipix"]) * hp.nside2pixarea(nside))
    snakemake_log(
        snakemake,
        f"  bin{i} (tom_bin={tom_bin}): "
        f"{bin_table.height:,} gals, "
        f"noise_cl={shear['noise_cl']:.4e}, "
        f"area={area_sr:.4e} sr",
    )

# "all" bin → last index
all_cat = {
    "ra": full_table["right_ascension"].to_numpy(),
    "dec": full_table["declination"].to_numpy(),
    "e1": full_table[e1_col].to_numpy(),
    "e2": full_table[e2_col].to_numpy(),
    "weight": full_table[w_col].to_numpy(),
}
shear_all = process_wl_catalog(all_cat, nside, pol_conv="EUCLID")
all_idx = len(tom_bins)
result[bin_key][f"bin{all_idx}"] = shear_all

area_sr = float(len(shear_all["ipix"]) * hp.nside2pixarea(nside))
snakemake_log(
    snakemake,
    f"  bin{all_idx} (all): "
    f"{full_table.height:,} gals, "
    f"noise_cl={shear_all['noise_cl']:.4e}, "
    f"area={area_sr:.4e} sr",
)

# ---------------------------------------------------------------------------
# Write pickle
output_pkl.parent.mkdir(parents=True, exist_ok=True)
with open(output_pkl, "wb") as f:
    pickle.dump(result, f)

snakemake_log(snakemake, f"Wrote {output_pkl} ({n_bins} bins)")
