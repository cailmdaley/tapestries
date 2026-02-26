"""Build per-method jackknife null shear map pickles.

Each galaxy's corrected ellipticity (e1, e2) is multiplied by a random
sign (+1 or -1), destroying the coherent cosmological signal while
preserving noise properties. The resulting null maps have:
  - Zero signal in expectation (cosmological shear cancels)
  - Same noise as the original (same galaxies, same weights)
  - Same pixel coverage and weight structure

Cross-correlating null maps with CMB κ tests for galaxy-side systematics
that survive random splitting — e.g., PSF residuals, selection effects,
calibration errors correlated with sky position.

Output format matches build_maps.py: one pickle per method, all bins nested.
The sign flip is applied per-galaxy BEFORE process_wl_catalog, ensuring
consistent signs across bins (each galaxy keeps its sign in the "all" bin).
"""

import pickle
from pathlib import Path

import healpy as hp
import numpy as np
import polars as pl

from dr1_cmbx.eDR1data.maps import process_wl_catalog
from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log


snakemake = snakemake  # type: ignore # noqa: F821

method = snakemake.params.method
nside = int(snakemake.config["nside"])
bad_tiles = snakemake.config.get("quality_cuts", {}).get("bad_tiles", [])
tom_bins = [int(b) for b in snakemake.config["tomography"]["bin_ids"]]
JK_SEED = 42

catalog_path = Path(snakemake.input.catalog)
output_pkl = Path(snakemake.output.null_maps_pkl)

snakemake_log(
    snakemake,
    f"Jackknife null maps: method={method}, "
    f"bins={tom_bins}+all, nside={nside}, seed={JK_SEED}",
)

# ---------------------------------------------------------------------------
# Load catalog, apply quality cuts once (same as build_maps.py)
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

full_table = (
    lf.filter(pl.col(w_col) > 0).select(columns).drop_nulls().collect()
)
snakemake_log(snakemake, f"Total galaxies with weight > 0: {full_table.height:,}")

# ---------------------------------------------------------------------------
# Generate random signs ONCE for all galaxies (consistent across bins)
rng = np.random.default_rng(JK_SEED)
signs = 2 * rng.integers(0, 2, size=full_table.height) - 1
n_pos = int((signs > 0).sum())
n_neg = int((signs < 0).sum())
snakemake_log(
    snakemake,
    f"Sign split: {n_pos:,} positive, {n_neg:,} negative "
    f"(ratio {n_pos / n_neg:.4f})",
)

# Apply sign flip to full table
full_e1 = full_table[e1_col].to_numpy() * signs
full_e2 = full_table[e2_col].to_numpy() * signs

# ---------------------------------------------------------------------------
# Process each tomographic bin + "all"
n_bins = len(tom_bins) + 1
bin_key = f"{n_bins}bin"
result = {bin_key: {}}

for i, tom_bin in enumerate(tom_bins):
    mask = (full_table["tom_bin_id"] == tom_bin).to_numpy()
    if not mask.any():
        raise RuntimeError(f"No galaxies for method={method} tom_bin={tom_bin}")

    cat_dict = {
        "ra": full_table["right_ascension"].to_numpy()[mask],
        "dec": full_table["declination"].to_numpy()[mask],
        "e1": full_e1[mask],
        "e2": full_e2[mask],
        "weight": full_table[w_col].to_numpy()[mask],
    }
    shear = process_wl_catalog(cat_dict, nside, pol_conv="EUCLID")
    result[bin_key][f"bin{i}"] = shear

    area_sr = float(len(shear["ipix"]) * hp.nside2pixarea(nside))
    snakemake_log(
        snakemake,
        f"  bin{i} (tom_bin={tom_bin}): "
        f"{int(mask.sum()):,} gals, "
        f"noise_cl={shear['noise_cl']:.4e}, "
        f"area={area_sr:.4e} sr",
    )

# "all" bin → last index
all_cat = {
    "ra": full_table["right_ascension"].to_numpy(),
    "dec": full_table["declination"].to_numpy(),
    "e1": full_e1,
    "e2": full_e2,
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
