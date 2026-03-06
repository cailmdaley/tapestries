"""Build galaxy density overdensity maps from the GC catalog.

Outputs one pickle containing all tomographic bins.  Format matches
dr1_cmbx.eDR1data.maps.gc_cat2map():

    {"7bin": {"bin0": {...}, "bin1": {...}, ..., "bin6": {...}}}

Bins 0-5 are tomographic bins 1-6; bin6 is the "all" (unbinned) sample.
Each bin dict is the output of process_gc_catalog() (sparse ipix, map,
mask, noise_cl, fsky, nmean_srad, counts_per_pixel) plus n_gal and
area_sr for convenience.
"""

import pickle
from pathlib import Path

import healpy as hp
import numpy as np
import polars as pl

from dr1_cmbx.eDR1data.maps import process_gc_catalog
from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log


snakemake = snakemake  # type: ignore # noqa: F821

catalog_path = Path(snakemake.input.catalog)
nside = int(snakemake.config["nside"])
bad_tiles = snakemake.config.get("quality_cuts", {}).get("bad_tiles", [])
tom_bins = [int(b) for b in snakemake.config["tomography"]["bin_ids"]]
output_pkl = Path(snakemake.output.maps_pkl)

snakemake_log(
    snakemake,
    f"Building density maps for bins={tom_bins}+all, nside={nside}",
)

# ---------------------------------------------------------------------------
# Load GC catalog, apply quality cuts once
lf = pl.scan_parquet(catalog_path)

if bad_tiles:
    n_before = lf.select(pl.len()).collect().item()
    lf = lf.filter(~pl.col("tile_index").is_in(bad_tiles))
    n_after = lf.select(pl.len()).collect().item()
    snakemake_log(
        snakemake,
        f"Filtered {n_before - n_after:,} from {len(bad_tiles)} bad tiles",
    )

columns = ["right_ascension", "declination", "pos_tom_bin_id"]
full_table = lf.select(columns).drop_nulls().collect()
snakemake_log(snakemake, f"Total galaxies: {full_table.height:,}")

npix = hp.nside2npix(nside)


def _build_bin(ra, dec, label):
    """Build overdensity map for one bin, return augmented dict."""
    n_gal = len(ra)
    ipix = hp.ang2pix(nside, ra, dec, lonlat=True)

    # Binary selection mask from occupied pixels
    sel_mask = np.zeros(npix, dtype=float)
    sel_mask[np.unique(ipix)] = 1.0

    cat_dict = {"ra": ra, "dec": dec}
    result = process_gc_catalog(cat_dict, sel_mask, nside)

    # Add convenience metadata
    result["n_gal"] = n_gal
    result["area_sr"] = float(
        (sel_mask > 0).sum() * hp.nside2pixarea(nside)
    )

    snakemake_log(
        snakemake,
        f"  {label}: {n_gal:,} gals, "
        f"noise_cl={result['noise_cl']:.4e}, "
        f"fsky={result['fsky']:.6f}",
    )
    return result


# ---------------------------------------------------------------------------
# Process each tomographic bin + "all"
n_bins = len(tom_bins) + 1
bin_key = f"{n_bins}bin"
output = {bin_key: {}}

for i, tom_bin in enumerate(tom_bins):
    bin_table = full_table.filter(pl.col("pos_tom_bin_id") == tom_bin)
    if bin_table.height == 0:
        raise RuntimeError(f"No galaxies for tom_bin={tom_bin}")
    ra = bin_table["right_ascension"].to_numpy()
    dec = bin_table["declination"].to_numpy()
    output[bin_key][f"bin{i}"] = _build_bin(ra, dec, f"bin{i} (tom={tom_bin})")

# "all" bin → last index
ra_all = full_table["right_ascension"].to_numpy()
dec_all = full_table["declination"].to_numpy()
all_idx = len(tom_bins)
output[bin_key][f"bin{all_idx}"] = _build_bin(
    ra_all, dec_all, f"bin{all_idx} (all)"
)

# ---------------------------------------------------------------------------
# Write pickle
output_pkl.parent.mkdir(parents=True, exist_ok=True)
with open(output_pkl, "wb") as f:
    pickle.dump(output, f)

snakemake_log(snakemake, f"Wrote {output_pkl} ({n_bins} bins)")
