"""Construct shear maps, weight maps, and survey masks from the RR2 catalog.

This script can be run either via Snakemake workflow or standalone via command-line.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import polars as pl

import healpy as hp

from dr1_cmbx.eDR1data.maps import write_sparse_map, process_shear_catalog


def parse_args():
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Build HEALPix shear maps from RR2 catalog"
    )
    parser.add_argument("--catalog", type=Path, required=True,
                       help="Path to catalog parquet file")
    parser.add_argument("--method", type=str, required=True,
                       choices=["lensmc", "metacal"],
                       help="Shear estimation method")
    parser.add_argument("--bin", type=str, required=True,
                       help="Tomographic bin (e.g., 'all', '1', '2', ...)")
    parser.add_argument("--nside", type=int, required=True,
                       help="HEALPix NSIDE resolution")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for map files")
    parser.add_argument("--bad-tiles", type=int, nargs="*", default=[],
                       help="List of bad tile indices to exclude")
    return parser.parse_args()


def main():
    """Main execution function supporting both Snakemake and CLI modes."""

    # Try Snakemake mode first, fall back to argparse
    try:
        from snakemake.script import snakemake

        # Snakemake mode - extract parameters from snakemake object
        catalog_path = Path(snakemake.input.catalog)
        method = snakemake.params.method
        bin_label = snakemake.wildcards.bin
        nside = int(snakemake.config["nside"])
        bad_tiles = snakemake.config.get("quality_cuts", {}).get("bad_tiles", [])

        # Output paths from Snakemake rule
        shear_output = Path(snakemake.output.shear_maps)
        sum_weights_output = Path(snakemake.output.sum_weights_map)
        neff_output = Path(snakemake.output.neff_map)

        # Logging
        log_fp = None
        if snakemake.log:
            log_path = Path(snakemake.log[0])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fp = log_path.open("w")

    except NameError:
        # Standalone CLI mode - parse arguments
        args = parse_args()

        catalog_path = args.catalog
        method = args.method
        bin_label = args.bin
        nside = args.nside
        bad_tiles = args.bad_tiles

        # Construct output paths
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build output filenames (match Snakemake pattern)
        base_name = catalog_path.stem.replace("_wl_", "_")  # Remove _wl_ from catalog name
        shear_output = output_dir / f"{base_name}_method={method}_bin={bin_label}_shear_maps.fits"
        sum_weights_output = output_dir / f"{base_name}_method={method}_bin={bin_label}_sum_weights_map.fits"
        neff_output = output_dir / f"{base_name}_method={method}_bin={bin_label}_neff_map.fits"

        # No log file in standalone mode
        log_fp = None

    # ---------------------------------------------------------------------------
    # Logging helper
    def log(message: str) -> None:
        print(message, file=sys.stdout)
        if log_fp is not None:
            print(message, file=log_fp)
            log_fp.flush()

    # ---------------------------------------------------------------------------
    # Load catalog data
    lf = pl.scan_parquet(catalog_path)

    # Apply quality cuts
    if bad_tiles:
        n_before = lf.select(pl.len()).collect().item()
        lf = lf.filter(~pl.col("tile_index").is_in(bad_tiles))
        n_after = lf.select(pl.len()).collect().item()
        log(f"Filtered {n_before - n_after:,} objects from {len(bad_tiles)} bad tiles")

    if bin_label != "all":
        lf = lf.filter(pl.col("tom_bin_id") == int(bin_label))

    columns = [
        "right_ascension", "declination",
        f"she_{method}_e1_corrected", f"she_{method}_e2_corrected", f"she_{method}_weight"
    ]

    table = lf.filter(pl.col(f"she_{method}_weight") > 0).select(columns).drop_nulls().collect()
    if table.height == 0:
        raise RuntimeError(f"No galaxies for method={method} bin={bin_label} after filtering")

    ra, dec = table["right_ascension"].to_numpy(), table["declination"].to_numpy()
    e1, e2, w = [table[f"she_{method}_{field}"].to_numpy() for field in ["e1_corrected", "e2_corrected", "weight"]]

    log(f"Loaded {len(ra):,} galaxies for method={method}, bin={bin_label}")

    shear = process_shear_catalog(
        {"ra": ra, "dec": dec, "e1": e1, "e2": e2, "weight": w},
        nside=nside,
        pol_conv="IAU",
    )
    c1, c2 = float(shear["c1_sub"]), float(shear["c2_sub"])
    log(f"Additive bias removed: c1={c1:.6e}, c2={c2:.6e}")

    # ---------------------------------------------------------------------------
    # Summary statistics from dr1_cmbx map builder outputs
    n_gal = len(w)
    w_sum, w_sq_sum = float(w.sum()), float((w**2).sum())
    area_sr = float(len(shear["ipix"]) * hp.nside2pixarea(nside))
    area_arcmin2 = area_sr * (180.0 / np.pi * 60.0) ** 2

    n_eff = float(shear["n_eff_global"])
    n_eff_per_sr = float(shear["nmean_srad"])
    n_eff_per_arcmin2 = n_eff / area_arcmin2 if area_arcmin2 > 0 else 0.0
    sigma_eps = float(shear["sigma_eps"])
    sigma_eps_comp = float(shear["sigma_eps_comp"])
    sigma_eps_sq = sigma_eps**2
    noise_cl = sigma_eps_sq / (2.0 * n_eff_per_sr) if n_eff_per_sr > 0 else 0.0

    log(
        "\n".join(
            [
                f"Summary stats:",
                f"  galaxies: {n_gal:,}",
                f"  sum(w): {w_sum:.6e}",
                f"  sum(w^2): {w_sq_sum:.6e}",
                f"  n_eff: {n_eff:.6e}",
                f"  area (sr): {area_sr:.6e}",
                f"  area (arcmin^2): {area_arcmin2:.6e}",
                f"  n_eff / sr: {n_eff_per_sr:.6e}",
                f"  n_eff / arcmin^2: {n_eff_per_arcmin2:.6e}",
                f"  sigma_eps (combined): {sigma_eps:.6e}",
                f"  sigma_eps (per-component): {sigma_eps_comp:.6e}",
                f"  expected noise C_ell: {noise_cl:.6e}",
            ]
        )
    )

    shear_pixels = shear["ipix"]
    shear_e1, shear_e2 = shear["e1"], shear["e2"]
    sum_weights_values = shear["sum_weights"]
    neff_values = shear["neff"]

    log(
        "\n".join(
            [
                f"Pixel counts:",
                f"  shear pixels: {len(shear_pixels):,}",
                f"  sum_weights pixels: {len(sum_weights_values):,}",
                f"  n_eff pixels: {len(neff_values):,}",
            ]
        )
    )

    # ---------------------------------------------------------------------------
    # Persist sparse outputs using library function with shape noise metadata
    metadata = {
        "N_GAL": n_gal,
        "N_EFF": n_eff,
        "AREA_SR": area_sr,
        "NEFF_SR": n_eff_per_sr,
        "SIG_EPS": sigma_eps,
        "SIG_COMP": sigma_eps_comp,
        "NOISE_CL": noise_cl,
        "C1_SUB": c1,
        "C2_SUB": c2,
    }

    write_sparse_map(shear_output, nside=nside, pixels=shear_pixels,
                     columns={"e1": shear_e1, "e2": shear_e2}, extra_header=metadata)
    write_sparse_map(sum_weights_output, nside=nside, pixels=shear_pixels,
                     columns={"sum_weights": sum_weights_values}, extra_header=metadata)
    write_sparse_map(neff_output, nside=nside, pixels=shear_pixels,
                     columns={"n_eff": neff_values}, extra_header=metadata)

    log("Wrote sparse shear, sum_weights, and n_eff maps with shape noise metadata")

    if log_fp is not None:
        log_fp.close()


if __name__ == '__main__':
    main()
