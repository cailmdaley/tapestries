# --- maps.smk: Map building rules ---
# Shear/density maps stored as per-method pickle (all bins inside).
# Masks stay as dense HEALPix FITS (NaMaster idiom).


rule catalog_to_map:
    """Build all tomographic shear maps for one method → single pickle."""
    input:
        catalog=catalog_dir / f"{vcat}.parquet",
    output:
        maps_pkl=map_dir / "wl_{method}" / f"{vcat}_{{method}}_maps.pkl",
    wildcard_constraints:
        method="lensmc|metacal",
    log:
        f"workflow/logs/catalog_to_map_{vcat}_{{method}}.log",
    params:
        method="{method}",
    resources:
        mem_mb=24000,
    script:
        "../scripts/build_maps.py"


rule build_density_map:
    """Build all tomographic density maps → single pickle."""
    input:
        catalog=gc_catalog_dir / f"{gc_catalog}.parquet",
    output:
        maps_pkl=map_dir / "gc" / f"{gc_catalog}_density_maps.pkl",
    log:
        f"workflow/logs/build_density_map_{gc_catalog}.log",
    resources:
        mem_mb=24000,
    script:
        "../scripts/build_density_maps.py"


rule build_rr2_apodized_masks:
    """Build official apodized RR2 footprint masks for tracer combinations."""
    input:
        shear_pkl=map_dir / "wl_lensmc" / f"{vcat}_lensmc_maps.pkl",
        density_pkl=map_dir / "gc" / f"{gc_catalog}_density_maps.pkl",
        act_mask=data_dir / cmb_experiments["act"]["mask"],
        spt_mask=data_dir / cmb_experiments["spt_winter_gmv"]["mask"],
    output:
        shear_apod=out_dir / "masks" / "rr2_apodized" / "shear_apod_mask.fits",
        density_apod=out_dir / "masks" / "rr2_apodized" / "density_apod_mask.fits",
        act_apod=out_dir / "masks" / "rr2_apodized" / "act_apod_mask.fits",
        spt_apod=out_dir / "masks" / "rr2_apodized" / "spt_apod_mask.fits",
        shear_x_density_apod=out_dir / "masks" / "rr2_apodized" / "shear_x_density_apod_mask.fits",
        shear_x_act_apod=out_dir / "masks" / "rr2_apodized" / "shear_x_act_apod_mask.fits",
        shear_x_spt_apod=out_dir / "masks" / "rr2_apodized" / "shear_x_spt_apod_mask.fits",
        density_x_act_apod=out_dir / "masks" / "rr2_apodized" / "density_x_act_apod_mask.fits",
        density_x_spt_apod=out_dir / "masks" / "rr2_apodized" / "density_x_spt_apod_mask.fits",
        act_x_spt_apod=out_dir / "masks" / "rr2_apodized" / "act_x_spt_apod_mask.fits",
        summary=out_dir / "masks" / "rr2_apodized" / "summary.json",
    log:
        f"workflow/logs/build_rr2_apodized_masks_{vcat}.log",
    params:
        nside=config["nside"],
        aposcale=config["cross_correlation"]["aposcale"],
    resources:
        mem_mb=24000,
    script:
        "../scripts/build_rr2_apodized_masks.py"


rule build_maps_fiducial:
    input:
        shear=[
            map_dir / f"wl_{method}" / f"{vcat}_{method}_maps.pkl"
            for method in shear_methods
        ],
        density=map_dir / "gc" / f"{gc_catalog}_density_maps.pkl",
