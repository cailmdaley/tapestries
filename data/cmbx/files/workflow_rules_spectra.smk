# --- spectra.smk: Cross-spectrum computation rules ---

rule build_apodized_pair_masks:
    """Build official apodized masks for each Euclid x CMB tracer pair."""
    input:
        unpack(get_euclid_map_files_always_map),
        unpack(get_cmbk_files),
    output:
        euclid_apod_mask=temp(out_dir / "masks" / "apodized_pairs" / "{method}_bin{bin}_x_{cmbk}" / "euclid_apod_mask.fits"),
        cmbk_apod_mask=temp(out_dir / "masks" / "apodized_pairs" / "{method}_bin{bin}_x_{cmbk}" / "cmbk_apod_mask.fits"),
        overlap_apod_mask=temp(out_dir / "masks" / "apodized_pairs" / "{method}_bin{bin}_x_{cmbk}" / "overlap_apod_mask.fits"),
        mask_summary=out_dir / "masks" / "apodized_pairs" / "{method}_bin{bin}_x_{cmbk}" / "summary.json",
    log:
        f"workflow/logs/build_apodized_pair_masks_{vcat}_{{method}}_bin{{bin}}_x_{{cmbk}}.log",
    params:
        aposcale=config["cross_correlation"]["aposcale"],
        nside=config["nside"],
        bin_idx=lambda w: _bin_to_idx(w.bin),
    resources:
        mem_mb=8000,
    script:
        "../scripts/build_apodized_pair_masks.py"


rule compute_cross_spectrum:
    input:
        # Production spectra are map-estimator by construction.
        # Catalog estimator is restricted to compute_catalog_cross_spectrum.
        unpack(get_euclid_map_files_always_map),
        unpack(get_cmbk_files),
        euclid_apod_mask=out_dir / "masks" / "apodized_pairs" / "{method}_bin{bin}_x_{cmbk}" / "euclid_apod_mask.fits",
        cmbk_apod_mask=out_dir / "masks" / "apodized_pairs" / "{method}_bin{bin}_x_{cmbk}" / "cmbk_apod_mask.fits",
        overlap_apod_mask=out_dir / "masks" / "apodized_pairs" / "{method}_bin{bin}_x_{cmbk}" / "overlap_apod_mask.fits",
        mask_summary=out_dir / "masks" / "apodized_pairs" / "{method}_bin{bin}_x_{cmbk}" / "summary.json",
    output:
        npz=out_dir / "cross_correlations" / "individual" / "{method}_bin{bin}_x_{cmbk}_cls.npz",
        evidence=out_dir / "cross_correlations" / "individual" / "{method}_bin{bin}_x_{cmbk}_evidence.json",
    log:
        f"workflow/logs/compute_cross_spectrum_{vcat}_{{method}}_bin{{bin}}_x_{{cmbk}}.log",
    params:
        nside=config["nside"],
        lmin=config["cross_correlation"]["binning"]["lmin"],
        lmax=config["cross_correlation"]["binning"]["lmax"],
        nells=config["cross_correlation"]["binning"]["nells"],
        aposcale=config["cross_correlation"]["aposcale"],
        n_iter=config["cross_correlation"]["n_iter"],
        wksp_cache=str(WORKSPACE_DIR),
        compute_cov=config["cross_correlation"]["covariance"]["compute"],
        estimator="map",
        bad_tiles=config.get("quality_cuts", {}).get("bad_tiles", []),
        bin_idx=lambda w: _bin_to_idx(w.bin),
        # Theory+noise guess spectra for Gaussian covariance (optional, runtime check).
        # Passed as params (not inputs) to avoid circular DAG dependency.
        theory_npz=lambda w: str(out_dir / "theory" / f"theory_cls_{w.method}_x_{w.cmbk}.npz"),
        cmb_noise_curve=lambda w: get_cmb_noise_curve_path(w.cmbk),
    threads: 16
    resources:
        mem_mb=lambda w: 16000 if w.bin == "all" else 8000,
    script:
        "../scripts/compute_cross_spectrum.py"


rule combine_shear_cross_spectra:
    input:
        npz_files=expand(
            out_dir / "cross_correlations" / "individual" / "{method}_bin{bin}_x_{{cmbk}}_cls.npz",
            method=shear_methods,
            bin=all_bins,
        ),
        nz_file=data_dir / NZ_FILES["wl"],
    output:
        sacc=out_dir / "cross_correlations" / "combined" / f"euclid_{{cmbk}}_shear_x_cmbk_cls_nside{config['nside']}.sacc",
    log:
        f"workflow/logs/combine_shear_cross_spectra_{vcat}_{{cmbk}}.log",
    params:
        description=lambda w: f"Euclid RR2 shear x {w.cmbk.upper()} CMB lensing cross-correlations",
        methods="lensmc, metacal",
        bins="1-6 (individual), all (combined)",
    script:
        "../scripts/combine_cross_spectra.py"


rule combine_density_cross_spectra:
    input:
        npz_files=expand(
            out_dir / "cross_correlations" / "individual" / "density_bin{bin}_x_{{cmbk}}_cls.npz",
            bin=gc_tom_bins,
        ),
        nz_file=data_dir / NZ_FILES["gc"],
    output:
        sacc=out_dir / "cross_correlations" / "combined" / f"euclid_{{cmbk}}_density_x_cmbk_cls_nside{config['nside']}.sacc",
    log:
        f"workflow/logs/combine_density_cross_spectra_{vcat}_{{cmbk}}.log",
    params:
        description=lambda w: f"Euclid RR2 galaxy density x {w.cmbk.upper()} CMB lensing cross-correlations",
        methods="density",
        bins="1-6",
    script:
        "../scripts/combine_cross_spectra.py"


rule compute_all_shear_cross_spectra:
    """Compute all shear x CMB-k cross-spectra across all CMB experiments."""
    input:
        expand(
            out_dir / "cross_correlations" / "individual" / "{method}_bin{bin}_x_{cmbk}_cls.npz",
            method=shear_methods,
            bin=all_bins,
            cmbk=cmbk_list,
        ),


rule compute_all_density_cross_spectra:
    """Compute all density x CMB-k cross-spectra across all CMB experiments."""
    input:
        expand(
            out_dir / "cross_correlations" / "individual" / "density_bin{bin}_x_{cmbk}_cls.npz",
            bin=gc_tom_bins,
            cmbk=cmbk_baseline,
        ),


rule compute_all_cross_spectra:
    """Compute ALL cross-spectra across all CMB experiments."""
    input:
        rules.compute_all_shear_cross_spectra.input,


rule combine_all_cross_spectra:
    """Combine all cross-spectra into SACC files."""
    input:
        expand(
            rules.combine_shear_cross_spectra.output,
            cmbk=cmbk_list,
        ),
        expand(
            rules.combine_density_cross_spectra.output,
            cmbk=cmbk_baseline,
        ),
