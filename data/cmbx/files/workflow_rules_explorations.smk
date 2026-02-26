"""
Snakemake workflow for research explorations and analysis exercises.
Separate from main production pipeline for systematic investigations.
"""

localrules: compare_nz_fiducial, plot_shear_maps_fiducial, animate_shear_maps_fiducial, plot_all_shear_cross_spectra, plot_all_mass_cross_spectra, plot_all_cross_spectra, compute_all_catalog_cross_spectra, plot_all_catalog_vs_map_comparisons, plot_catalog_validation_summary, plot_all_catalog_vs_map_composites, plot_act_vs_spt_composite, plot_all_redshift_trends, plot_all_theory_comparisons, plot_all_theory_residual_analyses, plot_systematic_spectra, plot_all_systematic_spectra, plot_all_contamination_metrics, plot_contamination_metric, plot_all_contamination_composites, plot_contamination_composite, plot_contamination_overview, plot_pte_heatmap, plot_all_systematic_budgets, plot_all_deprojected_spectra, plot_all_spt_estimator_composites, compute_all_jackknife_null_spectra, plot_all_jackknife_null_tests, plot_bmode_null_test, plot_all_bmode_null_tests, compute_all_cmb_noise_curves, compute_all_sim_null_spectra, plot_all_sim_null_tests, plot_signal_overview, plot_all_signal_overviews, build_all_likelihood_inputs, evaluate_all_fiducial_likelihoods, evaluate_all_deprojected_likelihoods, compute_all_patch_cross_spectra, plot_all_spatial_variations, compute_all_overlap_cross_spectra, plot_all_ell_range_robustness_spt_variants, plot_all_ell_range_robustness, compute_all_kappa_spectra, plot_euclid_nz_overview, plot_analysis_summary, plot_method_comparison_composite, plot_catalog_vs_map_composite, plot_covariance_validation

map_dir_str = str(map_dir)

rule explore_isd_weights:
    """Generate animated visualizations of ISD correction weight iterations"""
    input:
        catalog=catalog_dir / f"{vcat}.parquet",
        isd_weights=expand(catalog_dir / "correction_weights_history_{bin_id}_bias2_PHZPTBIDCUT.npy", bin_id=tom_bins)
    output:
        gif_files=expand(out_dir / "explorations" / "isd_analysis" / "isd_weights_iterations_bin_{bin_id}.gif", bin_id=tom_bins),
        product_plots=expand(out_dir / "explorations" / "isd_analysis" / "isd_product_bin_{bin_id}.png", bin_id=tom_bins)
    log:
        "workflow/logs/explore_isd_weights.log"
    params:
        output_dir=out_dir / "explorations" / "isd_analysis"
    resources:
        runtime=600  # 10 minutes for GIF generation
    script:
        "../scripts/explore_isd_weights.py"

rule compare_nz_distributions:
    """Compare theoretical and empirical N(z) distributions"""
    input:
        catalog=catalog_dir / f"{vcat}.parquet",
        theoretical_nz=data_dir / "euclid/dndz/nz.wl6.vis245-nowht.dr1s.fits"
    output:
        weighted=out_dir / "explorations" / "nz_comparison" / "nz_comparison_weighted_{phz_col}_dz{dz}.png",
        unweighted=out_dir / "explorations" / "nz_comparison" / "nz_comparison_unweighted_{phz_col}_dz{dz}.png"
    log:
        "workflow/logs/compare_nz_distributions_{phz_col}_dz{dz}.log"
    params:
        output_dir=out_dir / "explorations" / "nz_comparison",
        binning_width=lambda w: float(w.dz) / 1000,  # Convert "010" -> 0.01, "030" -> 0.03
        phz_column=lambda w: w.phz_col.lower()
    script:
        "../scripts/compare_nz_distributions.py"

rule compare_nz_fiducial:
    input:
        expand(out_dir / "explorations" / "nz_comparison" / "nz_comparison_{weight_type}_{phz_col}_dz{dz}.png",
               weight_type=["weighted", "unweighted"],
               phz_col=["phz_median", "phz_mode_1"],
               dz=["010"]),


rule compute_global_vlims:
    """Compute global color limits across bins 1-6 for consistent scaling."""
    input:
        maps_pkl=lambda w: _shear_pkl(w.method),
    output:
        str(out_dir / "explorations" / "shear_maps" / "global_vlims_method={method}.json")
    log:
        "workflow/logs/compute_global_vlims_method={method}.log"
    params:
        nside=config["nside"],
    script:
        "../scripts/compute_global_vlims.py"


rule plot_shear_maps:
    """Create diagnostic plots for shear maps."""
    input:
        maps_pkl=lambda w: _shear_pkl(w.method),
        global_vlims=out_dir / "explorations" / "shear_maps" / "global_vlims_method={method}.json",
    output:
        sum_weights_plot=out_dir / "explorations" / "shear_maps" / "sum_weights_method={method}_bin={bin}.png",
        neff_plot=out_dir / "explorations" / "shear_maps" / "neff_method={method}_bin={bin}.png",
        e1_map=out_dir / "explorations" / "shear_maps" / "e1_method={method}_bin={bin}.png",
        e2_map=out_dir / "explorations" / "shear_maps" / "e2_method={method}_bin={bin}.png",
    log:
        "workflow/logs/plot_shear_maps_method={method}_bin={bin}.log",
    params:
        nside=config["nside"],
        bin_idx=lambda w: _bin_to_idx(w.bin),
        global_vlims=lambda wildcards, input: __import__("json").load(open(input.global_vlims)),
    script:
        "../scripts/plot_shear_maps.py"


rule plot_shear_maps_fiducial:
    input:
        expand(
            out_dir / "explorations" / "shear_maps" / "{plot_type}_method={method}_bin={bin}.png",
            plot_type=["sum_weights", "neff", "e1", "e2"],
            method=shear_methods,
            bin=all_bins,
        )


rule animate_shear_maps:
    """Create animated GIFs showing progression across tomographic bins"""
    input:
        lambda wildcards: expand(
            out_dir / "explorations" / "shear_maps" / "{{field}}_method={{method}}_bin={bin}.png",
            bin=all_bins,
        )
    output:
        out_dir / "explorations" / "shear_maps" / "animated_{field}_method={method}.gif"
    log:
        "workflow/logs/animate_shear_maps_field={field}_method={method}.log"
    script:
        "../scripts/animate_shear_maps.py"


rule animate_shear_maps_fiducial:
    input:
        expand(
            out_dir / "explorations" / "shear_maps" / "animated_{field}_method={method}.gif",
            field=["e1", "e2", "neff", "sum_weights"],
            method=shear_methods,
        )


rule plot_cross_spectrum:
    """Plot individual cross-spectrum with error bars from covariance."""
    input:
        npz=out_dir / "cross_correlations" / "individual" / "{method}_bin{bin}_x_{cmbk}_cls.npz"
    output:
        png=out_dir / "explorations" / "cross_spectra_plots" / "{method}_bin{bin}_x_{cmbk}.png"
    log:
        "workflow/logs/plot_cross_spectrum_{method}_bin{bin}_x_{cmbk}.log"
    script:
        "../scripts/plot_cross_spectrum.py"


rule plot_all_shear_cross_spectra:
    """Plot all shear x CMB-k cross-spectra across all experiments."""
    input:
        expand(
            out_dir / "explorations" / "cross_spectra_plots" / "{method}_bin{bin}_x_{cmbk}.png",
            method=shear_methods,
            bin=all_bins,
            cmbk=cmbk_list,
        )


rule plot_all_mass_cross_spectra:
    """Plot all mass x CMB-k cross-spectra across all experiments."""
    input:
        expand(
            out_dir / "explorations" / "cross_spectra_plots" / "{method}_bin{bin}_x_{cmbk}.png",
            method=["sks"],
            bin=["all"],
            cmbk=cmbk_list,
        ) + expand(
            out_dir / "explorations" / "cross_spectra_plots" / "sksp_bin{bin}_x_{cmbk}.png",
            bin=tom_bins,
            cmbk=cmbk_list,
        )


rule plot_all_cross_spectra:
    """Plot ALL cross-spectra across all experiments."""
    input:
        rules.plot_all_shear_cross_spectra.input,
        rules.plot_all_mass_cross_spectra.input


rule plot_systematic_null_tests:
    """Spectral view of systematic × tracer null test pulls (shear + density)."""
    input:
        evidence_files=expand(
            out_dir / "cross_correlations" / "systematics" / "{sysmap}_x_{method}_bin{bin}_evidence.json",
            sysmap=sysmap_list,
            method=all_methods,
            bin=all_bins,
        ),
        sys_dir=out_dir / "cross_correlations" / "systematics",
        upstream_ev=claims_dir / "compute_systematic_cross_spectrum" / "evidence.json",
    output:
        png=claims_dir / "plot_systematic_null_tests" / "systematic_null_test_summary.png",
        png_density=claims_dir / "plot_systematic_null_tests" / "systematic_null_test_density.png",
        evidence=claims_dir / "plot_systematic_null_tests" / "evidence.json",
    log:
        f"workflow/logs/plot_systematic_null_tests_{vcat}.log",
    script:
        "../scripts/plot_systematic_null_tests.py"


rule validate_sacc:
    """Validate SACC file completeness: tracers, n(z), covariance, data types."""
    input:
        sacc=out_dir / "cross_correlations" / "combined" / f"euclid_{{cmbk}}_shear_x_cmbk_cls_nside{config['nside']}.sacc",
    output:
        report=out_dir / "cross_correlations" / "combined" / "validation_{cmbk}.json",
    log:
        f"workflow/logs/validate_sacc_{vcat}_{{cmbk}}.log",
    script:
        "../scripts/validate_sacc.py"


# ---------------------------------------------------------------------------
# Catalog-based cross-spectrum validation
# ---------------------------------------------------------------------------

rule compute_catalog_cross_spectrum:
    """Catalog-based cross-spectrum using shared compute path (validation).

    Bypasses map-making entirely. Agreement with map-based spectra confirms
    both map construction and spectrum computation.
    """
    input:
        unpack(get_euclid_map_files_always_map),
        unpack(get_cmbk_files),
        catalog=catalog_dir / f"{vcat}.parquet",
    output:
        npz=out_dir / "cross_correlations" / "catalog_validation" / "{method}_bin{bin}_x_{cmbk}_cat_cls.npz",
        evidence=out_dir / "cross_correlations" / "catalog_validation" / "{method}_bin{bin}_x_{cmbk}_cat_evidence.json",
    log:
        f"workflow/logs/compute_catalog_cross_spectrum_{vcat}_{{method}}_bin{{bin}}_x_{{cmbk}}.log",
    params:
        nside=config["nside"],
        lmin=config["cross_correlation"]["binning"]["lmin"],
        lmax=config["cross_correlation"]["binning"]["lmax"],
        nells=config["cross_correlation"]["binning"]["nells"],
        aposcale=config["cross_correlation"]["aposcale"],
        n_iter=config["cross_correlation"]["n_iter"],
        wksp_cache=str(WORKSPACE_DIR),
        compute_cov=config["cross_correlation"]["covariance"]["compute"],
        estimator="catalog",
        bad_tiles=config.get("quality_cuts", {}).get("bad_tiles", []),
        bin_idx=lambda w: _bin_to_idx(w.bin),
        # Theory+noise guess spectra for Gaussian covariance (same as map rule).
        theory_npz=lambda w: str(out_dir / "theory" / f"theory_cls_{w.method}_x_{w.cmbk}.npz"),
        cmb_noise_curve=lambda w: get_cmb_noise_curve_path(w.cmbk),
    threads: 16
    resources:
        mem_mb=10000
    script:
        "../scripts/compute_cross_spectrum.py"


rule compute_all_catalog_cross_spectra:
    """All catalog-based shear cross-spectra for validation."""
    input:
        expand(
            out_dir / "cross_correlations" / "catalog_validation" / "{method}_bin{bin}_x_{cmbk}_cat_cls.npz",
            method=shear_methods,
            bin=all_bins,
            cmbk=cmbk_list,
        ),


rule plot_catalog_vs_map_comparison:
    """Overlay catalog-based and map-based cross-spectra for validation."""
    input:
        map_npz=out_dir / "cross_correlations" / "individual" / "{method}_bin{bin}_x_{cmbk}_cls.npz",
        cat_npz=out_dir / "cross_correlations" / "catalog_validation" / "{method}_bin{bin}_x_{cmbk}_cat_cls.npz",
    output:
        png=out_dir / "explorations" / "catalog_validation" / "{method}_bin{bin}_x_{cmbk}_comparison.png",
    log:
        "workflow/logs/plot_catalog_vs_map_{method}_bin{bin}_x_{cmbk}.log",
    script:
        "../scripts/plot_catalog_vs_map_comparison.py"


rule plot_all_catalog_vs_map_comparisons:
    """All catalog vs map comparison plots."""
    input:
        expand(
            out_dir / "explorations" / "catalog_validation" / "{method}_bin{bin}_x_{cmbk}_comparison.png",
            method=shear_methods,
            bin=all_bins,
            cmbk=cmbk_list,
        ),


rule plot_catalog_vs_map_composite:
    """Catalog vs map estimator comparison: all-bin E+B cross-spectra."""
    input:
        map_npz=out_dir / "cross_correlations" / "individual" / "{method}_binall_x_act_cls.npz",
        cat_npz=out_dir / "cross_correlations" / "catalog_validation" / "{method}_binall_x_act_cat_cls.npz",
    output:
        png=claims_dir / "plot_catalog_vs_map_comparison" / "catalog_vs_map_{method}.png",
        evidence=claims_dir / "plot_catalog_vs_map_comparison" / "catalog_vs_map_{method}_evidence.json",
    log:
        "workflow/logs/plot_catalog_vs_map_composite_{method}.log",
    script:
        "../scripts/plot_catalog_vs_map_composite.py"


rule plot_all_catalog_vs_map_composites:
    input:
        expand(
            claims_dir / "plot_catalog_vs_map_comparison" / "catalog_vs_map_{method}.png",
            method=["lensmc", "metacal"],
        ),


# ---------------------------------------------------------------------------
# Comparison plots: ACT vs SPT, redshift trend, method comparison
# ---------------------------------------------------------------------------

rule plot_act_vs_spt_composite:
    """ACT vs SPT composite with theory overlay (γκ + δκ, self-contained)."""
    input:
        data_dir=out_dir / "cross_correlations" / "individual",
        theory_dir=out_dir / "theory",
    output:
        png=claims_dir / "plot_act_vs_spt" / "act_vs_spt_composite.png",
        evidence=claims_dir / "plot_act_vs_spt" / "act_vs_spt_composite_evidence.json",
    log:
        "workflow/logs/plot_act_vs_spt_composite.log",
    script:
        "../scripts/plot_act_vs_spt_composite.py"


rule plot_redshift_trend:
    """All tomographic bins on one panel — signal should increase with bin number."""
    input:
        npz_files=expand(
            out_dir / "cross_correlations" / "individual" / "{{method}}_bin{bin}_x_{{cmbk}}_cls.npz",
            bin=tom_bins,
        ),
        theory_npz=out_dir / "theory" / "theory_cls_{method}_x_{cmbk}.npz",
    output:
        png=claims_dir / "plot_redshift_trend" / "redshift_trend_{method}_x_{cmbk}.png",
        evidence=claims_dir / "plot_redshift_trend" / "redshift_trend_{method}_x_{cmbk}_evidence.json",
    log:
        "workflow/logs/plot_redshift_trend_{method}_x_{cmbk}.log",
    script:
        "../scripts/plot_redshift_trend.py"


rule plot_all_redshift_trends:
    """All redshift trend plots."""
    input:
        expand(
            claims_dir / "plot_redshift_trend" / "redshift_trend_{method}_x_{cmbk}.png",
            method=shear_methods,
            cmbk=cmbk_list,
        )
        + expand(
            claims_dir / "plot_redshift_trend" / "redshift_trend_density_x_{cmbk}.png",
            cmbk=cmbk_baseline,
        ),


rule plot_method_comparison_composite:
    """Spectral overlay: lensmc vs metacal with fractional difference (6-panel grid)."""
    input:
        data_dir=out_dir / "cross_correlations" / "individual",
    output:
        png=claims_dir / "plot_method_comparison" / "method_comparison_composite.png",
        evidence=claims_dir / "plot_method_comparison" / "method_comparison_composite_evidence.json",
    log:
        "workflow/logs/plot_method_comparison_composite.log",
    script:
        "../scripts/plot_method_comparison_composite.py"


# ---------------------------------------------------------------------------
# Theory comparison
# ---------------------------------------------------------------------------

def get_theory_nz_file(wildcards):
    """Select n(z) file based on method: GC n(z) for density, WL n(z) for shear."""
    if wildcards.method == "density":
        return data_dir / NZ_FILES["gc"]
    return data_dir / NZ_FILES["wl"]


rule compute_theory_cls:
    """Compute theory C_l at Planck 2018 fiducial cosmology via eDR1like.

    For shear: C_l^{κE} (vanilla + NLA IA), C_l^{EE}, C_l^{κκ}.
    For density: C_l^{κδ} (SPV3 galaxy bias), C_l^{δδ}, C_l^{κκ}.
    """
    input:
        nz_file=get_theory_nz_file,
        data_dir=out_dir / "cross_correlations" / "individual",
    output:
        npz=out_dir / "theory" / "theory_cls_{method}_x_{cmbk}.npz",
    log:
        "workflow/logs/compute_theory_cls_{method}_x_{cmbk}.log",
    params:
        lmax=config["cross_correlation"]["binning"]["lmax"],
    resources:
        mem_mb=1000,
    script:
        "../scripts/compute_theory_cls.py"


rule plot_theory_comparison:
    """Overlay measured C_l^{kE} with theory prediction (2x3 grid, bins 1-6)."""
    input:
        theory_npz=out_dir / "theory" / "theory_cls_{method}_x_{cmbk}.npz",
        data_dir=out_dir / "cross_correlations" / "individual",
    output:
        png=claims_dir / "plot_theory_comparison" / "theory_comparison_{method}_x_{cmbk}.png",
        evidence=claims_dir / "plot_theory_comparison" / "theory_comparison_{method}_x_{cmbk}_evidence.json",
    log:
        "workflow/logs/plot_theory_comparison_{method}_x_{cmbk}.log",
    script:
        "../scripts/plot_theory_comparison.py"


rule plot_all_theory_comparisons:
    """All theory comparison plots."""
    input:
        expand(
            claims_dir / "plot_theory_comparison" / "theory_comparison_{method}_x_{cmbk}.png",
            method=shear_methods,
            cmbk=cmbk_list,
        )
        + expand(
            claims_dir / "plot_theory_comparison" / "theory_comparison_density_x_{cmbk}.png",
            cmbk=cmbk_baseline,
        ),


rule plot_theory_residual_analysis:
    """Amplitude fit and ratio diagnostic: data/theory per bandpower.

    Disentangles multiplicative calibration offsets (A_hat != 1) from
    shape discrepancies (chi2_shape fails after fitting A). Compares
    ACT and SPT residuals side by side.
    """
    input:
        theory_npzs=expand(
            out_dir / "theory" / "theory_cls_{{method}}_x_{cmbk}.npz",
            cmbk=cmbk_baseline,
        ),
        data_dir=out_dir / "cross_correlations" / "individual",
    output:
        png=claims_dir / "plot_theory_residual_analysis" / "theory_residual_{method}.png",
        evidence=claims_dir / "plot_theory_residual_analysis" / "theory_residual_{method}_evidence.json",
    log:
        "workflow/logs/plot_theory_residual_analysis_{method}.log",
    params:
        cmbk_experiments=cmbk_baseline,
    script:
        "../scripts/plot_theory_residual_analysis.py"


rule plot_all_theory_residual_analyses:
    """All theory residual analysis plots."""
    input:
        expand(
            claims_dir / "plot_theory_residual_analysis" / "theory_residual_{method}.png",
            method=all_methods,
        ),


# ---------------------------------------------------------------------------
# Systematic spectra: raw ingredients of the contamination metric
# ---------------------------------------------------------------------------

rule plot_systematic_spectra:
    """Raw C^{fS}, C^{κS}, C^{SS} spectra per systematic map.

    Shows the three cross-spectra that compose X^f_S(ℓ), one figure per
    systematic. All three tracer methods (lensmc, metacal, density) are
    overlaid in the top panel using the all-bin spectrum. C^{κS} and C^{SS}
    are shown for ACT and SPT in the middle/bottom panels.
    """
    input:
        sys_dir=out_dir / "cross_correlations" / "systematics",
        cmbk_sys_dir=out_dir / "cross_correlations" / "cmbk_systematics",
        # Ensure compute jobs have run — all-bin spectra for each method
        fS_npz=expand(
            out_dir / "cross_correlations" / "systematics" / "{{sysmap}}_x_{method}_binall_cls.npz",
            method=all_methods,
        ),
        kS_npz=expand(
            out_dir / "cross_correlations" / "cmbk_systematics" / "{{sysmap}}_x_{cmbk}_cls.npz",
            cmbk=cmbk_baseline,
        ),
    output:
        png=claims_dir / "plot_systematic_spectra" / "systematic_spectra_{sysmap}.png",
        evidence=claims_dir / "plot_systematic_spectra" / "systematic_spectra_{sysmap}_evidence.json",
    log:
        "workflow/logs/plot_systematic_spectra_{sysmap}.log",
    params:
        cmbk_experiments=cmbk_baseline,
        methods=all_methods,
    script:
        "../scripts/plot_systematic_spectra.py"


rule plot_all_systematic_spectra:
    """All systematic spectra ingredient plots (one per systematic map)."""
    input:
        expand(
            claims_dir / "plot_systematic_spectra" / "systematic_spectra_{sysmap}.png",
            sysmap=sysmap_list,
        ),


rule aggregate_systematic_spectra_evidence:
    """Aggregate systematic spectra evidence across all sysmaps."""
    input:
        evidence_files=expand(
            claims_dir / "plot_systematic_spectra" / "systematic_spectra_{sysmap}_evidence.json",
            sysmap=sysmap_list,
        ),
    output:
        evidence=claims_dir / "plot_systematic_spectra" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule plot_contamination_overview:
    """2×11 contamination overview: X^f_S per sysmap, density vs shear columns.

    Shows X = C^{κS}·C^{fS}/C^{SS} with propagated Knox error bars.
    Left column: density (ACT + SPT). Right column: shear (lensmc/metacal × ACT/SPT).
    """
    input:
        sys_dir=out_dir / "cross_correlations" / "systematics",
        cmbk_sys_dir=out_dir / "cross_correlations" / "cmbk_systematics",
        fS_npz=expand(
            out_dir / "cross_correlations" / "systematics" / "{sysmap}_x_{method}_binall_cls.npz",
            sysmap=sysmap_list, method=all_methods,
        ),
        kS_npz=expand(
            out_dir / "cross_correlations" / "cmbk_systematics" / "{sysmap}_x_{cmbk}_cls.npz",
            sysmap=sysmap_list, cmbk=cmbk_baseline,
        ),
    output:
        png=claims_dir / "plot_contamination_overview" / "contamination_overview.png",
        evidence=claims_dir / "plot_contamination_overview" / "evidence.json",
    log:
        f"workflow/logs/plot_contamination_overview_{vcat}.log",
    params:
        cmbk_experiments=cmbk_baseline,
        shear_methods=shear_methods,
        sys_map_keys=sysmap_list,
    script:
        "../scripts/plot_contamination_overview.py"


rule plot_pte_heatmap:
    """PTE heatmap: 11x18 matshow of systematic null test PTE per sysmap × tracer × bin.

    Rows = systematic maps, columns = density(1-6) + lensmc(1-6) + metacal(1-6).
    Preliminary diagnostic — error bars not yet validated.
    """
    input:
        sys_dir=out_dir / "cross_correlations" / "systematics",
        evidence_files=expand(
            out_dir / "cross_correlations" / "systematics" / "{sysmap}_x_{method}_bin{bin}_evidence.json",
            sysmap=sysmap_list, method=all_methods, bin=tom_bins,
        ),
    output:
        png=claims_dir / "plot_pte_heatmap" / "pte_heatmap.png",
        evidence=claims_dir / "plot_pte_heatmap" / "evidence.json",
    log:
        f"workflow/logs/plot_pte_heatmap_{vcat}.log",
    params:
        methods=all_methods,
        sys_map_keys=sysmap_list,
        tom_bins=tom_bins,
    script:
        "../scripts/plot_pte_heatmap.py"


# ---------------------------------------------------------------------------
# Contamination metric: DES×SPT-style X^f_S(ℓ) diagnostic (per-sysmap)
# ---------------------------------------------------------------------------

rule plot_contamination_metric:
    """Plot contamination metric C^{κS} * C^{fS} per systematic map.

    Proxy for X^f_S(ℓ) = C^{κS}_ℓ · C^{fS}_ℓ / C^{SS}_ℓ (2203.12440 §A.4).
    Non-zero product means the systematic correlates with both tracers.
    """
    input:
        cmbk_sys_npz=out_dir / "cross_correlations" / "cmbk_systematics" / "{sysmap}_x_{cmbk}_cls.npz",
        sys_npz_files=expand(
            out_dir / "cross_correlations" / "systematics" / "{{sysmap}}_x_{{method}}_bin{bin}_cls.npz",
            bin=tom_bins,
        ),
        signal_ref=out_dir / "cross_correlations" / "individual" / "{method}_bin3_x_{cmbk}_cls.npz",
    output:
        png=claims_dir / "plot_contamination_metric" / "contamination_{sysmap}_{method}_x_{cmbk}.png",
        evidence=claims_dir / "plot_contamination_metric" / "contamination_{sysmap}_{method}_x_{cmbk}_evidence.json",
    log:
        "workflow/logs/plot_contamination_{sysmap}_{method}_x_{cmbk}.log",
    script:
        "../scripts/plot_contamination_metric.py"


rule plot_all_contamination_metrics:
    """All contamination metric plots."""
    input:
        expand(
            claims_dir / "plot_contamination_metric" / "contamination_{sysmap}_{method}_x_{cmbk}.png",
            sysmap=sysmap_list,
            method=shear_methods,
            cmbk=cmbk_list,
        ),


# ---------------------------------------------------------------------------
# Contamination metric composite: compress 50 per-wildcard plots into 2
# ---------------------------------------------------------------------------

rule plot_contamination_composite:
    """Composite 5×2 contamination metric grid (systematics × CMB baselines).

    Replaces 50 individual per-wildcard plots with a single overview per method.
    """
    input:
        cmbk_sys_npz=expand(
            out_dir / "cross_correlations" / "cmbk_systematics" / "{sysmap}_x_{cmbk}_cls.npz",
            sysmap=["stellar_density", "galactic_extinction", "exposures", "zodiacal_light", "saa"],
            cmbk=cmbk_baseline,
        ),
        sys_npz=expand(
            out_dir / "cross_correlations" / "systematics" / "{sysmap}_x_{{method}}_bin{bin}_cls.npz",
            sysmap=["stellar_density", "galactic_extinction", "exposures", "zodiacal_light", "saa"],
            bin=tom_bins,
        ),
        signal_ref=expand(
            out_dir / "cross_correlations" / "individual" / "{{method}}_bin3_x_{cmbk}_cls.npz",
            cmbk=cmbk_baseline,
        ),
        individual_evidence=expand(
            claims_dir / "plot_contamination_metric" / "contamination_{sysmap}_{{method}}_x_{cmbk}_evidence.json",
            sysmap=sysmap_list,
            cmbk=cmbk_baseline,
        ),
    output:
        png=claims_dir / "plot_contamination_metric" / "contamination_composite_{method}.png",
        evidence=claims_dir / "plot_contamination_metric" / "contamination_composite_{method}_evidence.json",
    log:
        "workflow/logs/plot_contamination_composite_{method}.log",
    script:
        "../scripts/plot_contamination_composite.py"


rule plot_all_contamination_composites:
    """Both composite contamination metric figures."""
    input:
        expand(
            claims_dir / "plot_contamination_metric" / "contamination_composite_{method}.png",
            method=shear_methods,
        ),


# ---------------------------------------------------------------------------
# Systematic uncertainty budget: bias significance heatmap
# ---------------------------------------------------------------------------

rule plot_systematic_budget:
    """Heatmap of systematic bias significance per (systematic, bin, CMB).

    Computes S_bias = sqrt(Σ_ℓ X(ℓ)²/σ(ℓ)²) where X is the contamination
    metric and σ is Knox uncertainty. Values > 1 indicate the systematic
    contributes > 1σ of bias to the signal.
    """
    input:
        cmbk_sys_npz=expand(
            out_dir / "cross_correlations" / "cmbk_systematics" / "{sysmap}_x_{cmbk}_cls.npz",
            sysmap=sysmap_list,
            cmbk=cmbk_baseline,
        ),
        sys_npz=expand(
            out_dir / "cross_correlations" / "systematics" / "{sysmap}_x_{method}_bin{bin}_cls.npz",
            sysmap=sysmap_list,
            bin=tom_bins,
            allow_missing=True,
        ),
        signal_npz=expand(
            out_dir / "cross_correlations" / "individual" / "{{method}}_bin{bin}_x_{cmbk}_cls.npz",
            bin=tom_bins,
            cmbk=cmbk_baseline,
        ),
    output:
        png=claims_dir / "plot_systematic_budget" / "systematic_budget_{method}.png",
        evidence=claims_dir / "plot_systematic_budget" / "systematic_budget_{method}_evidence.json",
    log:
        "workflow/logs/plot_systematic_budget_{method}.log",
    params:
        sys_dir=out_dir / "cross_correlations" / "systematics",
        signal_dir=out_dir / "cross_correlations" / "individual",
        cmbk_sys_dir=out_dir / "cross_correlations" / "cmbk_systematics",
        sysmaps=sysmap_list,
        cmbk_experiments=cmbk_baseline,
    script:
        "../scripts/plot_systematic_budget.py"


rule plot_all_systematic_budgets:
    """All systematic budget heatmaps."""
    input:
        expand(
            claims_dir / "plot_systematic_budget" / "systematic_budget_{method}.png",
            method=all_methods,
        ),


# ---------------------------------------------------------------------------
# Template deprojection: subtract systematic contamination, re-fit theory
# ---------------------------------------------------------------------------

rule plot_deprojected_spectra:
    """Subtract total contamination metric from signal and re-fit amplitude.

    X_total(ℓ) = Σ_S C^{κS}_ℓ · C^{fS}_ℓ / C^{SS}_ℓ is subtracted from C^{Eκ}_ℓ.
    Theory amplitude A_hat(IA) compared before and after deprojection.
    Addresses ACT systematic bias exceeding signal in low-z bins.
    """
    input:
        signal_dir=out_dir / "cross_correlations" / "individual",
        sys_dir=out_dir / "cross_correlations" / "systematics",
        cmbk_sys_dir=out_dir / "cross_correlations" / "cmbk_systematics",
        theory_npz=out_dir / "theory" / "theory_cls_{method}_x_{cmbk}.npz",
        marg_evidence=claims_dir / "evaluate_deprojected_likelihood" / "deprojected_likelihood_{method}_x_{cmbk}_evidence.json",
    output:
        png=claims_dir / "plot_deprojected_spectra" / "deprojected_{method}_x_{cmbk}.png",
        evidence=claims_dir / "plot_deprojected_spectra" / "deprojected_{method}_x_{cmbk}_evidence.json",
    log:
        "workflow/logs/plot_deprojected_{method}_x_{cmbk}.log",
    params:
        sysmaps=sysmap_list,
    script:
        "../scripts/plot_deprojected_spectra.py"


rule plot_all_deprojected_spectra:
    """All deprojected spectra plots."""
    input:
        expand(
            claims_dir / "plot_deprojected_spectra" / "deprojected_{method}_x_{cmbk}.png",
            method=all_methods,
            cmbk=cmbk_baseline,
        ),


# ---------------------------------------------------------------------------
# SPT estimator comparison: GMV vs TT vs PP vs MV
# ---------------------------------------------------------------------------

rule plot_spt_estimator_composite:
    """Spectral ratios: SPT TT/PP/MV vs GMV baseline (6-panel grid per method)."""
    input:
        npz_files=expand(
            out_dir / "cross_correlations" / "individual" / "{{method}}_bin{bin}_x_{cmbk}_cls.npz",
            bin=all_bins,
            cmbk=spt_variants,
        ),
    output:
        png=claims_dir / "plot_spt_estimator_comparison" / "spt_estimator_composite_{method}.png",
        evidence=claims_dir / "plot_spt_estimator_comparison" / "spt_estimator_composite_{method}_evidence.json",
    log:
        "workflow/logs/plot_spt_estimator_composite_{method}.log",

    script:
        "../scripts/plot_spt_estimator_composite.py"


rule plot_all_spt_estimator_composites:
    """All method-level SPT estimator composite plots."""
    input:
        expand(
            claims_dir / "plot_spt_estimator_comparison" / "spt_estimator_composite_{method}.png",
            method=shear_methods,
        ),


# ---------------------------------------------------------------------------
# Jackknife null test: random sign flip
# ---------------------------------------------------------------------------

jk_dir = out_dir / "cross_correlations" / "jackknife_null"


rule build_jackknife_null_maps:
    """Build null shear maps via random sign flip of galaxy ellipticities.

    Destroys coherent cosmological signal while preserving noise properties.
    One pickle per method, all bins nested (matches build_maps pattern).
    """
    input:
        catalog=catalog_dir / f"{vcat}.parquet",
    output:
        null_maps_pkl=jk_dir / "maps" / "{method}_jk_null_maps.pkl",
    log:
        f"workflow/logs/build_jackknife_null_maps_{vcat}_{{method}}.log",
    params:
        method="{method}",
    resources:
        mem_mb=24000,
    script:
        "../scripts/build_jackknife_null_maps.py"


rule compute_jackknife_null_spectrum:
    """Cross-correlate jackknife null shear maps with CMB kappa.

    Under H0 (no galaxy-side systematics), the null spectrum should be
    consistent with zero. Uses pre-built apodized masks from production.

    Reuses compute_cross_spectrum.py: same input names, same params.
    """
    input:
        maps_pkl=jk_dir / "maps" / "{method}_jk_null_maps.pkl",
        cmbk_map=lambda w: data_dir / cmb_experiments[w.cmbk]["kappa_map"],
        cmbk_mask=lambda w: data_dir / cmb_experiments[w.cmbk]["mask"],
        euclid_apod_mask=out_dir / "masks" / "apodized_pairs" / "{method}_bin{bin}_x_{cmbk}" / "euclid_apod_mask.fits",
        cmbk_apod_mask=out_dir / "masks" / "apodized_pairs" / "{method}_bin{bin}_x_{cmbk}" / "cmbk_apod_mask.fits",
        overlap_apod_mask=out_dir / "masks" / "apodized_pairs" / "{method}_bin{bin}_x_{cmbk}" / "overlap_apod_mask.fits",
        mask_summary=out_dir / "masks" / "apodized_pairs" / "{method}_bin{bin}_x_{cmbk}" / "summary.json",
    output:
        npz=jk_dir / "{method}_bin{bin}_x_{cmbk}_jk_null_cls.npz",
        evidence=jk_dir / "{method}_bin{bin}_x_{cmbk}_jk_null_evidence.json",
    log:
        f"workflow/logs/compute_jackknife_null_{vcat}_{{method}}_bin{{bin}}_x_{{cmbk}}.log",
    params:
        nside=config["nside"],
        lmin=config["cross_correlation"]["binning"]["lmin"],
        lmax=config["cross_correlation"]["binning"]["lmax"],
        nells=config["cross_correlation"]["binning"]["nells"],
        aposcale=config["cross_correlation"]["aposcale"],
        n_iter=config["cross_correlation"]["n_iter"],
        wksp_cache=str(WORKSPACE_DIR),
        compute_cov=config["cross_correlation"]["covariance"]["compute"],
        bin_idx=lambda w: _bin_to_idx(w.bin),
    threads: 16
    resources:
        mem_mb=7000,
    script:
        "../scripts/compute_cross_spectrum.py"


rule plot_jackknife_null_test:
    """Summary plot: null(shear) x CMB kappa per method x CMB experiment.

    Shows all tomographic bins' null cross-spectra with Knox error bars.
    Under H0, all spectra consistent with zero.
    """
    input:
        npz_files=expand(
            jk_dir / "{{method}}_bin{bin}_x_{{cmbk}}_jk_null_cls.npz",
            bin=tom_bins,
        ),
    output:
        png=claims_dir / "plot_jackknife_null_test" / "jackknife_null_{method}_x_{cmbk}.png",
        evidence=claims_dir / "plot_jackknife_null_test" / "jackknife_null_{method}_x_{cmbk}_evidence.json",
    log:
        "workflow/logs/plot_jackknife_null_test_{method}_x_{cmbk}.log",
    script:
        "../scripts/plot_jackknife_null_test.py"


rule compute_all_jackknife_null_spectra:
    """All jackknife null cross-spectra (baseline CMB experiments)."""
    input:
        expand(
            jk_dir / "{method}_bin{bin}_x_{cmbk}_jk_null_cls.npz",
            method=shear_methods,
            bin=all_bins,
            cmbk=cmbk_baseline,
        ),


rule plot_all_jackknife_null_tests:
    """All jackknife null test summary plots."""
    input:
        expand(
            claims_dir / "plot_jackknife_null_test" / "jackknife_null_{method}_x_{cmbk}.png",
            method=shear_methods,
            cmbk=cmbk_baseline,
        ),


# ---------------------------------------------------------------------------
# B-mode null test: B×κ cross-spectrum should be consistent with zero
# ---------------------------------------------------------------------------
# Uses existing NPZ files — B-mode is cls[1] for spin-2 × spin-0.
# NaMaster Gaussian covariance BB block provides proper E→B leakage variance.
# Reference: ACT×DES (2309.04412) found PTE=0.81 with 511 sim covariance.

rule plot_bmode_null_test:
    """B-mode null test: B×κ should be zero (no physical B-mode lensing signal).

    Reads existing cross-spectrum NPZ files, extracts B-mode component and
    uses Knox BB variance from observed auto-spectra (which include E→B
    leakage). Computes chi2 for consistency with zero.
    """
    input:
        npz_files=expand(
            out_dir / "cross_correlations" / "individual" / "{{method}}_bin{bin}_x_{{cmbk}}_cls.npz",
            bin=tom_bins,
        ),
    output:
        png=claims_dir / "plot_bmode_null_test" / "bmode_null_{method}_x_{cmbk}.png",
        evidence=claims_dir / "plot_bmode_null_test" / "bmode_null_{method}_x_{cmbk}_evidence.json",
    log:
        "workflow/logs/plot_bmode_null_test_{method}_x_{cmbk}.log",
    script:
        "../scripts/plot_bmode_null_test.py"


rule plot_all_bmode_null_tests:
    """All B-mode null test summary plots."""
    input:
        expand(
            claims_dir / "plot_bmode_null_test" / "bmode_null_{method}_x_{cmbk}.png",
            method=["lensmc"],
            cmbk=cmbk_baseline,
        ),


rule aggregate_bmode_null_evidence:
    """Aggregate B-mode null test evidence."""
    input:
        individual=expand(
            claims_dir / "plot_bmode_null_test" / "bmode_null_{method}_x_{cmbk}_evidence.json",
            method=["lensmc"],
            cmbk=cmbk_baseline,
        ),
    output:
        evidence=claims_dir / "plot_bmode_null_test" / "evidence.json",
    run:
        import json
        from datetime import datetime, timezone
        from pathlib import Path

        records = []
        for p in input.individual:
            try:
                with open(p) as f:
                    records.append(json.load(f))
            except Exception:
                pass

        n_pass = sum(r.get("n_pass", 0) for r in records)
        n_fail = sum(r.get("n_fail", 0) for r in records)
        n_total = sum(r.get("n_total", 0) for r in records)
        joint_chi2 = sum(r.get("joint_chi2", 0) for r in records)
        joint_dof = sum(r.get("joint_dof", 0) for r in records)

        pngs = sorted(str(p.name) for p in Path(output.evidence).parent.glob("*.png"))
        agg = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "input": {"n_records": len(input.individual)},
            "output": {f"png_{i}": p for i, p in enumerate(pngs)},
            "params": {},
            "test": "B-mode null (B×κ consistent with zero)",
            "covariance": "Knox BB (observed auto-spectra including E→B leakage)",
            "result": "PASS" if n_fail == 0 else "FAIL",
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_total": n_total,
            "joint_chi2": round(joint_chi2, 1),
            "joint_dof": joint_dof,
            "n_records": len(records),
            "records": records,
        }

        with open(output.evidence, "w") as f:
            json.dump(agg, f, indent=2)


# ---------------------------------------------------------------------------
# CMB reconstruction noise curves (N_0) for Gaussian covariance
# ---------------------------------------------------------------------------

# SPT experiments with sim_maps get N_0 computed from simulations.
# ACT uses the published N_L file from the DR6 data release.
noise_curve_dir = out_dir / "noise_curves"

# Identify SPT experiments that have sim_maps for N_0 computation
spt_noise_experiments = {
    name: exp for name, exp in cmb_experiments.items()
    if "sim_maps" in exp
}

rule compute_cmb_noise_curve:
    """Compute CMB lensing reconstruction noise N_0(ℓ) from simulation maps.

    N_0 = mean(auto_sim) - mean(cross_sim_pairs). Isolates reconstruction noise
    by removing the common cosmological signal via cross-correlations.
    """
    input:
        sim_maps=lambda w: [data_dir / p for p in cmb_experiments[w.cmbk]["sim_maps"]],
        mask=lambda w: data_dir / cmb_experiments[w.cmbk]["mask"],
    output:
        noise_curve=noise_curve_dir / "{cmbk}_N0.txt",
    log:
        f"workflow/logs/compute_cmb_noise_curve_{vcat}_{{cmbk}}.log",
    wildcard_constraints:
        cmbk="|".join(spt_noise_experiments.keys()),
    params:
        nside=config["nside"],
        lmax=config["cross_correlation"]["binning"]["lmax"],
        aposcale=config["cross_correlation"]["aposcale"],
        n_iter=config["cross_correlation"]["n_iter"],
    threads: 16
    resources:
        mem_mb=32000,
    script:
        "../scripts/compute_cmb_noise_curve.py"


rule compute_all_cmb_noise_curves:
    """Compute N_0 for all SPT experiments with simulations."""
    input:
        expand(noise_curve_dir / "{cmbk}_N0.txt", cmbk=spt_noise_experiments.keys()),


# Helper to resolve noise curve path for any CMB experiment
def get_cmb_noise_curve(cmbk):
    """Return the noise curve path for a CMB experiment.

    ACT: static file from DR6 release (in inputs/).
    SPT: computed from simulations (in results/).
    """
    exp = cmb_experiments[cmbk]
    if "noise_curve" in exp:
        return data_dir / exp["noise_curve"]
    elif "sim_maps" in exp:
        return noise_curve_dir / f"{cmbk}_N0.txt"
    return None


# ---------------------------------------------------------------------------
# Simulation null test: cross-correlate SPT sim kappa with Euclid shear
# ---------------------------------------------------------------------------

sim_dir = out_dir / "cross_correlations" / "sim_null"


rule compute_sim_null_spectrum:
    """Cross-correlate SPT sim kappa with Euclid shear (expected null signal).

    SPT Gaussian and Agora simulations have no correlated cosmological signal
    with the real Euclid galaxy field. Non-zero cross-spectra would indicate
    spurious correlations from residual systematics, foreground contamination,
    or pipeline artifacts.

    Reuses compute_cross_spectrum.py: same masks, same parameters, different kappa map.
    Workspace cache reuse: same masks → same coupling matrix → fast recomputation.
    """
    input:
        unpack(get_euclid_map_files_always_map),
        cmbk_map=lambda w: data_dir / sim_null_config[w.cmbk]["kappa_pattern"].format(sim_id=w.sim_id),
        cmbk_mask=lambda w: data_dir / sim_null_config[w.cmbk]["mask"],
        euclid_apod_mask=out_dir / "masks" / "apodized_pairs" / "{method}_bin{bin}_x_{cmbk}" / "euclid_apod_mask.fits",
        cmbk_apod_mask=out_dir / "masks" / "apodized_pairs" / "{method}_bin{bin}_x_{cmbk}" / "cmbk_apod_mask.fits",
        overlap_apod_mask=out_dir / "masks" / "apodized_pairs" / "{method}_bin{bin}_x_{cmbk}" / "overlap_apod_mask.fits",
        mask_summary=out_dir / "masks" / "apodized_pairs" / "{method}_bin{bin}_x_{cmbk}" / "summary.json",
    output:
        npz=sim_dir / "{cmbk}" / "{method}_bin{bin}_x_{cmbk}_sim{sim_id}_cls.npz",
        evidence=sim_dir / "{cmbk}" / "{method}_bin{bin}_x_{cmbk}_sim{sim_id}_evidence.json",
    log:
        f"workflow/logs/compute_sim_null_{vcat}_{{method}}_bin{{bin}}_x_{{cmbk}}_sim{{sim_id}}.log",
    params:
        nside=config["nside"],
        lmin=config["cross_correlation"]["binning"]["lmin"],
        lmax=config["cross_correlation"]["binning"]["lmax"],
        nells=config["cross_correlation"]["binning"]["nells"],
        aposcale=config["cross_correlation"]["aposcale"],
        n_iter=config["cross_correlation"]["n_iter"],
        wksp_cache=str(WORKSPACE_DIR),
        compute_cov=config["cross_correlation"]["covariance"]["compute"],
        bin_idx=lambda w: _bin_to_idx(w.bin),
    threads: 16
    resources:
        mem_mb=7000,
    script:
        "../scripts/compute_cross_spectrum.py"


rule compute_all_sim_null_spectra:
    """All simulation null cross-spectra (lensmc only, 6 bins × 11 sims)."""
    input:
        expand(
            sim_dir / "{cmbk}" / "{method}_bin{bin}_x_{cmbk}_sim{sim_id}_cls.npz",
            method=["lensmc"],
            bin=tom_bins,
            cmbk=list(sim_null_config.keys()),
            sim_id=sim_null_config.get("spt_winter_gmv", {}).get("sim_ids", []),
        ),


rule plot_sim_null_test:
    """Summary: sim kappa × Euclid shear across N sims per bin.

    Shows per-bandpower mean and scatter across simulations. Under H0 (no
    correlated signal), each sim spectrum should be consistent with zero.
    Validates Knox covariance calibration and absence of spurious correlations.
    """
    input:
        sim_npz_files=lambda w: expand(
            sim_dir / "{{cmbk}}" / "{{method}}_bin{bin}_x_{{cmbk}}_sim{sim_id}_cls.npz",
            bin=tom_bins,
            sim_id=sim_null_config[w.cmbk]["sim_ids"],
        ),
        data_npz_files=expand(
            out_dir / "cross_correlations" / "individual" / "{{method}}_bin{bin}_x_{{cmbk}}_cls.npz",
            bin=tom_bins,
        ),
    output:
        png=claims_dir / "plot_sim_null_test" / "sim_null_{method}_x_{cmbk}.png",
        evidence=claims_dir / "plot_sim_null_test" / "sim_null_{method}_x_{cmbk}_evidence.json",
    log:
        "workflow/logs/plot_sim_null_test_{method}_x_{cmbk}.log",
    params:
        sim_ids=lambda w: sim_null_config[w.cmbk]["sim_ids"],
    script:
        "../scripts/plot_sim_null_test.py"


rule plot_all_sim_null_tests:
    """All simulation null test summary plots."""
    input:
        expand(
            claims_dir / "plot_sim_null_test" / "sim_null_{method}_x_{cmbk}.png",
            method=["lensmc"],
            cmbk=list(sim_null_config.keys()),
        ),


# ---------------------------------------------------------------------------
# Tapestry map docs
# ---------------------------------------------------------------------------

rule plot_shared_map_conventions:
    """Where ACT, SPT, and Euclid RR2 overlap on the sky."""
    input:
        act_mask=data_dir / cmb_experiments["act"]["mask"],
        spt_mask=data_dir / cmb_experiments["spt_winter_gmv"]["mask"],
        shear_pkl=_shear_pkl("lensmc"),
    output:
        png=claims_dir / "shared_map_conventions" / "shared_map_conventions.png",
        evidence=claims_dir / "shared_map_conventions" / "evidence.json",
    log:
        "workflow/logs/plot_shared_map_conventions.log",
    params:
        all_bin_idx=_bin_to_idx("all"),
    resources:
        mem_mb=16000,
    script:
        "../scripts/plot_shared_map_conventions.py"


rule plot_cmb_lensing_maps:
    """ACT+SPT CMB lensing kappa maps (Wiener-filtered) on RR2."""
    input:
        act_kappa=data_dir / cmb_experiments["act"]["kappa_map"],
        act_mask=data_dir / cmb_experiments["act"]["mask"],
        spt_kappa=data_dir / cmb_experiments["spt_winter_gmv"]["kappa_map"],
        spt_mask=data_dir / cmb_experiments["spt_winter_gmv"]["mask"],
        shear_pkl=_shear_pkl("lensmc"),
    output:
        png=claims_dir / "cmb_lensing_maps" / "cmb_lensing_maps.png",
        evidence=claims_dir / "cmb_lensing_maps" / "evidence.json",
    log:
        "workflow/logs/plot_cmb_lensing_maps.log",
    params:
        all_bin_idx=_bin_to_idx("all"),
    resources:
        mem_mb=16000,
    script:
        "../scripts/plot_cmb_lensing_maps.py"


rule plot_act_dr6_lensing:
    """ACT DR6 map docs: unmasked and apodized kappa views."""
    input:
        kappa=data_dir / cmb_experiments["act"]["kappa_map"],
        mask=data_dir / cmb_experiments["act"]["mask"],
        shear_pkl=_shear_pkl("lensmc"),
    output:
        png_map=claims_dir / "act_dr6_lensing" / "act_dr6_lensing_map.png",
        png_mask=claims_dir / "act_dr6_lensing" / "act_dr6_lensing_apodized.png",
        evidence=claims_dir / "act_dr6_lensing" / "evidence.json",
    log:
        "workflow/logs/plot_act_dr6_lensing.log",
    params:
        all_bin_idx=_bin_to_idx("all"),
        claim_id="act_dr6_lensing",
        experiment_name="ACT DR6",
        experiment_short="ACT",
    resources:
        mem_mb=12000,
    script:
        "../scripts/plot_cmb_lensing_doc.py"


rule plot_spt_winter_lensing:
    """SPT winter GMV map docs: unmasked and apodized kappa views."""
    input:
        kappa=data_dir / cmb_experiments["spt_winter_gmv"]["kappa_map"],
        mask=data_dir / cmb_experiments["spt_winter_gmv"]["mask"],
        shear_pkl=_shear_pkl("lensmc"),
    output:
        png_map=claims_dir / "spt_winter_lensing" / "spt_winter_lensing_map.png",
        png_mask=claims_dir / "spt_winter_lensing" / "spt_winter_lensing_apodized.png",
        evidence=claims_dir / "spt_winter_lensing" / "evidence.json",
    log:
        "workflow/logs/plot_spt_winter_lensing.log",
    params:
        all_bin_idx=_bin_to_idx("all"),
        claim_id="spt_winter_lensing",
        experiment_name="SPT-3G GMV",
        experiment_short="SPT",
    resources:
        mem_mb=12000,
    script:
        "../scripts/plot_cmb_lensing_doc.py"


rule plot_wiener_filter_validation:
    """Wiener filter validation: C_ℓ^map and W(ℓ) for ACT/SPT κ.

    Panel (a): ACT/SPT map auto-spectra, residuals after theory subtraction,
    and Planck 2018 C_ℓ^{κκ}. Panel (b): theory-informed Wiener filter
    W(ℓ) = C_ℓ^{κκ,theory} / (C_ℓ^{κκ,theory} + N_ℓ) using known noise curves.
    """
    input:
        act_kappa=data_dir / cmb_experiments["act"]["kappa_map"],
        act_mask=data_dir / cmb_experiments["act"]["mask"],
        spt_kappa=data_dir / cmb_experiments["spt_winter_gmv"]["kappa_map"],
        spt_mask=data_dir / cmb_experiments["spt_winter_gmv"]["mask"],
        act_noise=data_dir / cmb_experiments["act"]["noise_curve"],
        spt_noise=out_dir / "noise_curves" / "spt_winter_gmv_N0.txt",
    output:
        png=claims_dir / "wiener_filter_validation" / "wiener_filter_validation.png",
        evidence=claims_dir / "wiener_filter_validation" / "evidence.json",
    log:
        "workflow/logs/plot_wiener_filter_validation.log",
    resources:
        mem_mb=16000,
    script:
        "../scripts/plot_wiener_filter_validation.py"


rule plot_systematic_maps:
    """All VMPZ-ID systematic maps on the RR2 footprint.

    One figure per map (north/south patches). 11 individual PNGs (~1MB each)
    instead of one 13MB composite. VMPZ-ID sparse FITS at nside 16384,
    downgraded to nside 2048.
    """
    input:
        sys_maps=expand(
            data_dir / "{path}",
            path=systematic_map_paths,
        ),
        shear_pkl=_shear_pkl("lensmc"),
    output:
        pngs=expand(
            claims_dir / "plot_systematic_maps" / "systematic_map_{sysmap}.png",
            sysmap=sysmap_list,
        ),
        evidence=claims_dir / "plot_systematic_maps" / "evidence.json",
    log:
        "workflow/logs/plot_systematic_maps.log",
    params:
        all_bin_idx=_bin_to_idx("all"),
        sys_map_keys=sysmap_list,
    resources:
        mem_mb=8000,
    script:
        "../scripts/plot_systematic_maps.py"


# ---------------------------------------------------------------------------
# Signal overview (hero figure)
# ---------------------------------------------------------------------------

rule plot_signal_overview:
    """Combined figure: ACT + SPT + theory on 6 panels (one per bin).

    The hero figure for the cross-correlation analysis. Shows all measured
    cross-spectra with both CMB experiments overlaid on theory predictions.
    """
    input:
        theory_npz=expand(
            out_dir / "theory" / "theory_cls_{method}_x_{cmbk}.npz",
            cmbk=["act"],
            allow_missing=True,
        )[0],
        spectra=expand(
            out_dir / "cross_correlations" / "individual" / "{method}_bin{bin}_x_{cmbk}_cls.npz",
            bin=tom_bins,
            cmbk=cmbk_baseline,
            allow_missing=True,
        ),
        data_dir=out_dir / "cross_correlations" / "individual",
    output:
        png=claims_dir / "plot_signal_overview" / "signal_overview_{method}.png",
        evidence=claims_dir / "plot_signal_overview" / "signal_overview_{method}_evidence.json",
    log:
        "workflow/logs/plot_signal_overview_{method}.log",
    script:
        "../scripts/plot_signal_overview.py"


rule plot_all_signal_overviews:
    input:
        expand(
            claims_dir / "plot_signal_overview" / "signal_overview_{method}.png",
            method=all_methods,
        ),


rule aggregate_signal_overview_evidence:
    input:
        evidence_files=expand(
            claims_dir / "plot_signal_overview" / "signal_overview_{method}_evidence.json",
            method=all_methods,
        ),
    output:
        evidence=claims_dir / "plot_signal_overview" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


# ---------------------------------------------------------------------------
# Likelihood integration
# ---------------------------------------------------------------------------

nz_fits_wl = data_dir / NZ_FILES["wl"]


def get_likelihood_nz_file(wildcards):
    """Select n(z) file: GC for density, WL for shear methods."""
    if wildcards.method == "density":
        return data_dir / NZ_FILES["gc"]
    return nz_fits_wl


rule build_likelihood_input:
    """Build eDR1like-compatible PKL from pipeline NPZ outputs."""
    input:
        npz_files=expand(
            out_dir / "cross_correlations" / "individual" / "{method}_bin{bin}_x_{cmbk}_cls.npz",
            bin=tom_bins,
            allow_missing=True,
        ),
        nz_fits=get_likelihood_nz_file,
    output:
        pkl=out_dir / "likelihood" / "{method}_x_{cmbk}_likelihood_input.pkl",
        evidence=claims_dir / "build_likelihood_input" / "{method}_x_{cmbk}_evidence.json",
    log:
        "workflow/logs/build_likelihood_input_{method}_{cmbk}.log",
    script:
        "../scripts/build_likelihood_input.py"


rule build_all_likelihood_inputs:
    """Build likelihood inputs for all method × CMB experiment combinations."""
    input:
        expand(
            out_dir / "likelihood" / "{method}_x_{cmbk}_likelihood_input.pkl",
            method=all_methods,
            cmbk=cmbk_baseline,
        ),


rule evaluate_fiducial_likelihood:
    """Evaluate likelihood at fiducial cosmology: joint chi2, detection S/N, GOF."""
    input:
        pkl=out_dir / "likelihood" / "{method}_x_{cmbk}_likelihood_input.pkl",
        theory_npz=out_dir / "theory" / "theory_cls_{method}_x_{cmbk}.npz",
    output:
        png=claims_dir / "evaluate_fiducial_likelihood" / "likelihood_{method}_x_{cmbk}.png",
        evidence=claims_dir / "evaluate_fiducial_likelihood" / "likelihood_{method}_x_{cmbk}_evidence.json",
    log:
        "workflow/logs/evaluate_fiducial_likelihood_{method}_{cmbk}.log",
    script:
        "../scripts/evaluate_fiducial_likelihood.py"


rule evaluate_all_fiducial_likelihoods:
    """Evaluate fiducial likelihood for all method × CMB baseline combinations."""
    input:
        expand(
            claims_dir / "evaluate_fiducial_likelihood" / "likelihood_{method}_x_{cmbk}.png",
            method=all_methods,
            cmbk=cmbk_baseline,
        ),


# ---------------------------------------------------------------------------
# Deprojected likelihood evaluation
# ---------------------------------------------------------------------------

rule evaluate_deprojected_likelihood:
    """Evaluate likelihood on deprojected spectra: joint chi2 after systematic removal.

    Combines template deprojection (X_total subtraction) with full covariance
    from PKL. Gives definitive "clean" detection significance and GOF.
    """
    input:
        pkl=out_dir / "likelihood" / "{method}_x_{cmbk}_likelihood_input.pkl",
        theory_npz=out_dir / "theory" / "theory_cls_{method}_x_{cmbk}.npz",
        signal_dir=out_dir / "cross_correlations" / "individual",
        sys_dir=out_dir / "cross_correlations" / "systematics",
        cmbk_sys_dir=out_dir / "cross_correlations" / "cmbk_systematics",
    output:
        png=claims_dir / "evaluate_deprojected_likelihood" / "deprojected_likelihood_{method}_x_{cmbk}.png",
        evidence=claims_dir / "evaluate_deprojected_likelihood" / "deprojected_likelihood_{method}_x_{cmbk}_evidence.json",
    log:
        "workflow/logs/evaluate_deprojected_likelihood_{method}_{cmbk}.log",
    params:
        sysmaps=sysmap_list,
    script:
        "../scripts/evaluate_deprojected_likelihood.py"


rule evaluate_all_deprojected_likelihoods:
    """Evaluate deprojected likelihood for all method × CMB baseline combinations."""
    input:
        expand(
            claims_dir / "evaluate_deprojected_likelihood" / "deprojected_likelihood_{method}_x_{cmbk}.png",
            method=all_methods,
            cmbk=cmbk_baseline,
        ),


# ---------------------------------------------------------------------------
# Spatial variation test: north/south patch split
# ---------------------------------------------------------------------------

patch_config = config.get("spatial_patches", {})
patch_list = list(patch_config.keys())
patch_dir = out_dir / "cross_correlations" / "patches"


rule build_patch_mask:
    """Build a binary HEALPix mask for a spatial patch (RA/Dec box cut)."""
    output:
        mask=patch_dir / "masks" / "{patch}_patch_mask.fits",
    log:
        "workflow/logs/build_patch_mask_{patch}.log",
    params:
        nside=config["nside"],
        patch_config=lambda w: patch_config[w.patch],
    resources:
        mem_mb=8000,
    script:
        "../scripts/build_patch_mask.py"


rule compute_patch_cross_spectrum:
    """Cross-correlate Euclid shear with CMB kappa within a spatial patch.

    Reuses compute_cross_spectrum.py with an additional patch_mask input.
    The patch mask zeros out the weight map and CMB mask outside the patch,
    so the cross-spectrum is computed only from data within the patch.

    Always map-based (patch masking requires spatial map operations).
    Workspace caching still works — different masks produce different cache keys.
    """
    input:
        unpack(get_euclid_map_files_always_map),
        unpack(get_cmbk_files),
        patch_mask=patch_dir / "masks" / "{patch}_patch_mask.fits",
    output:
        npz=patch_dir / "{patch}" / "{method}_bin{bin}_x_{cmbk}_cls.npz",
        evidence=patch_dir / "{patch}" / "{method}_bin{bin}_x_{cmbk}_evidence.json",
    log:
        f"workflow/logs/compute_patch_cross_spectrum_{vcat}_{{patch}}_{{method}}_bin{{bin}}_x_{{cmbk}}.log",
    params:
        nside=config["nside"],
        lmin=config["cross_correlation"]["binning"]["lmin"],
        lmax=config["cross_correlation"]["binning"]["lmax"],
        nells=config["cross_correlation"]["binning"]["nells"],
        aposcale=config["cross_correlation"]["aposcale"],
        n_iter=config["cross_correlation"]["n_iter"],
        wksp_cache=str(WORKSPACE_DIR),
        compute_cov=config["cross_correlation"]["covariance"]["compute"],
    threads: 16
    resources:
        mem_mb=32000,
    script:
        "../scripts/compute_cross_spectrum.py"


rule compute_all_patch_cross_spectra:
    """All spatial patch cross-spectra: 2 patches × lensmc × 6 bins × ACT only.

    ACT-only because SPT winter field (Dec < -42) doesn't overlap the north
    patch (Dec -37 to -26.5). Both patches overlap with ACT DR6.
    """
    input:
        expand(
            patch_dir / "{patch}" / "{method}_bin{bin}_x_{cmbk}_cls.npz",
            patch=patch_list,
            method=["lensmc"],
            bin=tom_bins,
            cmbk=["act"],
        ),


rule plot_spatial_variation:
    """Compare north vs south patch cross-spectra per bin.

    Chi2 test for north-south consistency using sum of Knox variances.
    Reports both full-range and ℓ>100 chi2 (low-ℓ bandpowers have < 1 mode
    at patch fsky ~0.001, making full-range chi2 unreliable).
    """
    input:
        **{
            f"{patch}_npz_{bin_id}": patch_dir / patch / f"{{method}}_bin{bin_id}_x_{{cmbk}}_cls.npz"
            for patch in patch_list
            for bin_id in tom_bins
        },
    output:
        png=claims_dir / "plot_spatial_variation" / "spatial_variation_{method}_x_{cmbk}.png",
        evidence=claims_dir / "plot_spatial_variation" / "spatial_variation_{method}_x_{cmbk}_evidence.json",
    log:
        "workflow/logs/plot_spatial_variation_{method}_x_{cmbk}.log",
    params:
        full_npz_dir=out_dir / "cross_correlations" / "individual",
    script:
        "../scripts/plot_spatial_variation.py"


rule plot_all_spatial_variations:
    """All spatial variation comparison plots (ACT only — SPT lacks north patch overlap)."""
    input:
        expand(
            claims_dir / "plot_spatial_variation" / "spatial_variation_{method}_x_{cmbk}.png",
            method=["lensmc"],
            cmbk=["act"],
        ),


# ---------------------------------------------------------------------------
# Spatial variation: overlap-based split (ACT-only / ACT+SPT / SPT-only)
# ---------------------------------------------------------------------------

overlap_dir = out_dir / "cross_correlations" / "overlap"


rule build_overlap_masks:
    """Build binary masks for ACT-only, ACT+SPT overlap, and SPT-only regions.

    Uses pixel-level intersection of ACT mask, SPT mask, and shear weight map
    to identify three natural regions of the shear footprint. The overlap region
    enables the strongest consistency test: same galaxies, two CMB pipelines.
    """
    input:
        act_mask=data_dir / cmb_experiments["act"]["mask"],
        spt_mask=data_dir / cmb_experiments["spt_winter_gmv"]["mask"],
        shear_pkl=_shear_pkl("lensmc"),
    output:
        act_only=overlap_dir / "masks" / "act_only_mask.fits",
        overlap=overlap_dir / "masks" / "overlap_mask.fits",
        spt_only=overlap_dir / "masks" / "spt_only_mask.fits",
        summary=overlap_dir / "masks" / "overlap_summary.json",
    log:
        "workflow/logs/build_overlap_masks.log",
    params:
        bin_idx=_bin_to_idx("1"),
        nside=config["nside"],
    resources:
        mem_mb=8000,
    script:
        "../scripts/build_overlap_masks.py"


overlap_regions = ["act_only", "overlap", "spt_only"]
# Which CMB experiments can be cross-correlated in each region
overlap_cmbk = {
    "act_only": ["act"],
    "overlap": ["act", "spt_winter_gmv"],
    "spt_only": ["spt_winter_gmv"],
}


def get_overlap_mask(wildcards):
    """Return the correct overlap mask for this region."""
    return overlap_dir / "masks" / f"{wildcards.region}_mask.fits"


rule compute_overlap_cross_spectrum:
    """Cross-correlate shear × CMB κ within an overlap region.

    Reuses compute_cross_spectrum.py with the overlap mask as patch_mask.
    The overlap region (ACT ∩ SPT ∩ shear) enables the strongest consistency
    test: same galaxies, two independent CMB pipelines.
    Always map-based (patch masking requires spatial map operations).
    """
    input:
        unpack(get_euclid_map_files_always_map),
        unpack(get_cmbk_files),
        patch_mask=get_overlap_mask,
    output:
        npz=overlap_dir / "{region}" / "{method}_bin{bin}_x_{cmbk}_cls.npz",
        evidence=overlap_dir / "{region}" / "{method}_bin{bin}_x_{cmbk}_evidence.json",
    log:
        f"workflow/logs/compute_overlap_cross_spectrum_{vcat}_{{region}}_{{method}}_bin{{bin}}_x_{{cmbk}}.log",
    params:
        nside=config["nside"],
        lmin=config["cross_correlation"]["binning"]["lmin"],
        lmax=config["cross_correlation"]["binning"]["lmax"],
        nells=config["cross_correlation"]["binning"]["nells"],
        aposcale=config["cross_correlation"]["aposcale"],
        n_iter=config["cross_correlation"]["n_iter"],
        wksp_cache=str(WORKSPACE_DIR),
        compute_cov=False,
        compute_knox=True,
    threads: 16
    resources:
        mem_mb=8000,
    script:
        "../scripts/compute_cross_spectrum.py"


rule compute_all_overlap_cross_spectra:
    """All overlap cross-spectra: overlap region × both CMB × 6 bins."""
    input:
        # Overlap region: both ACT and SPT (the key consistency test)
        expand(
            overlap_dir / "overlap" / "{method}_bin{bin}_x_{cmbk}_cls.npz",
            method=["lensmc"],
            bin=tom_bins,
            cmbk=["act", "spt_winter_gmv"],
        ),
        # ACT-only region: ACT only (control)
        expand(
            overlap_dir / "act_only" / "{method}_bin{bin}_x_{cmbk}_cls.npz",
            method=["lensmc"],
            bin=tom_bins,
            cmbk=["act"],
        ),


rule plot_overlap_comparison:
    """Compare ACT vs SPT cross-spectra in the overlap region.

    6-panel grid showing both experiments on the same sky. Chi2 of the
    difference spectrum tests CMB-side consistency with identical galaxies.
    """
    input:
        overlap_npzs=expand(
            overlap_dir / "overlap" / "lensmc_bin{bin}_x_{cmbk}_cls.npz",
            bin=tom_bins,
            cmbk=["act", "spt_winter_gmv"],
        ),
        overlap_summary=overlap_dir / "masks" / "overlap_summary.json",
    output:
        png=claims_dir / "plot_overlap_comparison" / "overlap_act_vs_spt.png",
        evidence=claims_dir / "plot_overlap_comparison" / "overlap_act_vs_spt_evidence.json",
    log:
        "workflow/logs/plot_overlap_comparison.log",
    params:
        overlap_dir=overlap_dir,
    script:
        "../scripts/plot_overlap_comparison.py"


rule aggregate_overlap_evidence:
    """Wrap per-rule evidence into standardized tapestry evidence.json.

    Adds input/output/params snakemake metadata to the evidence doc for
    dashboard rendering. Required because plot_overlap_comparison writes
    a named evidence file, not evidence.json directly.
    """
    input:
        per_rule=claims_dir / "plot_overlap_comparison" / "overlap_act_vs_spt_evidence.json",
    output:
        evidence=claims_dir / "plot_overlap_comparison" / "evidence.json",
    run:
        import json
        from datetime import datetime, timezone
        from pathlib import Path

        with open(input.per_rule) as f:
            inner = json.load(f)

        doc = {
            "id": "plot_overlap_comparison",
            "generated": datetime.now(timezone.utc).isoformat(),
            "input": {
                "overlap_npzs": "results/.../overlap/lensmc_bin*_x_{cmbk}_cls.npz",
                "overlap_summary": "results/.../masks/overlap_summary.json",
            },
            "output": {
                "png": "overlap_act_vs_spt.png",
                "evidence": "evidence.json",
            },
            "params": {},
            "evidence": inner.get("evidence", {}),
        }

        Path(output.evidence).parent.mkdir(parents=True, exist_ok=True)
        with open(output.evidence, "w") as f:
            json.dump(doc, f, indent=2)


# ---------------------------------------------------------------------------
# Results table (paper compilation)
# ---------------------------------------------------------------------------

rule generate_results_table:
    """Compile all key numbers into paper-ready LaTeX tables.

    Reads evidence.json from all tapestry nodes and produces three tables:
    1. Detection & theory comparison (per method × CMB)
    2. Per-bin significance and amplitude fits (lensmc baseline)
    3. Validation tests summary
    """
    input:
        fiducial_evidence=claims_dir / "evaluate_fiducial_likelihood" / "evidence.json",
        deprojected_evidence=claims_dir / "evaluate_deprojected_likelihood" / "evidence.json",
        residual_evidence=claims_dir / "plot_theory_residual_analysis" / "evidence.json",
        overview_evidence=claims_dir / "plot_signal_overview" / "evidence.json",
        act_spt_evidence=claims_dir / "plot_act_vs_spt" / "evidence.json",
        method_evidence=claims_dir / "plot_method_comparison" / "evidence.json",
        estimator_evidence=claims_dir / "plot_spt_estimator_comparison" / "evidence.json",
        catalog_evidence=claims_dir / "plot_catalog_validation_summary" / "evidence.json",
        jackknife_evidence=claims_dir / "plot_jackknife_null_test" / "evidence.json",
        sim_null_evidence=claims_dir / "plot_sim_null_test" / "evidence.json",
        spatial_evidence=claims_dir / "plot_spatial_variation" / "evidence.json",
        systematic_evidence=claims_dir / "plot_systematic_null_tests" / "evidence.json",
        budget_evidence=claims_dir / "plot_systematic_budget" / "evidence.json",
        robustness_evidence=claims_dir / "plot_ell_range_robustness" / "evidence.json",
        bmode_evidence=claims_dir / "plot_bmode_null_test" / "evidence.json",
    output:
        evidence=claims_dir / "generate_results_table" / "evidence.json",
        tex=claims_dir / "generate_results_table" / "results_tables.tex",
    log:
        "workflow/logs/generate_results_table.log",
    params:
        claims_dir=claims_dir,
    script:
        "../scripts/generate_results_table.py"


# ---------------------------------------------------------------------------
# Analysis synthesis (one-page summary)
# ---------------------------------------------------------------------------

rule plot_analysis_summary:
    """One-page synthesis of all key findings.

    Reads evidence.json from claims directories and composes a four-panel
    summary: detection S/N, A_lens per tracer, consistency, null tests.
    """
    input:
        claims_dir=claims_dir,
        xspec_evidence=claims_dir / "compute_cross_spectrum" / "evidence.json",
        fiducial_evidence=claims_dir / "evaluate_fiducial_likelihood" / "evidence.json",
        act_spt_evidence=claims_dir / "plot_act_vs_spt" / "evidence.json",
        method_evidence=claims_dir / "plot_method_comparison" / "evidence.json",
        catalog_evidence=claims_dir / "plot_catalog_validation_summary" / "evidence.json",
        systematic_evidence=claims_dir / "plot_systematic_null_tests" / "evidence.json",
        jackknife_evidence=claims_dir / "plot_jackknife_null_test" / "evidence.json",
        sim_null_evidence=claims_dir / "plot_sim_null_test" / "evidence.json",
    output:
        png=claims_dir / "plot_analysis_summary" / "analysis_summary.png",
        evidence=claims_dir / "plot_analysis_summary" / "evidence.json",
    log:
        "workflow/logs/plot_analysis_summary.log",
    params:
        xspec_dir=out_dir / "cross_correlations" / "individual",
    script:
        "../scripts/plot_analysis_summary.py"


# ---------------------------------------------------------------------------
# ℓ-range robustness test
# ---------------------------------------------------------------------------

rule plot_ell_range_robustness:
    """Amplitude stability under progressive ℓ_max cuts.

    For each bin × CMB, progressively includes bandpowers up to ℓ_max
    and fits NLA IA theory amplitude. Stable A_hat indicates robustness;
    trends reveal scale-dependent biases (baryons, contamination, filtering).
    """
    wildcard_constraints:
        method="lensmc|metacal",
    input:
        theory_npzs=expand(
            out_dir / "theory" / "theory_cls_{{method}}_x_{cmbk}.npz",
            cmbk=cmbk_baseline,
        ),
        data_dir=out_dir / "cross_correlations" / "individual",
    output:
        png=claims_dir / "plot_ell_range_robustness" / "ell_range_robustness_{method}.png",
        evidence=claims_dir / "plot_ell_range_robustness" / "ell_range_robustness_{method}_evidence.json",
    log:
        "workflow/logs/plot_ell_range_robustness_{method}.log",
    params:
        cmbk_experiments=cmbk_baseline,
    script:
        "../scripts/plot_ell_range_robustness.py"


rule plot_ell_range_robustness_spt_variants:
    """ℓ-range robustness for SPT estimator variants (TT, PP, MV vs GMV).

    Same progressive ℓ_max amplitude fit but comparing across all SPT
    estimators. Reveals whether scale-dependent deficit is driven by
    temperature (TT), polarization (PP), or both — constraining origin
    (foregrounds vs transfer function vs noise bias).
    """
    wildcard_constraints:
        method="lensmc|metacal",
    input:
        theory_npzs=expand(
            out_dir / "theory" / "theory_cls_{{method}}_x_{cmbk}.npz",
            cmbk=spt_variants,
        ),
        data_dir=out_dir / "cross_correlations" / "individual",
    output:
        png=claims_dir / "plot_ell_range_robustness" / "ell_range_robustness_{method}_spt_variants.png",
        evidence=claims_dir / "plot_ell_range_robustness" / "ell_range_robustness_{method}_spt_variants_evidence.json",
    log:
        "workflow/logs/plot_ell_range_robustness_{method}_spt_variants.log",
    params:
        cmbk_experiments=spt_variants,
    script:
        "../scripts/plot_ell_range_robustness.py"


rule plot_all_ell_range_robustness_spt_variants:
    """All SPT variant ℓ-range robustness plots."""
    input:
        expand(
            claims_dir / "plot_ell_range_robustness" / "ell_range_robustness_{method}_spt_variants.png",
            method=["lensmc"],
        ),


rule plot_all_ell_range_robustness:
    """All ℓ-range robustness plots."""
    input:
        expand(
            claims_dir / "plot_ell_range_robustness" / "ell_range_robustness_{method}.png",
            method=shear_methods,
        ),


rule aggregate_ell_range_robustness_evidence:
    """Aggregate ℓ-range robustness evidence across methods."""
    input:
        individual=expand(
            claims_dir / "plot_ell_range_robustness" / "ell_range_robustness_{method}_evidence.json",
            method=shear_methods,
        ),
        upstream_ev=claims_dir / "compute_cross_spectrum" / "evidence.json",
    output:
        evidence=claims_dir / "plot_ell_range_robustness" / "evidence.json",
    run:
        import json
        from datetime import datetime, timezone
        records = []
        for f in input.individual:
            with open(f) as fh:
                records.append(json.load(fh))
        # Merge evidence from both methods
        merged = {}
        for rec in records:
            for cmbk, cmbk_ev in rec.get("evidence", {}).items():
                key = f"{rec['id'].split('_')[-1]}_{cmbk}"
                merged[key] = {
                    "worst_max_deviation_sigma": cmbk_ev.get("worst_max_deviation_sigma"),
                    "worst_bin": cmbk_ev.get("worst_bin"),
                    "median_A_range": cmbk_ev.get("median_A_range"),
                }
        # Overall stability
        all_devs = [v["worst_max_deviation_sigma"] for v in merged.values() if v.get("worst_max_deviation_sigma") is not None]
        evidence = {
            "id": "plot_ell_range_robustness",
            "generated": datetime.now(timezone.utc).isoformat(),
            "evidence": {
                "n_methods": len(records),
                "overall_worst_deviation_sigma": round(max(all_devs), 2) if all_devs else None,
                "overall_median_deviation_sigma": round(float(__import__('numpy').median(all_devs)), 2) if all_devs else None,
                "per_method_cmb": merged,
            },
            "artifacts": {"individual_evidence": [str(f) for f in input.individual]},
        }
        with open(output.evidence, "w") as fh:
            json.dump(evidence, fh, indent=2)


# ---------------------------------------------------------------------------
# CMB lensing auto/cross-spectrum (κ × κ) — third probe of 3×2pt
# ---------------------------------------------------------------------------

# Pairs: auto-spectra (same experiment) need N0 debiasing;
# cross-spectrum (different experiments) is noise-free.
kappa_pairs = [
    ("act", "act"),
    ("spt_winter_gmv", "spt_winter_gmv"),
    ("act", "spt_winter_gmv"),
]

kappa_dir = out_dir / "kappa_spectra"


def get_kappa_files(wildcards, which):
    """Get CMB κ map and mask for a kappa spectrum pair."""
    cmbk = getattr(wildcards, which)
    exp = cmb_experiments[cmbk]
    return {
        f"kappa{which[-1]}": data_dir / exp["kappa_map"],
        f"mask{which[-1]}": data_dir / exp["mask"],
    }


# ---------------------------------------------------------------------------
# κ × κ auto/cross-spectra (DESCOPED)
# Requires RDN0 + N₁ debiasing we don't have. The 3×2pt data vector uses
# only γκ and δκ cross-correlations; κκ enters at the likelihood level via
# published noise curves. See fiber: descope-cmb-lensing-32af86ea
# ---------------------------------------------------------------------------

rule compute_kappa_spectrum:
    """Compute κ × κ auto/cross-spectrum via NaMaster.

    DESCOPED: kept for future reference when RDN0+N₁ debiasing is available.
    For auto-spectra, subtracts N0 reconstruction noise.
    For cross-spectra, noise is independent — no debiasing needed.
    """
    input:
        kappa1=lambda w: data_dir / cmb_experiments[w.cmbk1]["kappa_map"],
        mask1=lambda w: data_dir / cmb_experiments[w.cmbk1]["mask"],
        kappa2=lambda w: data_dir / cmb_experiments[w.cmbk2]["kappa_map"],
        mask2=lambda w: data_dir / cmb_experiments[w.cmbk2]["mask"],
    output:
        npz=kappa_dir / "{cmbk1}_x_{cmbk2}_cls.npz",
        evidence=kappa_dir / "{cmbk1}_x_{cmbk2}_evidence.json",
    log:
        f"workflow/logs/compute_kappa_spectrum_{vcat}_{{cmbk1}}_x_{{cmbk2}}.log",
    params:
        nside=config["nside"],
        lmin=config["cross_correlation"]["binning"]["lmin"],
        lmax=config["cross_correlation"]["binning"]["lmax"],
        nells=config["cross_correlation"]["binning"]["nells"],
        aposcale=config["cross_correlation"]["aposcale"],
        n_iter=config["cross_correlation"]["n_iter"],
        wksp_cache=str(WORKSPACE_DIR),
        n0_curve1=lambda w: get_cmb_noise_curve(w.cmbk1),
        n0_curve2=lambda w: get_cmb_noise_curve(w.cmbk2),
        # Theory C_ℓ^{κκ} from any theory NPZ (all have same Planck 2018 κκ)
        theory_npz=str(out_dir / "theory" / "theory_cls_lensmc_x_act.npz"),
    threads: 16
    resources:
        mem_mb=16000,
    script:
        "../scripts/compute_kappa_spectrum.py"


rule compute_all_kappa_spectra:
    """All κ × κ spectra: auto + cross."""
    input:
        expand(
            kappa_dir / "{cmbk1}_x_{cmbk2}_cls.npz",
            zip,
            cmbk1=[p[0] for p in kappa_pairs],
            cmbk2=[p[1] for p in kappa_pairs],
        ),


rule plot_kappa_spectrum:
    """Plot κ × κ auto/cross-spectra vs Planck 2018 theory.

    Three-panel figure: ACT auto, ACT×SPT cross, SPT auto.
    Auto-spectra show N0-debiased measurements. Knox error bars.
    """
    input:
        spectra=expand(
            kappa_dir / "{cmbk1}_x_{cmbk2}_cls.npz",
            zip,
            cmbk1=[p[0] for p in kappa_pairs],
            cmbk2=[p[1] for p in kappa_pairs],
        ),
    output:
        png=claims_dir / "plot_kappa_spectrum" / "kappa_spectrum.png",
        evidence=claims_dir / "plot_kappa_spectrum" / "kappa_spectrum_evidence.json",
    log:
        "workflow/logs/plot_kappa_spectrum.log",
    script:
        "../scripts/plot_kappa_spectrum.py"


rule aggregate_kappa_spectrum_evidence:
    """Aggregate κ × κ evidence."""
    input:
        evidence_files=[
            claims_dir / "plot_kappa_spectrum" / "kappa_spectrum_evidence.json",
        ],
        individual=expand(
            kappa_dir / "{cmbk1}_x_{cmbk2}_evidence.json",
            zip,
            cmbk1=[p[0] for p in kappa_pairs],
            cmbk2=[p[1] for p in kappa_pairs],
        ),
    output:
        evidence=claims_dir / "plot_kappa_spectrum" / "evidence.json",
    run:
        import json
        from datetime import datetime, timezone
        records = []
        for f in input.individual:
            with open(f) as fh:
                records.append(json.load(fh))
        # Plot evidence
        with open(input.evidence_files[0]) as fh:
            plot_ev = json.load(fh)
        # Merge
        evidence = {
            "id": "plot_kappa_spectrum",
            "generated": datetime.now(timezone.utc).isoformat(),
            "n_spectra": len(records),
            "spectra": {},
            "artifacts": plot_ev.get("artifacts", {}),
        }
        for rec in records:
            key = rec["id"]
            ev = rec.get("evidence", {})
            evidence["spectra"][key] = {
                "is_auto": rec.get("is_auto", False),
                "snr_knox": ev.get("snr_knox"),
                "A_lens": ev.get("A_lens"),
                "sigma_A": ev.get("sigma_A"),
                "chi2_theory": ev.get("chi2_theory"),
                "pte_theory": ev.get("pte_theory"),
                "fsky_cross": ev.get("fsky_cross"),
            }
        with open(output.evidence, "w") as fh:
            json.dump(evidence, fh, indent=2)


rule plot_covariance_validation:
    """Covariance validation: NaMaster Gaussian vs Knox formula.

    Compares the two independent covariance estimates for density cross-spectra
    (where both are available). NaMaster captures mask mode-coupling that Knox
    ignores; their agreement validates the error bars used throughout.
    """
    input:
        cross_dir=out_dir / "cross_correlations" / "individual",
        spectra=expand(
            out_dir / "cross_correlations" / "individual" / "density_bin{bin}_x_{cmbk}_cls.npz",
            bin=tom_bins,
            cmbk=cmbk_baseline,
        ),
    output:
        png=claims_dir / "plot_covariance_validation" / "covariance_validation.png",
        evidence=claims_dir / "plot_covariance_validation" / "evidence.json",
    params:
        cmbk_baseline=cmbk_baseline,
    log:
        "workflow/logs/plot_covariance_validation.log",
    script:
        "../scripts/plot_covariance_validation.py"


# ---------------------------------------------------------------------------
# Coupling matrix conditioning diagnostic
# ---------------------------------------------------------------------------

rule diagnose_coupling_matrix:
    """Diagnose NaMaster coupling matrix conditioning for spin-0 vs spin-2.

    Computes condition numbers, singular value spectra, and roundtrip
    amplification tests for the same Euclid mask with different field spins.
    Motivated by Giulio's flag on spin-2 mask inversion stability and the
    suspicious SPT ℓ≈100 spike.
    """
    input:
        shear_pkl=_shear_pkl("lensmc"),
        cmbk_masks=[
            data_dir / cmb_experiments["act"]["mask"],
            data_dir / cmb_experiments["spt_winter_gmv"]["mask"],
        ],
    output:
        npz=out_dir / "explorations" / "coupling_matrix" / "coupling_matrix_diagnostics.npz",
        png=claims_dir / "diagnose_coupling_matrix" / "coupling_matrix_diagnostics.png",
        evidence=claims_dir / "diagnose_coupling_matrix" / "evidence.json",
    params:
        all_bin_idx=_bin_to_idx("all"),
        nside=config["nside"],
        lmin=config["cross_correlation"]["binning"]["lmin"],
        lmax=config["cross_correlation"]["binning"]["lmax"],
        nells=config["cross_correlation"]["binning"]["nells"],
        aposcale=config["cross_correlation"]["aposcale"],
        n_iter=config["cross_correlation"]["n_iter"],
        cmbk_names=["act", "spt_winter_gmv"],
    log:
        "workflow/logs/diagnose_coupling_matrix.log",
    threads: 16
    resources:
        mem_mb=32000,
    script:
        "../scripts/diagnose_coupling_matrix.py"


# ---------------------------------------------------------------------------
# Euclid n(z) overview (tapestry: euclid_nz_overview)
# ---------------------------------------------------------------------------

rule plot_euclid_nz_overview:
    """Euclid RR2 redshift distributions: WL and GC side by side.

    Two-panel figure showing n(z) for all 6 tomographic bins in each probe.
    WL bins span z≈0.3–1.8; GC bins span z≈0.2–1.0. Mean redshifts are
    annotated per bin. Tapestry node sitting between the raw data fiber and
    the euclid_rr2_maps node.
    """
    input:
        wl_nz=data_dir / NZ_FILES["wl"],
        gc_nz=data_dir / NZ_FILES["gc"],
    output:
        png=claims_dir / "euclid_nz_overview" / "euclid_nz_overview.png",
        evidence=claims_dir / "euclid_nz_overview" / "evidence.json",
    log:
        "workflow/logs/plot_euclid_nz_overview.log",
    script:
        "../scripts/plot_euclid_nz_overview.py"


# ---------------------------------------------------------------------------
# Mass map cross-correlation rules (exploratory — not part of main pipeline)
# ---------------------------------------------------------------------------

rule combine_mass_cross_spectra:
    input:
        npz_files=[
            out_dir / "cross_correlations" / "individual" / "sks_binall_x_{cmbk}_cls.npz",
        ] + expand(
            out_dir / "cross_correlations" / "individual" / "sksp_bin{bin}_x_{{cmbk}}_cls.npz",
            bin=tom_bins,
        ),
    output:
        sacc=out_dir / "cross_correlations" / "combined" / f"euclid_{{cmbk}}_mass_x_cmbk_cls_nside{config['nside']}.sacc",
    log:
        f"workflow/logs/combine_mass_cross_spectra_{vcat}_{{cmbk}}.log",
    params:
        description=lambda w: f"Euclid RR2 mass maps x {w.cmbk.upper()} CMB lensing cross-correlations",
        methods="Kaiser-Squires reconstruction",
        bins="SKS (1 bin), SKSP (bins 1-6)",
    script:
        "../scripts/combine_cross_spectra.py"


rule compute_all_mass_cross_spectra:
    """Compute all mass x CMB-k cross-spectra across all CMB experiments."""
    input:
        expand(
            out_dir / "cross_correlations" / "individual" / "{method}_bin{bin}_x_{cmbk}_cls.npz",
            method=["sks"],
            bin=["all"],
            cmbk=cmbk_list,
        ) + expand(
            out_dir / "cross_correlations" / "individual" / "sksp_bin{bin}_x_{cmbk}_cls.npz",
            bin=tom_bins,
            cmbk=cmbk_list,
        ),


rule compute_allbin_cross_spectra:
    """Compute all-bin cross-spectra for lensmc, metacal, and SKS mass map."""
    input:
        expand(
            out_dir / "cross_correlations" / "individual" / "{method}_binall_x_{cmbk}_cls.npz",
            method=["lensmc", "metacal", "sks"],
            cmbk=cmbk_list,
        ),
