# --- tapestry.smk: Claims evidence aggregation and dashboard rules ---

localrules: tapestry, aggregate_cross_spectrum_evidence, aggregate_systematic_evidence, aggregate_act_vs_spt_evidence, aggregate_method_comparison_evidence, aggregate_theory_comparison_evidence, aggregate_theory_residual_evidence, aggregate_redshift_trend_evidence, aggregate_contamination_metric_evidence, aggregate_systematic_budget_evidence, aggregate_deprojected_spectra_evidence, aggregate_spt_estimator_evidence, aggregate_cmbk_systematic_evidence, generate_theory_cls_evidence, generate_validate_sacc_evidence, generate_catalog_cross_spectrum_evidence, generate_combine_shear_evidence, generate_plot_cross_spectrum_evidence, generate_catalog_vs_map_evidence, generate_compare_nz_evidence, aggregate_jackknife_null_evidence, aggregate_likelihood_input_evidence, aggregate_fiducial_likelihood_evidence, aggregate_deprojected_likelihood_evidence, aggregate_sim_null_evidence, aggregate_spatial_variation_evidence

claims_dir = Path("results/claims")

# --- Tapestry aggregate target ---
# Collects all active tapestry-tagged fiber evidence into one target.
# To update: `felt ls -t tapestry:` and sync specNames below.
TAPESTRY_SPEC_NAMES = [
    # Data / maps
    "euclid_rr2_maps",
    "cmb_lensing_maps",
    "act_dr6_lensing",
    "spt_winter_lensing",
    "shared_map_conventions",
    "euclid_nz_overview",
    # Methodology
    "plot_covariance_validation",
    "wiener_filter_validation",
    "plot_catalog_vs_map_comparison",
    # Measurement
    "plot_signal_overview",
    # Contamination
    "plot_systematic_maps",
    "plot_systematic_spectra",
    "plot_contamination_metric",
    "plot_contamination_overview",
    "plot_pte_heatmap",
    # Consistency
    "plot_method_comparison",
    "plot_spt_estimator_comparison",
    # Null tests
    "plot_bmode_null_test",
]


rule tapestry:
    """Aggregate target: all active tapestry-tagged fiber evidence."""
    input:
        [claims_dir / spec / "evidence.json" for spec in TAPESTRY_SPEC_NAMES],

sim_null_config = SIM_NULL_TEST


rule plot_euclid_rr2_maps:
    """Euclid RR2 map docs: shear weights+density, unmasked and apodized."""
    input:
        shear_pkl=_shear_pkl("lensmc"),
        density_pkl=_density_pkl(),
        act_mask=data_dir / cmb_experiments["act"]["mask"],
        spt_mask=data_dir / cmb_experiments["spt_winter_gmv"]["mask"],
        shear_apod_mask=out_dir / "masks" / "rr2_apodized" / "shear_apod_mask.fits",
        density_apod_mask=out_dir / "masks" / "rr2_apodized" / "density_apod_mask.fits",
        support_apod_mask=out_dir / "masks" / "rr2_apodized" / "shear_x_density_apod_mask.fits",
        shear_x_act_apod_mask=out_dir / "masks" / "rr2_apodized" / "shear_x_act_apod_mask.fits",
        shear_x_spt_apod_mask=out_dir / "masks" / "rr2_apodized" / "shear_x_spt_apod_mask.fits",
        density_x_act_apod_mask=out_dir / "masks" / "rr2_apodized" / "density_x_act_apod_mask.fits",
        density_x_spt_apod_mask=out_dir / "masks" / "rr2_apodized" / "density_x_spt_apod_mask.fits",
    output:
        png_map=claims_dir / "euclid_rr2_maps" / "euclid_rr2_maps.png",
        png_mask=claims_dir / "euclid_rr2_maps" / "euclid_rr2_maps_apodized.png",
        evidence=claims_dir / "euclid_rr2_maps" / "evidence.json",
    log:
        "workflow/logs/plot_euclid_rr2_maps.log",
    params:
        all_bin_idx=_bin_to_idx("all"),
    resources:
        mem_mb=12000,
    script:
        "../scripts/plot_euclid_rr2_maps.py"


rule aggregate_cross_spectrum_evidence:
    """Aggregate per-wildcard cross-spectrum evidence into per-rule summary."""
    input:
        evidence_files=expand(
            out_dir / "cross_correlations" / "individual" / "{method}_bin{bin}_x_{cmbk}_evidence.json",
            method=shear_methods,
            bin=all_bins,
            cmbk=cmbk_list,
        ),
    output:
        evidence=claims_dir / "compute_cross_spectrum" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule aggregate_systematic_evidence:
    """Aggregate per-wildcard systematic evidence into per-rule summary."""
    input:
        evidence_files=expand(
            out_dir / "cross_correlations" / "systematics" / "{sysmap}_x_{method}_bin{bin}_evidence.json",
            sysmap=sysmap_list,
            method=shear_methods,
            bin=all_bins,
        ),
    output:
        evidence=claims_dir / "compute_systematic_cross_spectrum" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule aggregate_act_vs_spt_evidence:
    """Aggregate ACT vs SPT consistency evidence (composite-only)."""
    input:
        evidence_files=[
            claims_dir / "plot_act_vs_spt" / "act_vs_spt_composite_evidence.json",
        ],
        composite_ev=claims_dir / "plot_act_vs_spt" / "act_vs_spt_composite_evidence.json",
        upstream_ev=claims_dir / "compute_cross_spectrum" / "evidence.json",
    output:
        evidence=claims_dir / "plot_act_vs_spt" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule aggregate_method_comparison_evidence:
    """Aggregate lensmc vs metacal comparison evidence."""
    input:
        evidence_files=[
            claims_dir / "plot_method_comparison" / "method_comparison_composite_evidence.json",
        ],
        composite_ev=claims_dir / "plot_method_comparison" / "method_comparison_composite_evidence.json",
        upstream_ev=claims_dir / "compute_cross_spectrum" / "evidence.json",
    output:
        evidence=claims_dir / "plot_method_comparison" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule aggregate_theory_comparison_evidence:
    """Aggregate data vs theory comparison evidence."""
    input:
        evidence_files=expand(
            claims_dir / "plot_theory_comparison" / "theory_comparison_{method}_x_{cmbk}_evidence.json",
            method=all_methods,
            cmbk=cmbk_baseline,
        ),
        upstream_ev=[
            claims_dir / "compute_cross_spectrum" / "evidence.json",
            claims_dir / "compute_theory_cls" / "evidence.json",
        ],
    output:
        evidence=claims_dir / "plot_theory_comparison" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule aggregate_theory_residual_evidence:
    """Aggregate theory residual analysis evidence."""
    input:
        evidence_files=expand(
            claims_dir / "plot_theory_residual_analysis" / "theory_residual_{method}_evidence.json",
            method=all_methods,
        ),
        upstream_ev=[
            claims_dir / "compute_cross_spectrum" / "evidence.json",
            claims_dir / "compute_theory_cls" / "evidence.json",
        ],
    output:
        evidence=claims_dir / "plot_theory_residual_analysis" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule aggregate_redshift_trend_evidence:
    """Aggregate redshift trend evidence."""
    input:
        evidence_files=expand(
            claims_dir / "plot_redshift_trend" / "redshift_trend_{method}_x_{cmbk}_evidence.json",
            method=all_methods,
            cmbk=cmbk_baseline,
        ),
        upstream_ev=[
            claims_dir / "compute_cross_spectrum" / "evidence.json",
            claims_dir / "compute_theory_cls" / "evidence.json",
        ],
    output:
        evidence=claims_dir / "plot_redshift_trend" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule aggregate_contamination_metric_evidence:
    """Aggregate contamination metric evidence."""
    input:
        evidence_files=expand(
            claims_dir / "plot_contamination_metric" / "contamination_{sysmap}_{method}_x_{cmbk}_evidence.json",
            sysmap=sysmap_list,
            method=shear_methods,
            cmbk=cmbk_list,
        )
        + expand(
            claims_dir / "plot_contamination_metric" / "contamination_{sysmap}_density_x_{cmbk}_evidence.json",
            sysmap=["stellar_density", "galactic_extinction", "exposures", "zodiacal_light"],
            cmbk=cmbk_baseline,
        ),
        composite_ev=expand(
            claims_dir / "plot_contamination_metric" / "contamination_composite_{method}_evidence.json",
            method=shear_methods,
        ),
        upstream_ev=[
            claims_dir / "compute_systematic_cross_spectrum" / "evidence.json",
            claims_dir / "compute_cmbk_systematic_cross_spectrum" / "evidence.json",
        ],
    output:
        evidence=claims_dir / "plot_contamination_metric" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule aggregate_systematic_budget_evidence:
    """Aggregate systematic budget evidence."""
    input:
        evidence_files=expand(
            claims_dir / "plot_systematic_budget" / "systematic_budget_{method}_evidence.json",
            method=all_methods,
        ),
        upstream_ev=[
            claims_dir / "compute_systematic_cross_spectrum" / "evidence.json",
            claims_dir / "compute_cmbk_systematic_cross_spectrum" / "evidence.json",
            claims_dir / "compute_cross_spectrum" / "evidence.json",
        ],
    output:
        evidence=claims_dir / "plot_systematic_budget" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule aggregate_deprojected_spectra_evidence:
    """Aggregate deprojected spectra evidence."""
    input:
        evidence_files=expand(
            claims_dir / "plot_deprojected_spectra" / "deprojected_{method}_x_{cmbk}_evidence.json",
            method=all_methods,
            cmbk=cmbk_baseline,
        ),
        upstream_ev=[
            claims_dir / "compute_systematic_cross_spectrum" / "evidence.json",
            claims_dir / "compute_cmbk_systematic_cross_spectrum" / "evidence.json",
            claims_dir / "compute_cross_spectrum" / "evidence.json",
            claims_dir / "compute_theory_cls" / "evidence.json",
        ],
    output:
        evidence=claims_dir / "plot_deprojected_spectra" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule aggregate_spt_estimator_evidence:
    """Aggregate SPT estimator comparison evidence."""
    input:
        evidence_files=expand(
            claims_dir / "plot_spt_estimator_comparison" / "spt_estimator_composite_{method}_evidence.json",
            method=shear_methods,
        ),
        upstream_ev=claims_dir / "compute_cross_spectrum" / "evidence.json",
    output:
        evidence=claims_dir / "plot_spt_estimator_comparison" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule aggregate_cmbk_systematic_evidence:
    """Aggregate CMBk x systematic cross-spectrum evidence."""
    input:
        evidence_files=expand(
            out_dir / "cross_correlations" / "cmbk_systematics" / "{sysmap}_x_{cmbk}_evidence.json",
            sysmap=sysmap_list,
            cmbk=cmbk_list,
        ),
    output:
        evidence=claims_dir / "compute_cmbk_systematic_cross_spectrum" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule generate_theory_cls_evidence:
    """Generate evidence for theory C_l computation from existing NPZ files."""
    input:
        npz_files=expand(
            out_dir / "theory" / "theory_cls_{method}_x_{cmbk}.npz",
            method=all_methods,
            cmbk=cmbk_baseline,
        ),
    output:
        evidence=claims_dir / "compute_theory_cls" / "evidence.json",
    script:
        "../scripts/generate_evidence_from_outputs.py"


rule generate_validate_sacc_evidence:
    """Generate tapestry evidence from SACC validation reports."""
    input:
        reports=expand(
            out_dir / "cross_correlations" / "combined" / "validation_{cmbk}.json",
            cmbk=cmbk_baseline,
        ),
        upstream_ev=claims_dir / "combine_shear_cross_spectra" / "evidence.json",
    output:
        evidence=claims_dir / "validate_sacc" / "evidence.json",
    script:
        "../scripts/generate_evidence_from_outputs.py"


rule generate_catalog_cross_spectrum_evidence:
    """Generate evidence for catalog-based cross-spectra from existing NPZ files."""
    input:
        npz_files=expand(
            out_dir / "cross_correlations" / "catalog_validation" / "{method}_bin{bin}_x_{cmbk}_cat_cls.npz",
            method=shear_methods,
            bin=all_bins,
            cmbk=cmbk_baseline,
        ),
    output:
        evidence=claims_dir / "compute_catalog_cross_spectrum" / "evidence.json",
    script:
        "../scripts/generate_evidence_from_outputs.py"


rule generate_combine_shear_evidence:
    """Generate evidence for combined SACC files."""
    input:
        sacc_files=expand(
            out_dir / "cross_correlations" / "combined" / f"euclid_{{cmbk}}_shear_x_cmbk_cls_nside{config['nside']}.sacc",
            cmbk=cmbk_baseline,
        ),
        upstream_ev=claims_dir / "compute_cross_spectrum" / "evidence.json",
    output:
        evidence=claims_dir / "combine_shear_cross_spectra" / "evidence.json",
    script:
        "../scripts/generate_evidence_from_outputs.py"


rule generate_plot_cross_spectrum_evidence:
    """Generate evidence for individual cross-spectrum plots from compute evidence."""
    input:
        evidence_files=expand(
            out_dir / "cross_correlations" / "individual" / "{method}_bin{bin}_x_{cmbk}_evidence.json",
            method=shear_methods,
            bin=all_bins,
            cmbk=cmbk_baseline,
        ),
        upstream_ev=claims_dir / "compute_cross_spectrum" / "evidence.json",
    output:
        evidence=claims_dir / "plot_cross_spectrum" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule generate_catalog_vs_map_evidence:
    """Aggregate evidence for catalog vs map comparison from per-method composites."""
    input:
        per_method=expand(
            claims_dir / "plot_catalog_vs_map_comparison" / "catalog_vs_map_{method}_evidence.json",
            method=shear_methods,
        ),
    output:
        evidence=claims_dir / "plot_catalog_vs_map_comparison" / "evidence.json",
    run:
        import json
        from datetime import datetime, timezone
        all_ev = {}
        output_pngs = {}
        for path in input.per_method:
            with open(path) as f:
                d = json.load(f)
            method = d.get("evidence", {}).get("method", "unknown")
            all_ev[method] = d.get("evidence", {})
            # Collect per-method PNGs for artifact rendering
            for key, val in d.get("output", {}).items():
                output_pngs[f"{method}_{key}"] = val
        doc = {
            "id": "plot_catalog_vs_map_comparison",
            "generated": datetime.now(timezone.utc).isoformat(),
            "output": output_pngs,
            "evidence": all_ev,
        }
        with open(output.evidence, "w") as f:
            json.dump(doc, f, indent=2)


rule generate_compare_nz_evidence:
    """Generate evidence for compare_nz_distributions from existing plots + theoretical n(z)."""
    input:
        theoretical_nz=data_dir / "euclid/dndz/nz.wl6.vis245-nowht.dr1s.fits",
        pngs=expand(
            out_dir / "explorations" / "nz_comparison" / "nz_comparison_{weight_type}_{phz_col}_dz{dz}.png",
            weight_type=["weighted", "unweighted"],
            phz_col=["phz_median", "phz_mode_1"],
            dz=["010"],
        ),
    output:
        evidence=claims_dir / "compare_nz_distributions" / "evidence.json",
    script:
        "../scripts/generate_evidence_from_outputs.py"


rule aggregate_jackknife_null_evidence:
    """Aggregate jackknife null test evidence."""
    input:
        evidence_files=expand(
            claims_dir / "plot_jackknife_null_test" / "jackknife_null_{method}_x_{cmbk}_evidence.json",
            method=shear_methods,
            cmbk=cmbk_baseline,
        ),
        upstream_ev=claims_dir / "compute_cross_spectrum" / "evidence.json",
    output:
        evidence=claims_dir / "plot_jackknife_null_test" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule aggregate_likelihood_input_evidence:
    """Aggregate likelihood input build evidence."""
    input:
        evidence_files=expand(
            claims_dir / "build_likelihood_input" / "{method}_x_{cmbk}_evidence.json",
            method=all_methods,
            cmbk=cmbk_baseline,
        ),
        upstream_ev=claims_dir / "compute_cross_spectrum" / "evidence.json",
    output:
        evidence=claims_dir / "build_likelihood_input" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule aggregate_fiducial_likelihood_evidence:
    """Aggregate fiducial likelihood evaluation evidence."""
    input:
        evidence_files=expand(
            claims_dir / "evaluate_fiducial_likelihood" / "likelihood_{method}_x_{cmbk}_evidence.json",
            method=all_methods,
            cmbk=cmbk_baseline,
        ),
        upstream_ev=[
            claims_dir / "build_likelihood_input" / "evidence.json",
            claims_dir / "compute_theory_cls" / "evidence.json",
        ],
    output:
        evidence=claims_dir / "evaluate_fiducial_likelihood" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule aggregate_deprojected_likelihood_evidence:
    """Aggregate deprojected likelihood evaluation evidence."""
    input:
        evidence_files=expand(
            claims_dir / "evaluate_deprojected_likelihood" / "deprojected_likelihood_{method}_x_{cmbk}_evidence.json",
            method=all_methods,
            cmbk=cmbk_baseline,
        ),
        upstream_ev=[
            claims_dir / "build_likelihood_input" / "evidence.json",
            claims_dir / "compute_theory_cls" / "evidence.json",
            claims_dir / "plot_deprojected_spectra" / "evidence.json",
        ],
    output:
        evidence=claims_dir / "evaluate_deprojected_likelihood" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule aggregate_sim_null_evidence:
    """Aggregate simulation null test evidence."""
    input:
        evidence_files=expand(
            claims_dir / "plot_sim_null_test" / "sim_null_{method}_x_{cmbk}_evidence.json",
            method=["lensmc"],
            cmbk=list(sim_null_config.keys()),
        ),
        upstream_ev=claims_dir / "compute_cross_spectrum" / "evidence.json",
    output:
        evidence=claims_dir / "plot_sim_null_test" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"


rule aggregate_spatial_variation_evidence:
    """Aggregate spatial variation test evidence (ACT only — SPT lacks north overlap)."""
    input:
        evidence_files=expand(
            claims_dir / "plot_spatial_variation" / "spatial_variation_{method}_x_{cmbk}_evidence.json",
            method=["lensmc"],
            cmbk=["act"],
        ),
        upstream_ev=claims_dir / "compute_cross_spectrum" / "evidence.json",
    output:
        evidence=claims_dir / "plot_spatial_variation" / "evidence.json",
    script:
        "../scripts/aggregate_evidence.py"
