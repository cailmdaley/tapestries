"""
Snakemake workflow for simulation validation checks.

Canonical rule surface for gsims/sim-null stream.
"""

sim_null_config = SIM_NULL_TEST
sim_dir = out_dir / "cross_correlations" / "sim_null"
claims_dir = Path("results/claims")


rule compute_sim_null_spectrum:
    """Cross-correlate SPT sim kappa with Euclid shear (expected null signal)."""
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
    """All simulation null cross-spectra (lensmc only, 6 bins x 11 sims)."""
    input:
        expand(
            sim_dir / "{cmbk}" / "{method}_bin{bin}_x_{cmbk}_sim{sim_id}_cls.npz",
            method=["lensmc"],
            bin=tom_bins,
            cmbk=list(sim_null_config.keys()),
            sim_id=sim_null_config.get("spt_winter_gmv", {}).get("sim_ids", []),
        ),


rule plot_sim_null_test:
    """Summary: sim kappa x Euclid shear across N sims per bin."""
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


rule validate_gsims_constitution:
    """Validate gsims against fiducial signal, covariance scale, and noise."""
    input:
        sim_pickles=expand(
            "/leonardo_work/EUHPC_E05_083/cmbx/gsims/wl/1bin/maps_cls_kll_rr2_actbaseline_{sim_id}.pkl",
            sim_id=range(20),
        ),
        lensmc_maps=_shear_pkl("lensmc"),
    output:
        png=claims_dir / "validate_gsims_constitution" / "validate_gsims_constitution.png",
        evidence=claims_dir / "validate_gsims_constitution" / "evidence.json",
    log:
        "workflow/logs/validate_gsims_constitution.log",
    params:
        expected_n_gal=4_768_723,
        expected_n_eff_sr=4.093e07,
        expected_sigma_eps=3.86e-01,
        expected_noise_cl=1.82e-09,
        lensmc_bin_key="bin1",
    resources:
        mem_mb=64000,
        runtime=120,
    script:
        "../scripts/validate_gsims_constitution.py"


rule sims_constitution_pass:
    """Canonical first-pass gsims constitution validation target."""
    input:
        claims_dir / "validate_gsims_constitution" / "evidence.json",
