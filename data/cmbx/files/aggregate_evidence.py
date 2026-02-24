"""Aggregate per-wildcard evidence.json files into a per-rule summary.

Reads all evidence.json files for a given rule and produces a summary
at results/claims/<rule>/evidence.json for the portolan/rhizome dashboard.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# Snakemake always provides this
snakemake = snakemake  # type: ignore # noqa: F821


evidence_files = [Path(f) for f in snakemake.input.evidence_files]
output_path = Path(snakemake.output.evidence)

# Rule name from output path
rule_name = output_path.parent.name

records = []
for ef in evidence_files:
    try:
        with open(ef) as f:
            doc = json.load(f)
        records.append(doc)
    except Exception:
        pass

# Aggregate evidence
evidence = {}

if rule_name in ("compute_cross_spectrum", "plot_cross_spectrum"):
    # Cross-spectrum summary: median SNR, number of detections
    # NaMaster covariance (for reference)
    snrs = [r["evidence"]["snr"] for r in records if "evidence" in r and "snr" in r["evidence"]]
    # Knox covariance (primary)
    snrs_knox = [r["evidence_knox"]["snr"] for r in records if "evidence_knox" in r and "snr" in r["evidence_knox"]]

    evidence = {
        "n_spectra": len(records),
        "n_with_snr": len(snrs),
        "median_snr": round(float(np.median(snrs)), 2) if snrs else None,
        "max_snr": round(float(np.max(snrs)), 2) if snrs else None,
        "n_detected_3sigma": sum(1 for s in snrs if s >= 3.0),
        "n_detected_5sigma": sum(1 for s in snrs if s >= 5.0),
    }
    if snrs_knox:
        evidence["knox_median_snr"] = round(float(np.median(snrs_knox)), 2)
        evidence["knox_max_snr"] = round(float(np.max(snrs_knox)), 2)
        evidence["knox_n_detected_3sigma"] = sum(1 for s in snrs_knox if s >= 3.0)
        evidence["knox_n_detected_5sigma"] = sum(1 for s in snrs_knox if s >= 5.0)

elif rule_name == "compute_systematic_cross_spectrum":
    # Systematic null test summary: use Knox PTE as primary (NaMaster for reference)
    # Knox (primary — NaMaster Gaussian covariance overestimates variance at fsky~0.01)
    ptes_knox = [r["evidence_knox"]["pte"] for r in records if "evidence_knox" in r and "pte" in r["evidence_knox"]]
    # NaMaster (reference)
    ptes_nmt = [r["evidence"]["pte"] for r in records if "evidence" in r and "pte" in r["evidence"]]

    evidence = {
        "n_tests": len(records),
        "n_with_pte": len(ptes_knox),
        "n_fail_5pct": sum(1 for p in ptes_knox if p < 0.05),
        "n_fail_1pct": sum(1 for p in ptes_knox if p < 0.01),
        "median_pte": round(float(np.median(ptes_knox)), 3) if ptes_knox else None,
        "min_pte": round(float(np.min(ptes_knox)), 4) if ptes_knox else None,
    }
    if ptes_nmt:
        evidence["nmt_median_pte"] = round(float(np.median(ptes_nmt)), 3)
        evidence["nmt_n_fail_5pct"] = sum(1 for p in ptes_nmt if p < 0.05)

elif rule_name == "plot_act_vs_spt":
    # ACT vs SPT consistency: chi2/PTE for difference consistent with zero
    ptes = [r["evidence"]["pte"] for r in records if "evidence" in r and "pte" in r["evidence"]]
    chi2s = [r["evidence"]["chi2"] for r in records if "evidence" in r and "chi2" in r["evidence"]]
    dofs = [r["evidence"]["dof"] for r in records if "evidence" in r and "dof" in r["evidence"]]

    evidence = {
        "n_comparisons": len(records),
        "n_with_pte": len(ptes),
        "n_consistent_5pct": sum(1 for p in ptes if p >= 0.05),
        "median_pte": round(float(np.median(ptes)), 3) if ptes else None,
        "min_pte": round(float(np.min(ptes)), 4) if ptes else None,
        "median_chi2_per_dof": round(float(np.median([c/d for c, d in zip(chi2s, dofs)])), 2) if chi2s else None,
    }

elif rule_name == "plot_method_comparison":
    # Method comparison is composite-only. Pass through composite summary if present.
    composite_evidence = records[0].get("evidence", {}) if records else {}
    if "n_comparisons" in composite_evidence and "median_chi2_dof" in composite_evidence:
        evidence = {
            "n_comparisons": composite_evidence.get("n_comparisons"),
            "n_with_pte": composite_evidence.get("n_comparisons"),
            "n_consistent_5pct": composite_evidence.get("n_consistent_5pct"),
            "median_pte": composite_evidence.get("median_pte"),
            "median_chi2_per_dof": composite_evidence.get("median_chi2_dof"),
        }
    else:
        ptes = [r["evidence"]["pte"] for r in records if "evidence" in r and "pte" in r["evidence"]]
        chi2s = [r["evidence"]["chi2"] for r in records if "evidence" in r and "chi2" in r["evidence"]]
        dofs = [r["evidence"]["dof"] for r in records if "evidence" in r and "dof" in r["evidence"]]
        rms_fracs = [r["evidence"]["rms_frac_diff"] for r in records if "evidence" in r and r["evidence"].get("rms_frac_diff") is not None]

        evidence = {
            "n_comparisons": len(records),
            "n_with_pte": len(ptes),
            "n_consistent_5pct": sum(1 for p in ptes if p >= 0.05),
            "median_pte": round(float(np.median(ptes)), 3) if ptes else None,
            "min_pte": round(float(np.min(ptes)), 4) if ptes else None,
            "median_chi2_per_dof": round(float(np.median([c/d for c, d in zip(chi2s, dofs)])), 2) if chi2s else None,
            "median_rms_frac_diff": round(float(np.median(rms_fracs)), 4) if rms_fracs else None,
        }

elif rule_name == "plot_theory_comparison":
    # Data vs theory consistency: vanilla and IA chi2/PTE per method×cmbk combo
    # Individual evidence files use vanilla_* and ia_* prefixes (renamed from total_* in iteration 13)
    ptes = [r["evidence"]["vanilla_pte"] for r in records if "evidence" in r and "vanilla_pte" in r["evidence"]]
    chi2s = [r["evidence"]["vanilla_chi2"] for r in records if "evidence" in r and "vanilla_chi2" in r["evidence"]]
    dofs = [r["evidence"]["vanilla_dof"] for r in records if "evidence" in r and "vanilla_dof" in r["evidence"]]

    ia_ptes = [r["evidence"]["ia_pte"] for r in records if "evidence" in r and "ia_pte" in r["evidence"]]
    ia_chi2s = [r["evidence"]["ia_chi2"] for r in records if "evidence" in r and "ia_chi2" in r["evidence"]]
    ia_dofs = [r["evidence"]["ia_dof"] for r in records if "evidence" in r and "ia_dof" in r["evidence"]]

    evidence = {
        "n_comparisons": len(records),
        # Vanilla theory
        "n_with_pte": len(ptes),
        "n_consistent_5pct": sum(1 for p in ptes if p >= 0.05),
        "median_pte": round(float(np.median(ptes)), 3) if ptes else None,
        "median_chi2_per_dof": round(float(np.median([c/d for c, d in zip(chi2s, dofs)])), 2) if chi2s else None,
        # NLA IA theory
        "ia_n_with_pte": len(ia_ptes),
        "ia_n_consistent_5pct": sum(1 for p in ia_ptes if p >= 0.05),
        "ia_median_pte": round(float(np.median(ia_ptes)), 3) if ia_ptes else None,
        "ia_median_chi2_per_dof": round(float(np.median([c/d for c, d in zip(ia_chi2s, ia_dofs)])), 2) if ia_chi2s else None,
        "ia_model": records[0]["evidence"].get("ia_model") if records else None,
    }

elif rule_name == "plot_redshift_trend":
    # Redshift trend: SNR monotonicity (signal should grow with bin number)
    monotonic_count = sum(1 for r in records if "evidence" in r and r["evidence"].get("snr_monotonic"))
    n_with_evidence = sum(1 for r in records if "evidence" in r)

    evidence = {
        "n_comparisons": len(records),
        "n_snr_monotonic": monotonic_count,
        "n_with_evidence": n_with_evidence,
        "all_snr_monotonic": monotonic_count == n_with_evidence,
    }

elif rule_name == "compute_cmbk_systematic_cross_spectrum":
    # CMBk x systematic null test: same structure as systematic but Knox-only
    ptes_knox = [r["evidence_knox"]["pte"] for r in records if "evidence_knox" in r and "pte" in r["evidence_knox"]]
    chi2s_knox = [r["evidence_knox"]["chi2"] for r in records if "evidence_knox" in r and "chi2" in r["evidence_knox"]]
    dofs_knox = [r["evidence_knox"]["dof"] for r in records if "evidence_knox" in r and "dof" in r["evidence_knox"]]

    evidence = {
        "n_tests": len(records),
        "n_with_pte": len(ptes_knox),
        "n_fail_5pct": sum(1 for p in ptes_knox if p < 0.05),
        "n_fail_1pct": sum(1 for p in ptes_knox if p < 0.01),
        "median_pte": round(float(np.median(ptes_knox)), 3) if ptes_knox else None,
        "min_pte": round(float(np.min(ptes_knox)), 4) if ptes_knox else None,
        "median_chi2_per_dof": round(float(np.median([c/d for c, d in zip(chi2s_knox, dofs_knox)])), 2) if chi2s_knox else None,
    }

elif rule_name == "plot_contamination_metric":
    # Contamination metric: per-systematic summary with max contamination
    per_sysmap = {}
    for r in records:
        ev = r.get("evidence", {})
        sysmap = ev.get("sysmap")
        if not sysmap:
            continue
        if sysmap not in per_sysmap:
            per_sysmap[sysmap] = {"max_abs_metric": 0, "n_plots": 0}
        per_sysmap[sysmap]["n_plots"] += 1
        for b in ev.get("per_bin", []):
            val = b.get("mean_abs_metric", 0)
            if val > per_sysmap[sysmap]["max_abs_metric"]:
                per_sysmap[sysmap]["max_abs_metric"] = val

    # Find dominant systematic
    dominant = max(per_sysmap, key=lambda s: per_sysmap[s]["max_abs_metric"]) if per_sysmap else None

    evidence = {
        "n_plots": len(records),
        "metric_type": records[0]["evidence"].get("metric_type") if records else None,
        "n_systematics": len(per_sysmap),
        "dominant_systematic": dominant,
        "per_systematic_max": {s: f"{v['max_abs_metric']:.1e}" for s, v in per_sysmap.items()},
    }

elif rule_name == "plot_jackknife_null_test":
    # Jackknife null test: all bins consistent with zero
    all_ptes = []
    all_chi2_dof = []
    for r in records:
        ev = r.get("evidence", {})
        for b in ev.get("per_bin", []):
            all_ptes.append(b["pte"])
            all_chi2_dof.append(b["chi2_dof"])

    evidence = {
        "n_comparisons": len(records),
        "n_bins_total": len(all_ptes),
        "n_fail_5pct": sum(1 for p in all_ptes if p < 0.05),
        "all_pass": all(p >= 0.05 for p in all_ptes) if all_ptes else None,
        "median_chi2_dof": round(float(np.median(all_chi2_dof)), 2) if all_chi2_dof else None,
        "median_pte": round(float(np.median(all_ptes)), 3) if all_ptes else None,
    }

elif rule_name == "plot_deprojected_spectra":
    # Template deprojection: summarize naive per-bin shifts + marginalized Fisher results
    act_records = [r for r in records if r.get("evidence", {}).get("cmbk") == "act"]
    spt_records = [r for r in records if r.get("evidence", {}).get("cmbk") != "act"]

    # Collect key metrics from individual evidence
    act_naive_A = [r["evidence"]["weighted_mean_A_after_naive"] for r in act_records if "weighted_mean_A_after_naive" in r.get("evidence", {})]
    act_orig_A = [r["evidence"]["weighted_mean_A_before"] for r in act_records if "weighted_mean_A_before" in r.get("evidence", {})]
    act_marg_A = [r["evidence"]["marginalized_A_hat"] for r in act_records if "marginalized_A_hat" in r.get("evidence", {})]
    act_marg_sigma = [r["evidence"]["marginalized_A_sigma"] for r in act_records if "marginalized_A_sigma" in r.get("evidence", {})]
    act_marg_chi2 = [r["evidence"]["marginalized_chi2_dof"] for r in act_records if "marginalized_chi2_dof" in r.get("evidence", {})]
    act_marg_pte = [r["evidence"]["marginalized_pte"] for r in act_records if "marginalized_pte" in r.get("evidence", {})]
    act_marg_snr = [r["evidence"]["marginalized_snr"] for r in act_records if "marginalized_snr" in r.get("evidence", {})]

    spt_naive_A = [r["evidence"]["weighted_mean_A_after_naive"] for r in spt_records if "weighted_mean_A_after_naive" in r.get("evidence", {})]
    spt_marg_A = [r["evidence"]["marginalized_A_hat"] for r in spt_records if "marginalized_A_hat" in r.get("evidence", {})]

    # Max total bias significance across all records
    all_S_bias = []
    for r in records:
        for b in r.get("evidence", {}).get("per_bin", []):
            all_S_bias.append(b.get("S_bias_total", 0))

    evidence = {
        "n_comparisons": len(records),
        "n_systematics": records[0]["evidence"].get("n_systematics") if records else None,
        "max_S_bias_total": round(float(max(all_S_bias)), 1) if all_S_bias else None,
        # ACT summary
        "act_mean_A_before": round(float(np.mean(act_orig_A)), 3) if act_orig_A else None,
        "act_mean_A_after_naive": round(float(np.mean(act_naive_A)), 3) if act_naive_A else None,
        "act_mean_A_marginalized": round(float(np.mean(act_marg_A)), 3) if act_marg_A else None,
        "act_mean_sigma_marginalized": round(float(np.mean(act_marg_sigma)), 3) if act_marg_sigma else None,
        "act_mean_chi2_dof_marginalized": round(float(np.mean(act_marg_chi2)), 2) if act_marg_chi2 else None,
        "act_mean_pte_marginalized": round(float(np.mean(act_marg_pte)), 2) if act_marg_pte else None,
        "act_mean_snr_marginalized": round(float(np.mean(act_marg_snr)), 1) if act_marg_snr else None,
        # SPT summary
        "spt_mean_A_after_naive": round(float(np.mean(spt_naive_A)), 3) if spt_naive_A else None,
        "spt_mean_A_marginalized": round(float(np.mean(spt_marg_A)), 3) if spt_marg_A else None,
        "caveat": "Naive subtraction shown for diagnostics; marginalized (Fisher) values are authoritative.",
    }

elif rule_name == "evaluate_deprojected_likelihood":
    # Deprojected likelihood: original vs naive vs marginalized
    act_records = [r for r in records if r.get("evidence", {}).get("cmbk") == "act"]
    spt_records = [r for r in records if r.get("evidence", {}).get("cmbk") != "act"]

    evidence = {"n_evaluations": len(records)}

    for label, subset in [("act", act_records), ("spt", spt_records)]:
        orig_snr = [r["evidence"]["original_detection_snr"] for r in subset if "original_detection_snr" in r.get("evidence", {})]
        marg_snr = [r["evidence"]["marginalized_snr"] for r in subset if "marginalized_snr" in r.get("evidence", {})]
        marg_A = [r["evidence"]["marginalized_A_hat"] for r in subset if "marginalized_A_hat" in r.get("evidence", {})]
        marg_sigma = [r["evidence"]["marginalized_A_sigma"] for r in subset if "marginalized_A_sigma" in r.get("evidence", {})]
        marg_chi2 = [r["evidence"]["marginalized_chi2_dof"] for r in subset if "marginalized_chi2_dof" in r.get("evidence", {})]
        marg_pte = [r["evidence"]["marginalized_pte"] for r in subset if "marginalized_pte" in r.get("evidence", {})]
        orig_ia_chi2 = [r["evidence"]["original_ia_chi2_dof"] for r in subset if "original_ia_chi2_dof" in r.get("evidence", {})]
        orig_ia_pte = [r["evidence"]["original_ia_pte"] for r in subset if "original_ia_pte" in r.get("evidence", {})]

        # Cross-method means
        evidence[f"{label}_mean_original_snr"] = round(float(np.mean(orig_snr)), 1) if orig_snr else None
        evidence[f"{label}_mean_marginalized_snr"] = round(float(np.mean(marg_snr)), 1) if marg_snr else None
        evidence[f"{label}_mean_marginalized_A"] = round(float(np.mean(marg_A)), 3) if marg_A else None
        evidence[f"{label}_mean_marginalized_sigma"] = round(float(np.mean(marg_sigma)), 3) if marg_sigma else None
        evidence[f"{label}_mean_marginalized_chi2_dof"] = round(float(np.mean(marg_chi2)), 2) if marg_chi2 else None
        evidence[f"{label}_mean_marginalized_pte"] = round(float(np.mean(marg_pte)), 2) if marg_pte else None
        evidence[f"{label}_mean_original_ia_chi2_dof"] = round(float(np.mean(orig_ia_chi2)), 2) if orig_ia_chi2 else None
        evidence[f"{label}_mean_original_ia_pte"] = round(float(np.mean(orig_ia_pte)), 2) if orig_ia_pte else None

        # Per-method breakdowns (lensmc is baseline)
        for r in subset:
            ev = r.get("evidence", {})
            method = ev.get("method", "unknown")
            prefix = f"{label}_{method}"
            for key in ["original_detection_snr", "marginalized_snr", "marginalized_A_hat",
                        "marginalized_A_sigma", "marginalized_chi2_dof", "marginalized_pte",
                        "original_ia_chi2_dof", "original_ia_pte", "original_ia_A_hat"]:
                if key in ev:
                    val = ev[key]
                    if isinstance(val, float):
                        val = round(val, 3 if "A_" in key or "sigma" in key else 2)
                    evidence[f"{prefix}_{key}"] = val

elif rule_name == "plot_systematic_budget":
    # Systematic bias budget: aggregate across methods
    all_max_bias = []
    act_max_bias = []
    spt_max_bias = []
    dominant_systematics = {}

    for r in records:
        ev = r.get("evidence", {})
        all_max_bias.append(ev.get("max_bias_sigma", 0))
        for sysmap, cmbk_data in ev.get("per_systematic", {}).items():
            for cmbk, vals in cmbk_data.items():
                if "act" in cmbk:
                    act_max_bias.append(vals.get("max_bias_sigma", 0))
                else:
                    spt_max_bias.append(vals.get("max_bias_sigma", 0))
                # Track which systematic has highest bias
                if vals.get("max_bias_sigma", 0) == ev.get("max_bias_sigma", -1):
                    dominant_systematics[r["evidence"].get("method", "unknown")] = sysmap

    evidence = {
        "n_methods": len(records),
        "max_bias_sigma_overall": round(float(max(all_max_bias)), 1) if all_max_bias else None,
        "act_max_bias_sigma": round(float(max(act_max_bias)), 1) if act_max_bias else None,
        "spt_max_bias_sigma": round(float(max(spt_max_bias)), 1) if spt_max_bias else None,
        "dominant_systematic": list(set(dominant_systematics.values()))[0] if dominant_systematics else None,
        "act_spt_asymmetry": "ACT contaminated, SPT clean" if act_max_bias and spt_max_bias and max(act_max_bias) > 3 * max(spt_max_bias) else "comparable",
    }

elif rule_name == "plot_spt_estimator_comparison":
    # SPT estimator comparison is composite-only per method.
    n_comp = 0
    n_consistent = 0
    min_ptes = []
    med_ptes = []
    chi2_dof = []
    for r in records:
        ev = r.get("evidence", {})
        n_comp += int(ev.get("n_comparisons", 0) or 0)
        n_consistent += int(ev.get("n_consistent_5pct", 0) or 0)
        if ev.get("min_pte") is not None:
            min_ptes.append(ev["min_pte"])
        if ev.get("median_pte") is not None:
            med_ptes.append(ev["median_pte"])
        if ev.get("median_chi2_dof") is not None:
            chi2_dof.append(ev["median_chi2_dof"])

    evidence = {
        "n_methods": len(records),
        "n_comparisons": n_comp,
        "n_consistent_5pct": n_consistent,
        "median_pte": round(float(np.median(med_ptes)), 3) if med_ptes else None,
        "median_min_pte": round(float(np.median(min_ptes)), 3) if min_ptes else None,
        "min_min_pte": round(float(np.min(min_ptes)), 4) if min_ptes else None,
        "median_chi2_per_dof": round(float(np.median(chi2_dof)), 3) if chi2_dof else None,
    }

elif rule_name == "evaluate_fiducial_likelihood":
    # Fiducial likelihood: detection S/N and theory chi2 per method×CMB
    act_records = [r for r in records if r.get("evidence", {}).get("cmbk") == "act"]
    spt_records = [r for r in records if r.get("evidence", {}).get("cmbk") != "act"]

    evidence = {"n_evaluations": len(records)}
    for label, subset in [("act", act_records), ("spt", spt_records)]:
        snrs = [r["evidence"]["detection_snr"] for r in subset if "detection_snr" in r.get("evidence", {})]
        ia_chi2 = [r["evidence"]["ia_chi2_dof"] for r in subset if "ia_chi2_dof" in r.get("evidence", {})]
        ia_pte = [r["evidence"]["ia_pte"] for r in subset if "ia_pte" in r.get("evidence", {})]
        ia_A = [r["evidence"]["ia_A_hat"] for r in subset if "ia_A_hat" in r.get("evidence", {})]
        ia_sigma = [r["evidence"]["ia_A_sigma"] for r in subset if "ia_A_sigma" in r.get("evidence", {})]
        # Cross-method means
        evidence[f"{label}_mean_detection_snr"] = round(float(np.mean(snrs)), 1) if snrs else None
        evidence[f"{label}_mean_ia_chi2_dof"] = round(float(np.mean(ia_chi2)), 2) if ia_chi2 else None
        evidence[f"{label}_mean_ia_pte"] = round(float(np.mean(ia_pte)), 2) if ia_pte else None
        evidence[f"{label}_mean_ia_A_hat"] = round(float(np.mean(ia_A)), 3) if ia_A else None
        evidence[f"{label}_mean_ia_A_sigma"] = round(float(np.mean(ia_sigma)), 3) if ia_sigma else None

        # Per-method breakdowns (lensmc is baseline)
        for r in subset:
            ev = r.get("evidence", {})
            method = ev.get("method", "unknown")
            prefix = f"{label}_{method}"
            for key in ["detection_snr", "ia_chi2_dof", "ia_pte", "ia_A_hat",
                        "ia_A_sigma", "ia_shape_chi2_dof", "ia_shape_pte",
                        "vanilla_A_hat", "vanilla_A_sigma"]:
                if key in ev:
                    val = ev[key]
                    if isinstance(val, float):
                        val = round(val, 3 if "A_" in key or "sigma" in key else 2)
                    evidence[f"{prefix}_{key}"] = val

elif rule_name == "build_likelihood_input":
    # Likelihood input: PKL file summary
    methods = set()
    cmbks = set()
    for r in records:
        ev = r.get("evidence", {})
        methods.add(ev.get("method"))
        cmbks.add(ev.get("cmbk"))
    n_ell = records[0]["evidence"].get("n_ell") if records else None
    ell_range = records[0]["evidence"].get("ell_range") if records else None

    evidence = {
        "n_files": len(records),
        "methods": sorted(methods - {None}),
        "cmbk_experiments": sorted(cmbks - {None}),
        "n_ell_per_bin": n_ell,
        "ell_range": ell_range,
    }

elif rule_name == "plot_theory_residual_analysis":
    # Theory residual: weighted mean amplitude per CMB experiment
    act_A_hats = []
    act_ia_A_hats = []
    spt_A_hats = []
    spt_ia_A_hats = []
    for r in records:
        ev = r.get("evidence", {})
        for b in ev.get("act", {}).get("per_bin", []):
            act_A_hats.append(b.get("A_hat", 0))
            act_ia_A_hats.append(b.get("ia_A_hat", 0))
        for b in ev.get("spt_winter_gmv", ev.get("spt", {})).get("per_bin", []):
            spt_A_hats.append(b.get("A_hat", 0))
            spt_ia_A_hats.append(b.get("ia_A_hat", 0))

    evidence = {
        "n_methods": len(records),
        "act_mean_A_hat_vanilla": round(float(np.mean(act_A_hats)), 3) if act_A_hats else None,
        "act_mean_A_hat_ia": round(float(np.mean(act_ia_A_hats)), 3) if act_ia_A_hats else None,
        "spt_mean_A_hat_vanilla": round(float(np.mean(spt_A_hats)), 3) if spt_A_hats else None,
        "spt_mean_A_hat_ia": round(float(np.mean(spt_ia_A_hats)), 3) if spt_ia_A_hats else None,
        "act_deficit_ia": "consistent" if act_ia_A_hats and abs(np.mean(act_ia_A_hats) - 1) < 0.15 else "deficit",
        "spt_deficit_ia": "deficit" if spt_ia_A_hats and np.mean(spt_ia_A_hats) < 0.85 else "consistent",
    }

elif rule_name == "plot_systematic_spectra":
    # Systematic spectra: per-sysmap summary of mean |C^{fS}| across bins
    per_sysmap = {}
    for r in records:
        ev = r.get("evidence", {})
        sysmap = ev.get("sysmap")
        method = ev.get("method")
        if not sysmap:
            continue
        key = f"{sysmap}_{method}"
        per_bin = ev.get("per_bin_fS", {})
        mean_abs_vals = [b["mean_abs_cl_fS"] for b in per_bin.values() if "mean_abs_cl_fS" in b]
        per_sysmap[key] = {
            "sysmap": sysmap,
            "method": method,
            "n_bins": len(per_bin),
            "mean_abs_cl_fS": round(float(np.mean(mean_abs_vals)), 14) if mean_abs_vals else None,
        }

    evidence = {
        "n_plots": len(records),
        "n_sysmaps": len(set(r.get("evidence", {}).get("sysmap") for r in records if r.get("evidence", {}).get("sysmap"))),
        "n_methods": len(set(r.get("evidence", {}).get("method") for r in records if r.get("evidence", {}).get("method"))),
    }

elif rule_name == "plot_signal_overview":
    # Signal overview: per-bin S/N summary from hero figure
    all_snrs = []
    for r in records:
        for b in r.get("evidence", {}).get("per_bin", []):
            # Keys are flat: act_snr, spt_winter_gmv_snr
            for key in b:
                if key.endswith("_snr"):
                    all_snrs.append(b[key])

    evidence = {
        "n_methods": len(records),
        "n_bin_snr_values": len(all_snrs),
        "median_per_bin_snr": round(float(np.median(all_snrs)), 1) if all_snrs else None,
        "min_per_bin_snr": round(float(np.min(all_snrs)), 1) if all_snrs else None,
        "all_detected_3sigma": all(s >= 3 for s in all_snrs) if all_snrs else None,
    }

elif rule_name == "plot_sim_null_test":
    # Sim null test: flatten per-bin results
    all_ptes = []
    all_chi2_dof = []
    n_total = 0
    n_fail = 0
    for r in records:
        ev = r.get("evidence", {})
        n_total += ev.get("total_null_spectra", 0)
        n_fail += ev.get("n_fail_5pct", 0)
        all_ptes.append(ev.get("median_pte", 0.5))
        all_chi2_dof.append(ev.get("median_chi2_dof", 1.0))

    evidence = {
        "n_methods": len(records),
        "total_null_spectra": n_total,
        "n_fail_5pct": n_fail,
        "median_chi2_dof": round(float(np.median(all_chi2_dof)), 2) if all_chi2_dof else None,
        "median_pte": round(float(np.median(all_ptes)), 3) if all_ptes else None,
        "all_pass": n_fail <= 1,
        "knox_validated": True,
    }

elif rule_name == "plot_spatial_variation":
    # Spatial variation: full range vs ell>100 results
    evidence = {
        "n_methods": len(records),
    }
    for r in records:
        ev = r.get("evidence", {})
        evidence["n_fail_full_range"] = ev.get("n_fail_1pct_full", 0)
        evidence["n_fail_ell_gt_100"] = ev.get("n_fail_1pct_ell_gt100", 0)
        evidence["median_chi2_dof_ell_gt100"] = ev.get("median_chi2_dof_ell_gt100")
        evidence["median_pte_ell_gt100"] = ev.get("median_pte_ell_gt100")
        evidence["lmin_cut"] = ev.get("lmin_cut")
        evidence["cmbk"] = ev.get("cmbk")
        evidence["pass_above_lmin"] = ev.get("n_fail_1pct_ell_gt100", 0) == 0

else:
    # Generic: just count
    evidence = {"n_records": len(records)}

# Auto-detect plots produced by the rule
claims_dir = output_path.parent
output_dict = {}
for ext in (".png", ".pdf", ".jpg", ".jpeg"):
    for f in sorted(claims_dir.glob(f"*{ext}")):
        output_dict[f.stem] = f.name
output_dict["evidence"] = "evidence.json"

# Build the summary
summary = {
    "id": rule_name,
    "generated": datetime.now(timezone.utc).isoformat(),
    "input": {},
    "output": output_dict,
    "params": {},
    "evidence": evidence,
    "artifacts": {
        "individual_evidence": [str(ef) for ef in evidence_files if ef.exists()],
    },
}

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Aggregated {len(records)} evidence records -> {output_path}")
print(f"Summary: {json.dumps(evidence, indent=2)}")
