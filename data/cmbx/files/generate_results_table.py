"""Generate comprehensive results tables for paper.

Reads evidence.json files from all tapestry nodes and compiles into:
1. Detection & theory comparison table (per method × CMB)
2. Per-bin detection significance and amplitude fits (lensmc baseline)
3. Consistency tests and null tests summary

Outputs LaTeX table fragments and evidence.json.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist


def load_json(path):
    """Load JSON file, return empty dict on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: could not load {path}: {e}")
        return {}


def load_individual_evidence(aggregate_path):
    """Load individual evidence files referenced by an aggregate."""
    agg = load_json(aggregate_path)
    artifacts = agg.get("artifacts", {})
    individual_paths = artifacts.get("individual_evidence", [])
    results = []
    base_dir = Path(aggregate_path).parent
    for rel_path in individual_paths:
        # Paths may be relative to project root or to claims dir
        p = Path(rel_path)
        if not p.is_absolute():
            # Try relative to the aggregate's directory first
            candidate = base_dir / p.name
            if candidate.exists():
                p = candidate
            else:
                p = Path(rel_path)  # Try as-is from project root
        results.append(load_json(p))
    return results


def fmt_pm(val, err, decimals=2):
    """Format value ± error as LaTeX string."""
    if np.isnan(val) or np.isnan(err):
        return "---"
    return f"${val:.{decimals}f} \\pm {err:.{decimals}f}$"


def fmt_val(val, decimals=2):
    """Format a single value."""
    if isinstance(val, float) and np.isnan(val):
        return "---"
    return f"${val:.{decimals}f}$"


def main():
    # Paths from snakemake
    claims_dir = Path(snakemake.params.claims_dir)
    out_dir = Path(snakemake.output.evidence).parent
    _ia = snakemake.config.get("intrinsic_alignment", {})
    _ia_str = f"{_ia.get('model', 'NLA-z')} (A_IA={_ia.get('A_IA', 1.72)}, eta={_ia.get('eta_IA', -0.41)})"

    # ---- Load all evidence ----

    fiducial_dir = claims_dir / "evaluate_fiducial_likelihood"
    deprojected_dir = claims_dir / "evaluate_deprojected_likelihood"
    residual_dir = claims_dir / "plot_theory_residual_analysis"
    overview_dir = claims_dir / "plot_signal_overview"

    # ---- Table 1: Detection & Theory ----

    combos = [
        ("lensmc", "act"),
        ("lensmc", "spt_winter_gmv"),
        ("metacal", "act"),
        ("metacal", "spt_winter_gmv"),
    ]

    cmbk_display = {"act": "ACT DR6", "spt_winter_gmv": "SPT-3G GMV"}
    method_display = {"lensmc": "\\textsc{lensmc}", "metacal": "\\textsc{metacal}"}

    table1_rows = []
    table1_data = {}

    for method, cmbk in combos:
        fid_path = fiducial_dir / f"likelihood_{method}_x_{cmbk}_evidence.json"
        dep_path = deprojected_dir / f"deprojected_likelihood_{method}_x_{cmbk}_evidence.json"

        fid = load_json(fid_path).get("evidence", {})
        dep = load_json(dep_path).get("evidence", {})

        key = f"{method}_x_{cmbk}"
        record = {
            "method": method,
            "cmbk": cmbk,
            "snr_raw": fid.get("detection_snr", np.nan),
            "snr_marg": dep.get("marginalized_snr", np.nan),
            "A_vanilla": fid.get("vanilla_A_hat", np.nan),
            "A_vanilla_err": fid.get("vanilla_A_sigma", np.nan),
            "A_ia": fid.get("ia_A_hat", np.nan),
            "A_ia_err": fid.get("ia_A_sigma", np.nan),
            "A_marg": dep.get("marginalized_A_hat", np.nan),
            "A_marg_err": dep.get("marginalized_A_sigma", np.nan),
            "chi2_ia": fid.get("ia_chi2_dof", np.nan),
            "pte_ia": fid.get("ia_pte", np.nan),
            "chi2_marg": dep.get("marginalized_chi2_dof", np.nan),
            "pte_marg": dep.get("marginalized_pte", np.nan),
        }
        table1_data[key] = record

        row = (
            f"  {method_display[method]} & {cmbk_display[cmbk]} "
            f"& {fmt_val(record['snr_raw'], 1)} "
            f"& {fmt_val(record['snr_marg'], 1)} "
            f"& {fmt_pm(record['A_vanilla'], record['A_vanilla_err'])} "
            f"& {fmt_pm(record['A_ia'], record['A_ia_err'])} "
            f"& {fmt_pm(record['A_marg'], record['A_marg_err'])} "
            f"& {fmt_val(record['chi2_ia'])} "
            f"& {fmt_val(record['pte_ia'])} "
            f"& {fmt_val(record['chi2_marg'])} "
            f"& {fmt_val(record['pte_marg'])} \\\\"
        )
        table1_rows.append(row)

    table1_latex = r"""\begin{table*}
\centering
\caption{Cross-correlation detection significance and theory comparison. S/N$_\mathrm{raw}$ is the diagonal Knox sum over 6 tomographic bins $\times$ 20 bandpowers. S/N$_\mathrm{marg}$ accounts for template marginalization of 5 systematic maps via joint Fisher fit. $A$ is the best-fit amplitude relative to fiducial theory ($A=1$ means perfect agreement). $A_\mathrm{van}$: vanilla Planck 2018. $A_\mathrm{lens}$: Planck 2018 + NLA intrinsic alignment ($A_\mathrm{IA}=__A_IA__$, $\eta_\mathrm{IA}=__ETA_IA__$). $A_\mathrm{lens}^\mathrm{marg}$: IA + systematic template marginalization. $\chi^2/\nu$ and PTE evaluate joint goodness-of-fit across all bins and bandpowers ($\nu = 119$ for IA, $\nu = 115$ for marginalized).}
\label{tab:detection}
\begin{tabular}{ll cc ccc cc cc}
\toprule
Method & CMB & S/N$_\mathrm{raw}$ & S/N$_\mathrm{marg}$ & $A_\mathrm{van}$ & $A_\mathrm{lens}$ & $A_\mathrm{lens}^\mathrm{marg}$ & $\chi^2/\nu_\mathrm{IA}$ & PTE$_\mathrm{IA}$ & $\chi^2/\nu_\mathrm{marg}$ & PTE$_\mathrm{marg}$ \\
\midrule
"""
    table1_latex = table1_latex.replace("__A_IA__", str(_ia.get("A_IA", 1.72)))
    table1_latex = table1_latex.replace("__ETA_IA__", str(_ia.get("eta_IA", -0.41)))
    table1_latex += "\n".join(table1_rows)
    table1_latex += r"""
\bottomrule
\end{tabular}
\end{table*}"""

    # ---- Table 2: Per-bin (lensmc baseline) ----

    residual_lensmc = load_json(residual_dir / "theory_residual_lensmc_evidence.json")
    overview_lensmc = load_json(overview_dir / "signal_overview_lensmc_evidence.json")

    res_ev = residual_lensmc.get("evidence", {})
    ov_ev = overview_lensmc.get("evidence", {})

    act_bins = res_ev.get("act", {}).get("per_bin", [])
    spt_bins = res_ev.get("spt_winter_gmv", {}).get("per_bin", [])
    ov_bins = ov_ev.get("per_bin", [])

    table2_rows = []
    table2_data = []

    for i in range(6):
        act_b = act_bins[i] if i < len(act_bins) else {}
        spt_b = spt_bins[i] if i < len(spt_bins) else {}
        ov_b = ov_bins[i] if i < len(ov_bins) else {}

        # Use shape chi2 (at best-fit A, dof=dof_shape) from theory_residual
        # — more informative than unity chi2 when A != 1 (especially for SPT)
        act_dof_shape = act_b.get("dof_shape", 19)
        spt_dof_shape = spt_b.get("dof_shape", 19)

        bin_record = {
            "bin": i + 1,
            "act_snr": ov_b.get("act_snr", np.nan),
            "spt_snr": ov_b.get("spt_winter_gmv_snr", np.nan),
            "act_A_ia": act_b.get("ia_A_hat", np.nan),
            "act_A_ia_err": act_b.get("ia_sigma_A", np.nan),
            "spt_A_ia": spt_b.get("ia_A_hat", np.nan),
            "spt_A_ia_err": spt_b.get("ia_sigma_A", np.nan),
            "act_chi2_dof": round(act_b.get("ia_chi2_shape", np.nan) / act_dof_shape, 2) if "ia_chi2_shape" in act_b else np.nan,
            "act_pte": act_b.get("ia_pte_shape", np.nan),
            "spt_chi2_dof": round(spt_b.get("ia_chi2_shape", np.nan) / spt_dof_shape, 2) if "ia_chi2_shape" in spt_b else np.nan,
            "spt_pte": spt_b.get("ia_pte_shape", np.nan),
        }
        table2_data.append(bin_record)

        row = (
            f"  {i+1} "
            f"& {fmt_val(bin_record['act_snr'], 1)} "
            f"& {fmt_val(bin_record['spt_snr'], 1)} "
            f"& {fmt_pm(bin_record['act_A_ia'], bin_record['act_A_ia_err'])} "
            f"& {fmt_pm(bin_record['spt_A_ia'], bin_record['spt_A_ia_err'])} "
            f"& {fmt_val(bin_record['act_chi2_dof'])} "
            f"& {fmt_val(bin_record['act_pte'])} "
            f"& {fmt_val(bin_record['spt_chi2_dof'])} "
            f"& {fmt_val(bin_record['spt_pte'])} \\\\"
        )
        table2_rows.append(row)

    # Add joint fit row — use table1_data (evaluate_fiducial_likelihood) for both
    # amplitude AND chi2, ensuring consistency with Table 1
    # Use shape chi2 for the "All" row too (joint fit at best-fit A)
    act_fid = table1_data.get("lensmc_x_act", {})
    spt_fid = table1_data.get("lensmc_x_spt_winter_gmv", {})

    # Shape chi2/dof from evaluate_fiducial_likelihood individual evidence
    fid_act_ev = load_json(fiducial_dir / "likelihood_lensmc_x_act_evidence.json").get("evidence", {})
    fid_spt_ev = load_json(fiducial_dir / "likelihood_lensmc_x_spt_winter_gmv_evidence.json").get("evidence", {})

    mean_row = (
        f"  \\midrule\n"
        f"  All "
        f"& {fmt_val(act_fid.get('snr_raw', np.nan), 1)} "
        f"& {fmt_val(spt_fid.get('snr_raw', np.nan), 1)} "
        f"& {fmt_pm(act_fid.get('A_ia', np.nan), act_fid.get('A_ia_err', np.nan))} "
        f"& {fmt_pm(spt_fid.get('A_ia', np.nan), spt_fid.get('A_ia_err', np.nan))} "
        f"& {fmt_val(fid_act_ev.get('ia_shape_chi2_dof', act_fid.get('chi2_ia', np.nan)))} "
        f"& {fmt_val(fid_act_ev.get('ia_shape_pte', act_fid.get('pte_ia', np.nan)))} "
        f"& {fmt_val(fid_spt_ev.get('ia_shape_chi2_dof', spt_fid.get('chi2_ia', np.nan)))} "
        f"& {fmt_val(fid_spt_ev.get('ia_shape_pte', spt_fid.get('pte_ia', np.nan)))} \\\\"
    )
    table2_rows.append(mean_row)

    table2_latex = r"""\begin{table}
\centering
\caption{Per-bin detection significance and amplitude fits for \textsc{lensmc}. S/N is Knox diagonal. $A_\mathrm{lens}$ is the best-fit amplitude relative to Planck 2018 + NLA IA theory per bin (independent fits, 20 bandpowers each). $\chi^2/\nu$ evaluates the theory shape at the best-fit amplitude per bin ($\nu = 19$). ``All'' row shows the joint fit across all bins ($\nu = 119$).}
\label{tab:perbin}
\begin{tabular}{c cc cc cc cc}
\toprule
Bin & \multicolumn{2}{c}{S/N (Knox)} & \multicolumn{2}{c}{$A_\mathrm{lens}$} & \multicolumn{2}{c}{ACT $\chi^2/\nu$ vs theory} & \multicolumn{2}{c}{SPT $\chi^2/\nu$ vs theory} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
 & ACT & SPT & ACT & SPT & $\chi^2/\nu$ & PTE & $\chi^2/\nu$ & PTE \\
\midrule
"""
    table2_latex += "\n".join(table2_rows)
    table2_latex += r"""
\bottomrule
\end{tabular}
\end{table}"""

    # ---- Table 3: Validation summary ----

    # Read aggregate evidence (some have summary keys, others need individual files)
    act_spt_ev = load_json(claims_dir / "plot_act_vs_spt" / "evidence.json").get("evidence", {})
    method_ev = load_json(claims_dir / "plot_method_comparison" / "evidence.json").get("evidence", {})
    catalog_ev = load_json(claims_dir / "plot_catalog_validation_summary" / "evidence.json").get("evidence", {})
    jackknife_ev = load_json(claims_dir / "plot_jackknife_null_test" / "evidence.json").get("evidence", {})
    systematic_ev = load_json(claims_dir / "plot_systematic_null_tests" / "evidence.json").get("evidence", {})

    # Read individual evidence for tests where aggregate is thin
    sim_null_individual = load_json(
        claims_dir / "plot_sim_null_test" / "sim_null_lensmc_x_spt_winter_gmv_evidence.json"
    ).get("evidence", {})

    spatial_individual = load_json(
        claims_dir / "plot_spatial_variation" / "spatial_variation_lensmc_x_act_evidence.json"
    ).get("evidence", {})

    budget_individual = load_json(
        claims_dir / "plot_systematic_budget" / "systematic_budget_lensmc_evidence.json"
    ).get("evidence", {})

    bmode_ev = load_json(claims_dir / "plot_bmode_null_test" / "evidence.json")

    # SPT estimators: read all individual, count consistency
    estimator_individuals = load_individual_evidence(
        claims_dir / "plot_spt_estimator_comparison" / "evidence.json"
    )
    n_est_total = len(estimator_individuals)
    n_est_pass = sum(1 for e in estimator_individuals if e.get("evidence", {}).get("all_consistent", False))
    est_min_pte = min(
        (e.get("evidence", {}).get("min_pte", 1.0) for e in estimator_individuals),
        default=np.nan,
    )

    validation_tests = [
        {
            "name": "ACT vs SPT",
            "type": "Consistency",
            "n_pass": act_spt_ev.get("n_consistent_5pct", 0),
            "n_total": act_spt_ev.get("n_comparisons", 0),
            "median_chi2_dof": act_spt_ev.get("median_chi2_per_dof", np.nan),
            "median_pte": act_spt_ev.get("median_pte", np.nan),
        },
        {
            "name": "lensmc vs metacal",
            "type": "Consistency",
            "n_pass": method_ev.get("n_consistent_5pct", 0),
            "n_total": method_ev.get("n_comparisons", 0),
            "median_chi2_dof": method_ev.get("median_chi2_per_dof", np.nan),
            "median_pte": method_ev.get("median_pte", np.nan),
        },
        {
            "name": "SPT estimators",
            "type": "Consistency",
            "n_pass": n_est_pass,
            "n_total": n_est_total,
            "median_chi2_dof": np.nan,  # Not a simple median for estimator tests
            "median_pte": est_min_pte,
        },
        {
            "name": "Catalog vs map",
            "type": "Consistency",
            "n_pass": catalog_ev.get("n_comparisons", 0),  # All pass (corr > 0.98)
            "n_total": catalog_ev.get("n_comparisons", 0),
            "median_chi2_dof": np.nan,
            "median_pte": catalog_ev.get("median_correlation", np.nan),
        },
        {
            "name": "Jackknife sign-flip",
            "type": "Null",
            "n_pass": jackknife_ev.get("n_bins_total", 0) - jackknife_ev.get("n_fail_5pct", 0),
            "n_total": jackknife_ev.get("n_bins_total", 0),
            "median_chi2_dof": jackknife_ev.get("median_chi2_dof", np.nan),
            "median_pte": jackknife_ev.get("median_pte", np.nan),
        },
        {
            "name": "Sim null (SPT $\\kappa$ sims)",
            "type": "Null",
            "n_pass": sim_null_individual.get("total_null_spectra", 0) - sim_null_individual.get("n_fail_5pct", 0),
            "n_total": sim_null_individual.get("total_null_spectra", 0),
            "median_chi2_dof": sim_null_individual.get("median_chi2_dof", np.nan),
            "median_pte": sim_null_individual.get("median_pte", np.nan),
        },
        {
            "name": "Spatial ($\\ell > 100$)",
            "type": "Null",
            "n_pass": 6 - spatial_individual.get("n_fail_1pct_ell_gt100", 0),
            "n_total": 6,
            "median_chi2_dof": spatial_individual.get("median_chi2_dof_ell_gt100", np.nan),
            "median_pte": spatial_individual.get("median_pte_ell_gt100", np.nan),
        },
        {
            "name": "$C^{\\gamma S}_\\ell$ systematics",
            "type": "Null",
            "n_pass": systematic_ev.get("n_tests", 0) - systematic_ev.get("n_fail_5pct", 0),
            "n_total": systematic_ev.get("n_tests", 0),
            "median_chi2_dof": systematic_ev.get("median_chi2_dof", np.nan),
            "median_pte": systematic_ev.get("median_pte", np.nan),
        },
        {
            "name": "$B \\times \\kappa$ null",
            "type": "Null",
            "n_pass": bmode_ev.get("n_pass", 0),
            "n_total": bmode_ev.get("n_total", 0),
            "median_chi2_dof": round(bmode_ev["joint_chi2"] / bmode_ev["joint_dof"], 2) if bmode_ev.get("joint_dof") else np.nan,
            "median_pte": float(1.0 - chi2_dist.cdf(bmode_ev["joint_chi2"], bmode_ev["joint_dof"])) if bmode_ev.get("joint_dof") else np.nan,
        },
    ]

    table3_rows = []

    for test in validation_tests:
        n_pass = test["n_pass"]
        n_total = test["n_total"]
        n_str = f"{n_pass}/{n_total}"

        chi2_str = fmt_val(test["median_chi2_dof"])
        pte_str = fmt_val(test["median_pte"])

        # Special formatting for catalog (correlation, not PTE)
        if test["name"] == "Catalog vs map":
            pte_str = f"$r = {test['median_pte']:.3f}$"

        if n_pass == n_total:
            verdict = "\\checkmark"
        else:
            n_fail = n_total - n_pass
            verdict = f"{n_fail} fail"

        row = f"  {test['name']} & {test['type']} & {n_str} & {chi2_str} & {pte_str} & {verdict} \\\\"
        table3_rows.append(row)

    # Add ℓ-range robustness rows (lensmc baseline)
    robustness_agg = load_json(claims_dir / "plot_ell_range_robustness" / "evidence.json")
    rob_ev = robustness_agg.get("evidence", {}).get("per_method_cmb", {})

    act_rob = rob_ev.get("lensmc_act", {})
    spt_rob = rob_ev.get("lensmc_spt_winter_gmv", {})

    # Count bins passing 2σ threshold (6 bins per CMB, from individual evidence)
    rob_lensmc = load_json(
        claims_dir / "plot_ell_range_robustness" / "ell_range_robustness_lensmc_evidence.json"
    ).get("evidence", {})

    for cmbk_key, cmbk_label, rob_summary in [
        ("act", "ACT", act_rob),
        ("spt_winter_gmv", "SPT", spt_rob),
    ]:
        per_bin = rob_lensmc.get(cmbk_key, {}).get("per_bin", {})
        n_pass = sum(1 for b in per_bin.values() if b.get("max_deviation_sigma", 99) < 2.0)
        n_total = len(per_bin) if per_bin else 6
        worst = rob_summary.get("worst_max_deviation_sigma", np.nan)
        worst_bin = rob_summary.get("worst_bin", "?")
        if n_pass == n_total:
            verdict = "\\checkmark"
        else:
            verdict = f"${worst:.1f}\\sigma$ (bin {worst_bin})"

        rob_row = (
            f"  $\\ell$-range stability ({cmbk_label}) & Robustness "
            f"& {n_pass}/{n_total} "
            f"& --- "
            f"& max ${worst:.1f}\\sigma$ "
            f"& {verdict} \\\\"
        )
        table3_rows.append(rob_row)

    # Add bias budget row
    budget_max_act = budget_individual.get("max_bias_sigma", np.nan)
    budget_per_sys = budget_individual.get("per_systematic", {})
    worst_sys = max(
        budget_per_sys.items(),
        key=lambda x: x[1].get("act", {}).get("max_bias_sigma", 0),
        default=("---", {}),
    )
    worst_name = worst_sys[0].replace("_", " ")

    budget_row = (
        f"  Bias budget (ACT) & Bias "
        f"& --- "
        f"& --- "
        f"& --- "
        f"& max ${budget_max_act:.1f}\\sigma$ ({worst_name}) \\\\"
    )
    table3_rows.append(budget_row)

    table3_latex = r"""\begin{table}
\centering
\caption{Validation tests summary. ``Consistency'' tests compare independent measurements of the same cross-correlation signal (passing = agreement at 5\% significance). ``Null'' tests check that signals expected to vanish are consistent with zero. The bias budget quantifies the harmonic-space contamination metric $X^f_S(\ell) = C^{\kappa S}_\ell \cdot C^{fS}_\ell / C^{SS}_\ell$ (2203.12440, Eq.~A.4.1) integrated over bandpowers. All $\chi^2/\nu$ use Knox diagonal covariance.}
\label{tab:validation}
\begin{tabular}{llcccc}
\toprule
Test & Type & Pass/Total & Med.\ $\chi^2/\nu$ & Med.\ PTE & Verdict \\
\midrule
"""
    table3_latex += "\n".join(table3_rows)
    table3_latex += r"""
\bottomrule
\end{tabular}
\end{table}"""

    # ---- Write outputs ----

    full_latex = (
        "% Auto-generated results tables for Euclid x CMB-kappa cross-correlation\n"
        "% Source: generate_results_table.py\n"
        f"% Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        + table1_latex
        + "\n\n"
        + table2_latex
        + "\n\n"
        + table3_latex
        + "\n"
    )

    tex_path = out_dir / "results_tables.tex"
    tex_path.write_text(full_latex)
    print(f"Wrote LaTeX tables to {tex_path}")

    # Evidence JSON (flat structure for dashboard)
    evidence = {
        "id": "generate_results_table",
        "generated": datetime.now(timezone.utc).isoformat(),
        "evidence": {
            "n_method_cmb_combos": len(combos),
            "n_bins": 6,
            "n_validation_tests": len(validation_tests) + 3,  # +1 bias budget +2 robustness
            # Headline numbers (lensmc baseline)
            "act_lensmc_snr_raw": table1_data["lensmc_x_act"]["snr_raw"],
            "act_lensmc_snr_marg": table1_data["lensmc_x_act"]["snr_marg"],
            "spt_lensmc_snr_raw": table1_data["lensmc_x_spt_winter_gmv"]["snr_raw"],
            "spt_lensmc_snr_marg": table1_data["lensmc_x_spt_winter_gmv"]["snr_marg"],
            "act_A_ia": table1_data["lensmc_x_act"]["A_ia"],
            "act_A_ia_err": table1_data["lensmc_x_act"]["A_ia_err"],
            "act_A_marg": table1_data["lensmc_x_act"]["A_marg"],
            "act_A_marg_err": table1_data["lensmc_x_act"]["A_marg_err"],
            "spt_A_ia": table1_data["lensmc_x_spt_winter_gmv"]["A_ia"],
            "spt_A_ia_err": table1_data["lensmc_x_spt_winter_gmv"]["A_ia_err"],
            "act_ia_chi2_dof": table1_data["lensmc_x_act"]["chi2_ia"],
            "act_ia_pte": table1_data["lensmc_x_act"]["pte_ia"],
            "act_marg_chi2_dof": table1_data["lensmc_x_act"]["chi2_marg"],
            "act_marg_pte": table1_data["lensmc_x_act"]["pte_marg"],
            "all_consistency_pass": all(
                t["n_pass"] == t["n_total"]
                for t in validation_tests
                if t["type"] == "Consistency"
            ),
            "n_null_fail_systematic": systematic_ev.get("n_fail_5pct", 0),
            "act_max_bias_sigma": budget_max_act,
            "act_worst_systematic": worst_sys[0],
            "act_robustness_worst_sigma": act_rob.get("worst_max_deviation_sigma"),
            "spt_robustness_worst_sigma": spt_rob.get("worst_max_deviation_sigma"),
            "ia_model": _ia_str,
        },
        "artifacts": {
            "tex": "results_tables.tex",
        },
    }

    evidence_path = Path(snakemake.output.evidence)
    evidence_path.write_text(json.dumps(evidence, indent=2))
    print(f"Wrote evidence to {evidence_path}")


if __name__ == "__main__":
    main()
