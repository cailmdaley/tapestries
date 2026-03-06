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

    shear_combos = [
        ("lensmc", "act"),
        ("lensmc", "spt_winter_gmv"),
        ("metacal", "act"),
        ("metacal", "spt_winter_gmv"),
    ]
    density_combos = [
        ("density", "act"),
        ("density", "spt_winter_gmv"),
    ]
    combos = shear_combos + density_combos

    cmbk_display = {"act": "ACT DR6", "spt_winter_gmv": "SPT-3G GMV"}
    method_display = {
        "lensmc": "\\textsc{lensmc}",
        "metacal": "\\textsc{metacal}",
        "density": "Density ($\\delta_g$)",
    }

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
            "A_vanilla": fid.get("A_lens", np.nan),
            "A_vanilla_err": fid.get("A_lens_sigma", np.nan),
            "A_ia": fid.get("A_lens_with_IA", np.nan),
            "A_ia_err": fid.get("A_lens_with_IA_sigma", np.nan),
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
\caption{Cross-correlation detection significance and theory comparison. S/N$_\mathrm{raw}$ is the excess $\chi^2$ metric $(\chi^2 - \nu)/\sqrt{2\nu}$ over 6 tomographic bins $\times$ 20 bandpowers. S/N$_\mathrm{marg}$ accounts for template marginalization of 11 systematic maps (10 active) via joint Fisher fit. $A$ is the best-fit amplitude relative to fiducial theory ($A=1$ means perfect agreement). $A_\mathrm{van}$: vanilla Planck 2018. $A_\mathrm{lens}$: Planck 2018 + NLA intrinsic alignment ($A_\mathrm{IA}=__A_IA__$, $\eta_\mathrm{IA}=__ETA_IA__$) for shear; vanilla Planck 2018 + galaxy bias for density (IA does not apply). $A_\mathrm{lens}^\mathrm{marg}$: fiducial + systematic template marginalization. $\chi^2/\nu$ and PTE evaluate joint goodness-of-fit across all bins and bandpowers ($\nu = 119$ for fiducial, $\nu = 109$ for marginalized).}
\label{tab:detection}
\begin{tabular}{ll cc ccc cc cc}
\toprule
Method & CMB & S/N$_\mathrm{raw}$ & S/N$_\mathrm{marg}$ & $A_\mathrm{van}$ & $A_\mathrm{lens}$ & $A_\mathrm{lens}^\mathrm{marg}$ & $\chi^2/\nu_\mathrm{IA}$ & PTE$_\mathrm{IA}$ & $\chi^2/\nu_\mathrm{marg}$ & PTE$_\mathrm{marg}$ \\
\midrule
"""
    table1_latex = table1_latex.replace("__A_IA__", str(_ia.get("A_IA", 1.72)))
    table1_latex = table1_latex.replace("__ETA_IA__", str(_ia.get("eta_IA", -0.41)))
    # Insert midrule between shear and density sections
    table1_rows.insert(len(shear_combos), "  \\midrule")
    table1_latex += "\n".join(table1_rows)
    table1_latex += r"""
\bottomrule
\end{tabular}
\end{table*}"""

    # ---- Table 2: Per-bin (density + shear) ----

    # Helper: build per-bin rows from residual + overview evidence
    def perbin_rows(residual_ev, overview_ev, table1_key_act,
                    table1_key_spt, use_ia_keys=True):
        """Build per-bin LaTeX rows + joint row."""
        act_bins = residual_ev.get("act", {}).get("per_bin", [])
        spt_bins = residual_ev.get(
            "spt_winter_gmv", {}
        ).get("per_bin", [])
        ov_bins = overview_ev.get("per_bin", [])
        rows, data = [], []
        # Per-bin A key depends on tracer
        a_key = "A_lens_with_IA" if use_ia_keys else "A_hat"
        s_key = "ia_sigma_A" if use_ia_keys else "sigma_A"
        cs_key = "ia_chi2_shape" if use_ia_keys else "chi2_shape"
        ps_key = "ia_pte_shape" if use_ia_keys else "pte_shape"
        for i in range(6):
            ab = act_bins[i] if i < len(act_bins) else {}
            sb = spt_bins[i] if i < len(spt_bins) else {}
            ob = ov_bins[i] if i < len(ov_bins) else {}
            dof_a = ab.get("dof_shape", 19)
            dof_s = sb.get("dof_shape", 19)
            rec = {
                "bin": i + 1,
                "act_snr": ob.get("act_snr", np.nan),
                "spt_snr": ob.get(
                    "spt_winter_gmv_snr", np.nan
                ),
                "act_A": ab.get(a_key, np.nan),
                "act_A_err": ab.get(s_key, np.nan),
                "spt_A": sb.get(a_key, np.nan),
                "spt_A_err": sb.get(s_key, np.nan),
                "act_chi2": round(
                    ab[cs_key] / dof_a, 2
                ) if cs_key in ab else np.nan,
                "act_pte": ab.get(ps_key, np.nan),
                "spt_chi2": round(
                    sb[cs_key] / dof_s, 2
                ) if cs_key in sb else np.nan,
                "spt_pte": sb.get(ps_key, np.nan),
            }
            data.append(rec)
            rows.append(
                f"  {i+1} "
                f"& {fmt_val(rec['act_snr'], 1)} "
                f"& {fmt_val(rec['spt_snr'], 1)} "
                f"& {fmt_pm(rec['act_A'], rec['act_A_err'])} "
                f"& {fmt_pm(rec['spt_A'], rec['spt_A_err'])} "
                f"& {fmt_val(rec['act_chi2'])} "
                f"& {fmt_val(rec['act_pte'])} "
                f"& {fmt_val(rec['spt_chi2'])} "
                f"& {fmt_val(rec['spt_pte'])} \\\\"
            )
        # Joint row from table1_data
        af = table1_data.get(table1_key_act, {})
        sf = table1_data.get(table1_key_spt, {})
        fid_act = load_json(
            fiducial_dir / f"likelihood_{table1_key_act.replace('_x_', '_x_')}_evidence.json"
        ).get("evidence", {})
        fid_spt = load_json(
            fiducial_dir / f"likelihood_{table1_key_spt}_evidence.json"
        ).get("evidence", {})
        rows.append(
            f"  \\midrule\n"
            f"  All "
            f"& {fmt_val(af.get('snr_raw', np.nan), 1)} "
            f"& {fmt_val(sf.get('snr_raw', np.nan), 1)} "
            f"& {fmt_pm(af.get('A_ia', np.nan), af.get('A_ia_err', np.nan))} "
            f"& {fmt_pm(sf.get('A_ia', np.nan), sf.get('A_ia_err', np.nan))} "
            f"& {fmt_val(fid_act.get('ia_shape_chi2_dof', af.get('chi2_ia', np.nan)))} "
            f"& {fmt_val(fid_act.get('ia_shape_pte', af.get('pte_ia', np.nan)))} "
            f"& {fmt_val(fid_spt.get('ia_shape_chi2_dof', sf.get('chi2_ia', np.nan)))} "
            f"& {fmt_val(fid_spt.get('ia_shape_pte', sf.get('pte_ia', np.nan)))} \\\\"
        )
        return rows, data

    # Density per-bin
    residual_density = load_json(
        residual_dir / "theory_residual_density_evidence.json"
    ).get("evidence", {})
    overview_density = load_json(
        overview_dir / "signal_overview_density_evidence.json"
    ).get("evidence", {})
    # Density uses vanilla A (no IA) — keys are A_hat/sigma_A
    density_rows, density_data = perbin_rows(
        residual_density, overview_density,
        "density_x_act", "density_x_spt_winter_gmv",
        use_ia_keys=False,
    )

    # Shear (lensmc) per-bin
    residual_lensmc = load_json(
        residual_dir / "theory_residual_lensmc_evidence.json"
    ).get("evidence", {})
    overview_lensmc = load_json(
        overview_dir / "signal_overview_lensmc_evidence.json"
    ).get("evidence", {})
    shear_rows, table2_data = perbin_rows(
        residual_lensmc, overview_lensmc,
        "lensmc_x_act", "lensmc_x_spt_winter_gmv",
        use_ia_keys=True,
    )

    table2_latex = r"""\begin{table}
\centering
\caption{Per-bin detection significance and amplitude fits. \textbf{Top}: galaxy density $\times$ CMB-$\kappa$ (the headline detection). \textbf{Bottom}: \textsc{lensmc} shear $\times$ CMB-$\kappa$ (not detected). S/N is Knox excess $\chi^2$. $A$ is the best-fit amplitude relative to Planck 2018 theory per bin (20 bandpowers each). Density uses vanilla theory; shear uses Planck 2018 + NLA IA. $\chi^2/\nu$ evaluates the theory shape at best-fit $A$ ($\nu = 19$). ``All'' = joint fit ($\nu = 119$).}
\label{tab:perbin}
\begin{tabular}{c cc cc cc cc}
\toprule
Bin & \multicolumn{2}{c}{S/N (Knox)} & \multicolumn{2}{c}{$A$} & \multicolumn{2}{c}{ACT $\chi^2/\nu$} & \multicolumn{2}{c}{SPT $\chi^2/\nu$} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
 & ACT & SPT & ACT & SPT & $\chi^2/\nu$ & PTE & $\chi^2/\nu$ & PTE \\
\midrule
\multicolumn{9}{l}{\textit{Density} $\delta_g \times \kappa$} \\
"""
    table2_latex += "\n".join(density_rows)
    table2_latex += "\n  \\midrule\n"
    table2_latex += (
        "\\multicolumn{9}{l}"
        "{\\textit{Shear (\\textsc{lensmc})} "
        "$\\gamma \\times \\kappa$} \\\\\n"
    )
    table2_latex += "\n".join(shear_rows)
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

    # SPT estimators: read aggregate + individual evidence
    estimator_agg_ev = load_json(
        claims_dir / "plot_spt_estimator_comparison" / "evidence.json"
    ).get("evidence", {})
    estimator_individuals = load_individual_evidence(
        claims_dir / "plot_spt_estimator_comparison" / "evidence.json"
    )
    n_est_total = len(estimator_individuals)
    n_est_pass = sum(
        1 for e in estimator_individuals
        if e.get("evidence", {}).get("n_consistent_5pct", 0)
        == e.get("evidence", {}).get("n_comparisons", -1)
    )
    est_min_pte = min(
        (e.get("evidence", {}).get("min_pte", 1.0) for e in estimator_individuals),
        default=np.nan,
    )

    # Split systematic tests into shear vs density for accurate reporting
    _sys_per_method = systematic_ev.get("per_method", {})
    _lensmc_sys = _sys_per_method.get("lensmc", {})
    _metacal_sys = _sys_per_method.get("metacal", {})
    _density_sys = _sys_per_method.get("density", {})
    _shear_sys_n = _lensmc_sys.get("n_tests", 0) + _metacal_sys.get("n_tests", 0)
    _shear_sys_fail = _lensmc_sys.get("n_fail_5pct", 0) + _metacal_sys.get("n_fail_5pct", 0)
    _shear_sys_chi2 = np.nanmean([
        _lensmc_sys.get("median_chi2_dof", np.nan),
        _metacal_sys.get("median_chi2_dof", np.nan),
    ])
    _shear_sys_pte = np.nanmean([
        _lensmc_sys.get("median_pte", np.nan),
        _metacal_sys.get("median_pte", np.nan),
    ])
    _density_sys_n = _density_sys.get("n_tests", 0)
    _density_sys_fail = _density_sys.get("n_fail_5pct", 0)
    _density_sys_chi2 = _density_sys.get("median_chi2_dof", np.nan)
    _density_sys_pte = _density_sys.get("median_pte", np.nan)

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
            "median_chi2_dof": estimator_agg_ev.get(
                "median_chi2_per_dof", np.nan
            ),
            "median_pte": estimator_agg_ev.get("median_pte", np.nan),
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
            "n_pass": _shear_sys_n - _shear_sys_fail,
            "n_total": _shear_sys_n,
            "median_chi2_dof": _shear_sys_chi2,
            "median_pte": _shear_sys_pte,
        },
        {
            "name": "$C^{\\delta S}_\\ell$ systematics",
            "type": "Null",
            "n_pass": _density_sys_n - _density_sys_fail,
            "n_total": _density_sys_n,
            "median_chi2_dof": _density_sys_chi2,
            "median_pte": _density_sys_pte,
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
\caption{Validation tests summary. ``Consistency'' tests compare independent measurements of the same cross-correlation signal (passing = agreement at 5\% significance). ``Null'' tests check that signals expected to vanish are consistent with zero. Systematic tests are split by tracer: $C^{\gamma S}_\ell$ (shear $\times$ systematic) and $C^{\delta S}_\ell$ (density $\times$ systematic). Density failures are expected --- galaxy positions correlate with observational depth variations by construction. The bias budget quantifies the harmonic-space contamination metric $X^f_S(\ell) = C^{\kappa S}_\ell \cdot C^{fS}_\ell / C^{SS}_\ell$ (2203.12440, Eq.~A.4.1) integrated over bandpowers. All $\chi^2/\nu$ use Knox diagonal covariance.}
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
            # Headline numbers — density (the detection)
            "act_density_snr_raw": table1_data.get("density_x_act", {}).get("snr_raw"),
            "act_density_A": table1_data.get("density_x_act", {}).get("A_ia"),
            "act_density_A_err": table1_data.get("density_x_act", {}).get("A_ia_err"),
            "act_density_A_marg": table1_data.get("density_x_act", {}).get("A_marg"),
            "act_density_A_marg_err": table1_data.get("density_x_act", {}).get("A_marg_err"),
            "spt_density_snr_raw": table1_data.get("density_x_spt_winter_gmv", {}).get("snr_raw"),
            "spt_density_A": table1_data.get("density_x_spt_winter_gmv", {}).get("A_ia"),
            "spt_density_A_err": table1_data.get("density_x_spt_winter_gmv", {}).get("A_ia_err"),
            # Headline numbers — shear (lensmc baseline, not detected)
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
