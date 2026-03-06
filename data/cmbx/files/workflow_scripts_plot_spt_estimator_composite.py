"""SPT estimator comparison in normalized units: Delta C_ell / sigma.

PP (polarization-only) kappa is foreground-robust; TT and GMV are compared to
PP through normalized residuals. Values near zero indicate no estimator bias.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import chi2 as chi2_dist


snakemake = snakemake  # type: ignore # noqa: F821

method = snakemake.wildcards.method
output_png = Path(snakemake.output.png)
evidence_path = Path(snakemake.output.evidence)
log_path = Path(snakemake.log[0]) if getattr(snakemake, "log", None) else None


def _log(message: str) -> None:
    print(message)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            print(message, file=f)


def _load_cross_spectrum(npz_path: Path) -> dict:
    data = np.load(str(npz_path), allow_pickle=True)
    cls = data["cls"]
    ells = data["ells"]
    cov = data["cov"] if "cov" in data else None
    knox_var = data["knox_var"] if "knox_var" in data else None
    metadata = data["metadata"].item() if "metadata" in data else {}
    err = None
    if cov is not None:
        nbins = len(ells)
        n_total = cov.shape[0]
        n_cls = n_total // nbins if n_total > nbins else 1
        if n_cls > 1:
            cov_4d = cov.reshape(nbins, n_cls, nbins, n_cls)
            err = np.sqrt(np.maximum(np.diag(cov_4d[:, 0, :, 0]), 0))
        else:
            err = np.sqrt(np.maximum(np.diag(cov)[:nbins], 0))
    elif knox_var is not None:
        err = np.sqrt(np.maximum(knox_var[:len(ells)], 0))

    return {
        "ells": ells,
        "cl_e": cls[0],
        "err": err if err is not None else np.zeros_like(cls[0]),
        "metadata": metadata,
    }


def _save_evidence(doc: dict, path: Path) -> None:
    doc.setdefault("generated", datetime.now(timezone.utc).isoformat())
    doc["input"] = {k: str(v) for k, v in dict(snakemake.input).items()}
    doc["output"] = {k: str(v) for k, v in dict(snakemake.output).items()}
    doc["params"] = {k: v for k, v in dict(snakemake.params).items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, default=str)


def _parse_wildcards_from_filename(npz: Path) -> tuple[str | None, str | None]:
    """Parse bin/cmbk from '<method>_bin<id>_x_<cmbk>_cls.npz' filename."""
    match = re.search(r"_bin([^_]+)_x_(.+)_cls$", npz.stem)
    if not match:
        return None, None
    return match.group(1), match.group(2)


_log(f"Building SPT estimator composite (normalized residuals) for method={method}")

bins = [str(b) for b in snakemake.params.get("bins", ["1", "2", "3", "4", "5", "6"])]
pp_key = "spt_winter_pp"
colors = {"spt_winter_tt": "#E8927C", "spt_winter_gmv": "#5B9EA6"}

# Curves to show: TT−PP and GMV−PP
diff_curves = {
    "spt_winter_tt": {
        "label": r"TT $-$ PP",
        "color": colors["spt_winter_tt"],
        "marker": "v",
    },
    "spt_winter_gmv": {
        "label": r"GMV $-$ PP",
        "color": colors["spt_winter_gmv"],
        "marker": "s",
    },
}

# Load all spectra
spectra = {}
for npz_path in snakemake.input.npz_files:
    npz = Path(npz_path)
    if not npz.exists():
        continue
    spec = _load_cross_spectrum(npz)
    bin_id, cmbk_from_name = _parse_wildcards_from_filename(npz)
    if bin_id is None:
        continue
    cmbk = spec["metadata"].get("cmbk") or cmbk_from_name
    if cmbk is None:
        continue
    spectra[(bin_id, cmbk)] = {
        "ells": spec["ells"],
        "cl_e": spec["cl_e"],
        "err": spec["err"] if spec["err"] is not None else np.zeros_like(spec["cl_e"]),
    }

sns.set_theme(style="ticks")
mplstyle = Path(__file__).parent / "paper.mplstyle"
if mplstyle.exists():
    plt.style.use(str(mplstyle))
fig, axes = plt.subplots(2, 3, figsize=(8.0, 6.0), sharey=True)
axes = axes.ravel()

n_comp = 0
n_fail = 0
all_ptes = []
chi2_dof_vals = []
flat_metrics = {}
missing_pairs = []
n_valid_bandpowers_total = 0
expected_comparisons = len(bins) * len(diff_curves)

for idx, bin_id in enumerate(bins):
    ax = axes[idx]
    pp = spectra.get((bin_id, pp_key))
    if pp is None:
        ax.text(0.5, 0.5, "missing PP", transform=ax.transAxes, ha="center", va="center", color="0.5")
        ax.set_title(f"z-bin {bin_id}", fontsize=11, fontweight="bold")
        continue

    ells = pp["ells"]
    cl_pp = pp["cl_e"]
    err_pp = pp["err"]

    for cmbk, cfg in diff_curves.items():
        var = spectra.get((bin_id, cmbk))
        if var is None:
            missing_pairs.append(f"bin{bin_id}:{cmbk}")
            continue

        cl_var = var["cl_e"]
        err_var = var["err"]
        ells_var = var["ells"]

        if np.array_equal(ells, ells_var):
            ells_use = ells
            cl_pp_use, err_pp_use = cl_pp, err_pp
            cl_var_use, err_var_use = cl_var, err_var
        else:
            common_ells, i_pp, i_var = np.intersect1d(
                ells, ells_var, assume_unique=False, return_indices=True
            )
            if common_ells.size == 0:
                missing_pairs.append(f"bin{bin_id}:{cmbk}:no_common_ell")
                continue
            ells_use = common_ells
            cl_pp_use, err_pp_use = cl_pp[i_pp], err_pp[i_pp]
            cl_var_use, err_var_use = cl_var[i_var], err_var[i_var]

        # Difference spectrum: variant − PP, then normalize by per-bandpower sigma.
        diff = cl_var_use - cl_pp_use
        err_diff = np.sqrt(err_var_use**2 + err_pp_use**2)
        valid = err_diff > 0
        n_valid_bandpowers_total += int(np.sum(valid))
        delta_cl_over_sigma = np.zeros_like(diff)
        delta_cl_over_sigma[valid] = diff[valid] / err_diff[valid]

        ax.errorbar(
            ells_use[valid],
            delta_cl_over_sigma[valid],
            yerr=np.ones(np.sum(valid)),
            fmt=cfg["marker"],
            color=cfg["color"],
            markersize=4,
            capsize=1.5,
            elinewidth=0.8,
            label=cfg["label"] if idx == 0 else None,
            zorder=3,
        )

        # Chi2 test in normalized units: residuals should be N(0,1) under H0.
        if np.any(valid):
            chi2_val = float(np.sum(delta_cl_over_sigma[valid] ** 2))
            dof = int(np.sum(valid))
            pte = float(1.0 - chi2_dist.cdf(chi2_val, dof))
            chi2_dof = chi2_val / dof
            n_comp += 1
            all_ptes.append(pte)
            chi2_dof_vals.append(chi2_dof)
            if pte < 0.05:
                n_fail += 1
            suffix = f"bin{bin_id}_{cmbk.split('_')[-1]}"
            flat_metrics[f"{suffix}_chi2_dof"] = round(chi2_dof, 3)
            flat_metrics[f"{suffix}_pte"] = round(pte, 4)
            flat_metrics[f"{suffix}_median_abs_delta_cl_over_sigma"] = round(
                float(np.median(np.abs(delta_cl_over_sigma[valid]))), 3
            )
            flat_metrics[f"{suffix}_max_abs_delta_cl_over_sigma"] = round(
                float(np.max(np.abs(delta_cl_over_sigma[valid]))), 3
            )

    ax.axhspan(-1.0, 1.0, color="0.92", zorder=0)
    ax.axhline(0.0, color="gray", ls="--", lw=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(30, 4000)
    ax.set_ylim(-4.5, 4.5)
    ax.set_title(f"z-bin {bin_id}", fontsize=11, fontweight="bold")

    if idx >= 3:
        ax.set_xlabel(r"Multipole $\ell$")
    if idx % 3 == 0:
        ax.set_ylabel(r"$\Delta C_\ell / \sigma$")

handles, labels = axes[0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.0), frameon=False)

if chi2_dof_vals:
    fig.text(
        0.5, -0.01,
        (f"{method} $\\times$ SPT | {n_comp} null tests vs PP | "
         f"median $\\chi^2/\\nu$={np.median(chi2_dof_vals):.2f} "
         r"(conservative: shared galaxy noise $\rightarrow$ correlated errors)"
         f" | failures(p$<$0.05)={n_fail}"),
        ha="center", fontsize=9, style="italic", color="0.4",
    )

sns.despine()
fig.tight_layout(rect=[0, 0.02, 1, 0.96])

output_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_png, bbox_inches="tight")
plt.close(fig)
_log(f"Saved: {output_png}")

evidence_doc = {
    "id": f"spt_estimator_composite_{method}",
    "generated": datetime.now(timezone.utc).isoformat(),
    "evidence": {
        "method": method,
        "design": "PP_baseline_normalized_residuals",
        "y_units": "delta_cl_over_sigma",
        "expected_comparisons": expected_comparisons,
        "n_comparisons": n_comp,
        "n_missing_comparisons": expected_comparisons - n_comp,
        "missing_comparisons": missing_pairs,
        "comparison_completeness": round(n_comp / expected_comparisons, 3) if expected_comparisons else None,
        "all_expected_comparisons_present": n_comp == expected_comparisons,
        "n_consistent_5pct": n_comp - n_fail,
        "n_valid_bandpowers_total": n_valid_bandpowers_total,
        "median_pte": round(float(np.median(all_ptes)), 3) if all_ptes else None,
        "min_pte": round(float(np.min(all_ptes)), 4) if all_ptes else None,
        "median_chi2_dof": round(float(np.median(chi2_dof_vals)), 3) if chi2_dof_vals else None,
        "residual_definition": "(C_ell_variant - C_ell_PP) / sqrt(sigma_variant^2 + sigma_PP^2)",
        **flat_metrics,
    },
}
_save_evidence(evidence_doc, evidence_path)
_log(f"Saved: {evidence_path}")
