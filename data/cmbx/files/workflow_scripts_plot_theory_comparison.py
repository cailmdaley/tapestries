"""Plot measured C^{Eκ}_ℓ overlaid with theory prediction.

Shows all 6 tomographic bins on separate panels in a 2×3 grid.
Theory from eDR1like (Planck 2018 fiducial + SPV3 bias model).
Overlays NLA IA prediction if available. IA params from config.yaml.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.plot_utils import FW, FH, setup_theme, LABELS, BIN_PALETTE, TOM_BINS, save_evidence
from dr1_notebooks.scratch.cdaley.spectrum_utils import load_cross_spectrum, load_evidence


snakemake = snakemake  # type: ignore # noqa: F821

method = snakemake.wildcards.method
cmbk = snakemake.wildcards.cmbk
theory_npz_path = Path(snakemake.input.theory_npz)
output_png = Path(snakemake.output.png)
evidence_path = Path(snakemake.output.evidence)

_ia = snakemake.config.get("intrinsic_alignment", {})
_ia_str = f"{_ia.get('model', 'NLA-z')} (A_IA={_ia.get('A_IA', 1.72)}, eta={_ia.get('eta_IA', -0.41)})"

snakemake_log(snakemake, f"Plotting theory comparison: {method} x {cmbk}")

# --- Load theory ---
theory_data = np.load(str(theory_npz_path), allow_pickle=True)
ell_theory = theory_data["ell_theory"]
has_ia = f"cl_ia_binned_bin1" in theory_data

if has_ia:
    snakemake_log(snakemake, "  NLA IA theory available — will overlay")

# --- Plot: 2×3 grid ---
setup_theme("ticks")
fig, axes = plt.subplots(2, 3, figsize=(2*FW, 2*FH), sharex=True)
axes = axes.flatten()

palette = BIN_PALETTE

bin_chi2_vanilla = []
bin_chi2_ia = []

for i, bin_id in enumerate(TOM_BINS):
    ax = axes[i]

    # Load measured data
    data_npz_path = Path(snakemake.input.data_dir) / f"{method}_bin{bin_id}_x_{cmbk}_cls.npz"
    if not data_npz_path.exists():
        ax.set_visible(False)
        continue

    spec = load_cross_spectrum(data_npz_path)
    ells = spec["ells"]
    cl_e = spec["cl_e"]
    knox_var = spec["knox_var"]
    knox_err = (
        np.sqrt(np.maximum(knox_var[:len(ells)], 0))
        if knox_var is not None else spec["err"]
    )
    ev = load_evidence(data_npz_path)
    snr = ev.get("snr")

    # Plot measured (Knox error bars — NaMaster diagonal has artifact at ℓ≈2700)
    prefactor = ells
    y = prefactor * cl_e
    yerr = prefactor * knox_err if knox_err is not None else None

    ax.errorbar(
        ells, y, yerr=yerr,
        fmt="o", color=palette[i], markersize=5,
        capsize=2, elinewidth=1, label="Data",
    )

    # Plot binned vanilla theory
    key_ells = f"ells_binned_bin{bin_id}"
    key_cl = f"cl_binned_bin{bin_id}"
    if key_ells in theory_data and key_cl in theory_data:
        ells_th = theory_data[key_ells]
        cl_th = theory_data[key_cl]
        ax.plot(
            ells_th, ells_th * cl_th,
            "k-", lw=1.5, alpha=0.8, label="Planck 2018",
        )

    # Plot binned IA theory
    key_ia_cl = f"cl_ia_binned_bin{bin_id}"
    if has_ia and key_ells in theory_data and key_ia_cl in theory_data:
        ells_th = theory_data[key_ells]
        cl_ia = theory_data[key_ia_cl]
        ax.plot(
            ells_th, ells_th * cl_ia,
            color="crimson", ls="--", lw=1.5, alpha=0.8,
            label="Planck 2018 + NLA IA",
        )

    # Unbinned vanilla theory (thin)
    key_unbinned = f"cl_theory_bin{bin_id}"
    if key_unbinned in theory_data:
        cl_th_full = theory_data[key_unbinned]
        ell_plot = ell_theory[2:]
        cl_plot = cl_th_full[2:]
        ax.plot(
            ell_plot, ell_plot * cl_plot,
            color="gray", lw=0.8, alpha=0.4, zorder=0,
        )

    # Unbinned IA theory (thin)
    key_ia_unbinned = f"cl_ia_theory_bin{bin_id}"
    if has_ia and key_ia_unbinned in theory_data:
        cl_ia_full = theory_data[key_ia_unbinned]
        ell_plot = ell_theory[2:]
        cl_plot = cl_ia_full[2:]
        ax.plot(
            ell_plot, ell_plot * cl_plot,
            color="crimson", lw=0.8, alpha=0.3, ls="--", zorder=0,
        )

    ax.axhline(0, color="gray", ls="--", lw=0.6, alpha=0.4)

    # Compute chi2 vs vanilla theory using Knox variance
    # (NaMaster diagonal has near-zero artifact at ℓ≈2700)
    chi2_str = ""
    if key_ells in theory_data and key_cl in theory_data and knox_err is not None:
        cl_th_binned = theory_data[key_cl]
        n_common = min(len(cl_e), len(cl_th_binned))
        residual = cl_e[:n_common] - cl_th_binned[:n_common]
        var = knox_err[:n_common] ** 2
        good = var > 0
        if np.sum(good) > 0:
            chi2_v = float(np.sum(residual[good] ** 2 / var[good]))
            dof_v = int(np.sum(good))
            pte_v = float(1.0 - stats.chi2.cdf(chi2_v, dof_v))
            chi2_str = f"$\\chi^2$={chi2_v:.1f}/{dof_v}"
            bin_chi2_vanilla.append({
                "bin": bin_id, "chi2": round(chi2_v, 1),
                "dof": dof_v, "pte": round(pte_v, 4),
            })

    # Compute chi2 vs IA theory (also Knox variance)
    chi2_ia_str = ""
    if has_ia and key_ia_cl in theory_data and knox_err is not None:
        cl_ia_binned = theory_data[key_ia_cl]
        n_common = min(len(cl_e), len(cl_ia_binned))
        residual_ia = cl_e[:n_common] - cl_ia_binned[:n_common]
        var = knox_err[:n_common] ** 2
        good = var > 0
        if np.sum(good) > 0:
            chi2_ia = float(np.sum(residual_ia[good] ** 2 / var[good]))
            dof_ia = int(np.sum(good))
            pte_ia = float(1.0 - stats.chi2.cdf(chi2_ia, dof_ia))
            chi2_ia_str = f"IA: {chi2_ia:.1f}/{dof_ia}"
            bin_chi2_ia.append({
                "bin": bin_id, "chi2": round(chi2_ia, 1),
                "dof": dof_ia, "pte": round(pte_ia, 4),
            })

    # Panel label
    panel_label = chr(ord("a") + i)
    ax.text(
        0.05, 0.93, f"({panel_label})", transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top",
    )

    title = f"bin {bin_id}"
    if snr is not None:
        title += f" (S/N={snr:.1f})"
    if chi2_str:
        title += f"\n{chi2_str}"
    if chi2_ia_str:
        title += f"  {chi2_ia_str}"
    ax.set_title(title, fontsize=9)

    if i >= 3:
        ax.set_xlabel(r"$\ell$")
    if i % 3 == 0:
        ylabel = r"$\ell\, C_\ell^{\kappa\delta}$" if method == "density" else r"$\ell\, C_\ell^{\kappa E}$"
        ax.set_ylabel(ylabel)

    ax.set_xscale("log")

    if i == 0:
        ax.legend(fontsize=7, loc="upper right")

method_label = LABELS.get(method, method)
cmbk_label = LABELS.get(cmbk, cmbk.upper())
fig.suptitle(
    f"{method_label} $\\times$ {cmbk_label}: data vs theory",
    fontsize=14, y=0.98,
)

sns.despine()
fig.tight_layout()

output_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_png, bbox_inches="tight")
plt.close(fig)

snakemake_log(snakemake, f"Saved: {output_png}")

# --- Evidence ---
from datetime import datetime, timezone

evidence_doc = {
    "id": f"theory_comparison_{method}_x_{cmbk}",
    "generated": datetime.now(timezone.utc).isoformat(),
    "artifacts": {"png": output_png.name},
}

if bin_chi2_vanilla:
    chi2_total = sum(r["chi2"] for r in bin_chi2_vanilla)
    dof_total = sum(r["dof"] for r in bin_chi2_vanilla)
    pte_total = float(1.0 - stats.chi2.cdf(chi2_total, dof_total))
    evidence_doc["evidence"] = {
        "per_bin_vanilla": bin_chi2_vanilla,
        "vanilla_chi2": round(chi2_total, 1),
        "vanilla_dof": dof_total,
        "vanilla_pte": round(pte_total, 4),
        "method": method,
        "cmbk": cmbk,
    }
    snakemake_log(
        snakemake,
        f"  Vanilla theory: chi2={chi2_total:.1f} (dof={dof_total}, PTE={pte_total:.4f})",
    )

if bin_chi2_ia:
    chi2_ia_total = sum(r["chi2"] for r in bin_chi2_ia)
    dof_ia_total = sum(r["dof"] for r in bin_chi2_ia)
    pte_ia_total = float(1.0 - stats.chi2.cdf(chi2_ia_total, dof_ia_total))
    ev = evidence_doc.setdefault("evidence", {})
    ev["per_bin_ia"] = bin_chi2_ia
    ev["ia_chi2"] = round(chi2_ia_total, 1)
    ev["ia_dof"] = dof_ia_total
    ev["ia_pte"] = round(pte_ia_total, 4)
    ev["ia_model"] = _ia_str
    snakemake_log(
        snakemake,
        f"  NLA IA theory: chi2={chi2_ia_total:.1f} (dof={dof_ia_total}, PTE={pte_ia_total:.4f})",
    )

evidence_path.parent.mkdir(parents=True, exist_ok=True)
save_evidence(evidence_doc, evidence_path, snakemake)
snakemake_log(snakemake, f"Saved: {evidence_path}")
