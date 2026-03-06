"""Contamination overview: X^f_S per systematic map.

2x11 grid — one row per systematic map, two columns:
  - Left: density (delta_g) x CMB kappa contamination
  - Right: shear (gamma) x CMB kappa contamination

Each subplot titled "{Sysmap} x {Tracer}". Color encodes CMB experiment
(ACT salmon, SPT teal); linestyle+marker encodes shear method (lensmc
solid+circle, metacal dashed+square). Density column has only two curves
(ACT, SPT) with default solid+circle style.

X^f_S(ell) = C^{kappa S} . C^{fS} / C^{SS} with propagated Knox error bars.
Same units as signal C^{f kappa} for direct comparison.

Caveat: C^{SS} noise bias at high ell makes X a lower bound on true contamination.

Reference: Chang+ 2203.12440 section A.4 (Eq. A.4.1, line 808).
"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.plot_utils import (
    FW, save_evidence, setup_theme, COLORS, LABELS, MARKERS, SYSMAP_LABELS,
)


snakemake = snakemake  # type: ignore # noqa: F821

output_png = Path(snakemake.output.png)
evidence_path = Path(snakemake.output.evidence)
sys_dir = Path(snakemake.input.sys_dir)
cmbk_sys_dir = Path(snakemake.input.cmbk_sys_dir)
cmbk_experiments = list(snakemake.params.cmbk_experiments)
shear_methods = list(snakemake.params.shear_methods)
sysmap_keys = list(snakemake.params.sys_map_keys)

snakemake_log(snakemake, f"Contamination overview: {len(sysmap_keys)} sysmaps")

# -- Ordered sysmap display -----------------------------------------------
sysmap_order = [k for k in [
    "stellar_density", "galactic_extinction", "exposures", "zodiacal_light",
    "saa", "psf", "noise", "persistence", "star_brightness_mean",
    "galaxy_number_density", "phz_quality",
] if k in sysmap_keys]

n_maps = len(sysmap_order)

# -- Tracer column labels (for subplot titles) ----------------------------
TRACER_LABELS = {
    "density": r"Galaxy density",
    "lensmc": r"LensMC shear",
    "metacal": r"Metacal shear",
}


def load_npz(path):
    """Load NPZ, return dict or None."""
    if not path.exists():
        return None
    data = np.load(str(path), allow_pickle=True)
    return {k: data[k] for k in data.files}


def compute_X(cl_fS, knox_var_fS, cl_kS, knox_var_kS, cl_SS):
    """Compute X = C^{kappa S} . C^{fS} / C^{SS} and propagated error."""
    safe_SS = np.where(np.abs(cl_SS) > 0, cl_SS, np.nan)
    X = cl_kS * cl_fS / safe_SS

    sigma2 = np.zeros_like(X)
    if knox_var_kS is not None:
        sigma2 += (cl_fS / safe_SS) ** 2 * np.abs(knox_var_kS)
    if knox_var_fS is not None:
        sigma2 += (cl_kS / safe_SS) ** 2 * np.abs(knox_var_fS)

    return X, np.sqrt(sigma2)


# -- Styling: color = CMB experiment, linestyle+marker = shear method ------
# lensmc: solid + circle; metacal: dashed + square; density: solid + circle
LINESTYLES = {"lensmc": "-", "metacal": "--", "density": "-"}
METHOD_MARKERS = {"lensmc": "o", "metacal": "s", "density": "o"}

# -- Column definitions ----------------------------------------------------
columns = [
    {"methods": ["density"]},
    {"methods": shear_methods},
]

# -- Figure layout: margins in inches, content-driven dimensions -----------
setup_theme()

row_height = 1.8
col_width = FW
margin_left = 0.75   # inches
margin_right = 0.15  # inches
margin_top = 0.15    # inches
margin_bottom = 0.55 # inches
col_gap = 0.6        # inches between columns

fig_width = margin_left + 2 * col_width + col_gap + margin_right
fig_height = margin_top + n_maps * row_height + margin_bottom

fig, axes = plt.subplots(
    n_maps, 2, figsize=(fig_width, fig_height),
    gridspec_kw={
        "hspace": 0.35,
        "wspace": col_gap / col_width,
        "left": margin_left / fig_width,
        "right": 1 - margin_right / fig_width,
        "top": 1 - margin_top / fig_height,
        "bottom": margin_bottom / fig_height,
    },
    squeeze=False,
)

evidence_data = {}

for row_idx, sysmap in enumerate(sysmap_order):
    sysmap_label = SYSMAP_LABELS.get(sysmap, sysmap)

    # Load C^{kappa S} and C^{SS} per CMB experiment
    kS_by_cmbk = {}
    for cmbk in cmbk_experiments:
        d = load_npz(cmbk_sys_dir / f"{sysmap}_x_{cmbk}_cls.npz")
        if d is not None:
            kS_by_cmbk[cmbk] = {
                "ells": d["ells"],
                "cls": d["cls"][0],
                "knox_var": d.get("knox_var"),
                "cl_sys_auto": d.get("cl_sys_auto"),
            }

    for col_idx, col_def in enumerate(columns):
        ax = axes[row_idx, col_idx]
        curve_idx = 0

        for method in col_def["methods"]:
            fS_data = load_npz(sys_dir / f"{sysmap}_x_{method}_binall_cls.npz")
            if fS_data is None:
                continue

            cl_fS = fS_data["cls"][0]
            knox_var_fS = fS_data.get("knox_var")

            for cmbk in cmbk_experiments:
                if cmbk not in kS_by_cmbk:
                    continue
                kS = kS_by_cmbk[cmbk]
                ells = kS["ells"]
                cl_kS = kS["cls"]
                knox_var_kS = kS.get("knox_var")
                cl_SS = kS.get("cl_sys_auto")

                if cl_SS is None:
                    continue

                X, sigma_X = compute_X(cl_fS, knox_var_fS, cl_kS, knox_var_kS, cl_SS)
                prefactor = ells * (ells + 1) / (2 * np.pi)
                y = prefactor * X
                yerr = prefactor * sigma_X

                # Style: color = CMB experiment, marker+ls = method
                cmbk_label = LABELS.get(cmbk, cmbk)
                method_label = LABELS.get(method, method)
                color = COLORS.get(cmbk, "black")
                marker = METHOD_MARKERS.get(method, "o")
                ls = LINESTYLES.get(method, "-")

                if len(col_def["methods"]) > 1:
                    label = f"{method_label} $\\times$ {cmbk_label}"
                else:
                    label = cmbk_label

                # Small ell offset to separate overlapping points
                n_curves = len(col_def["methods"]) * len(cmbk_experiments)
                spread = 0.03
                frac = (curve_idx - (n_curves - 1) / 2) / max(n_curves - 1, 1)
                x = ells * (1 + spread * frac)

                ax.errorbar(x, y, yerr=yerr, fmt=marker, ls=ls,
                            color=color, ms=5.5, lw=1.2, elinewidth=0.7,
                            capsize=2, label=label, alpha=0.85)
                curve_idx += 1

                # Store evidence
                ev_key = f"{sysmap}_{method}_{cmbk}"
                finite = np.isfinite(X)
                evidence_data[ev_key] = {
                    "max_abs_X_Dl": round(float(np.nanmax(np.abs(y[finite]))), 10) if np.any(finite) else None,
                    "mean_abs_X_Dl": round(float(np.nanmean(np.abs(y[finite]))), 10) if np.any(finite) else None,
                }

        ax.axhline(0, color="0.4", ls="-", lw=0.6, alpha=0.6)
        ax.set_xscale("log")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3),
                            useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.tick_params(labelsize=6)

        # Per-subplot title: "{Sysmap} x {Tracer}"
        tracer_label = TRACER_LABELS.get(col_def["methods"][0], "Shear")
        if len(col_def["methods"]) > 1:
            tracer_label = "Shear"
        ax.set_title(f"{sysmap_label} $\\times$ {tracer_label}",
                     fontsize=8, pad=3)

        # Bottom row: x-axis label
        if row_idx == n_maps - 1:
            ax.set_xlabel(r"Multipole $\ell$")
        else:
            ax.tick_params(labelbottom=False)

        # y-axis label: left column only
        if col_idx == 0:
            ax.set_ylabel(r"$\ell(\ell+1)\,X^f_S\,/\,2\pi$", fontsize=7)

        # Legend in top-right panel only
        if row_idx == 0 and col_idx == 1:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.9,
                      handlelength=3.0, borderpad=0.4)

output_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_png, bbox_inches="tight", dpi=150)
plt.close(fig)
snakemake_log(snakemake, f"Saved: {output_png}")

# -- Evidence --------------------------------------------------------------
evidence_doc = {
    "id": "plot_contamination_overview",
    "generated": datetime.now(timezone.utc).isoformat(),
    "output": {"png": output_png.name},
    "evidence": {
        "n_sysmaps": n_maps,
        "cmbk_experiments": cmbk_experiments,
        "shear_methods": shear_methods,
        "per_combination": evidence_data,
    },
}

evidence_path.parent.mkdir(parents=True, exist_ok=True)
save_evidence(evidence_doc, evidence_path, snakemake)
snakemake_log(snakemake, f"Saved: {evidence_path}")
