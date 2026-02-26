"""Raw ingredient spectra for the contamination metric, per systematic map.

Shows the three cross-spectra that compose X^f_S(ℓ) = C^{κS} · C^{fS} / C^{SS}:

  - C^{fS}_ℓ : tracer × systematic, all-bin (top panel, 3 curves: lensmc, metacal, density)
  - C^{κS}_ℓ : CMB κ × systematic (middle panel, ACT + SPT)
  - C^{SS}_ℓ : systematic auto-spectrum per footprint (bottom panel)

One figure per systematic map. Uses only bin_id="all".
"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.plot_utils import (
    FW, FH, save_evidence, setup_theme, COLORS, LABELS, MARKERS, SYSMAP_LABELS,
)


snakemake = snakemake  # type: ignore # noqa: F821

sysmap_name = snakemake.wildcards.sysmap
output_png = Path(snakemake.output.png)
evidence_path = Path(snakemake.output.evidence)
sys_dir = Path(snakemake.input.sys_dir)
cmbk_sys_dir = Path(snakemake.input.cmbk_sys_dir)
cmbk_experiments = snakemake.params.cmbk_experiments
methods = snakemake.params.methods

snakemake_log(snakemake, f"Systematic spectra: {sysmap_name} (all methods)")

# ── Load C^{fS} all-bin per method ───────────────────────────────────────────
fS_data = {}  # method -> dict with ells, cls, knox_var
for method in methods:
    npz_path = sys_dir / f"{sysmap_name}_x_{method}_binall_cls.npz"
    if not npz_path.exists():
        snakemake_log(snakemake, f"  Missing: {npz_path.name}")
        continue
    data = np.load(str(npz_path), allow_pickle=True)
    fS_data[method] = {
        "ells": data["ells"],
        "cls": data["cls"][0],   # E-mode for spin-2; [0] for spin-0
        "knox_var": data["knox_var"] if "knox_var" in data else None,
    }

# ── Load C^{κS} per CMB experiment ───────────────────────────────────────────
kS_data = {}  # cmbk -> dict with ells, cls, cl_sys_auto
for cmbk in cmbk_experiments:
    npz_path = cmbk_sys_dir / f"{sysmap_name}_x_{cmbk}_cls.npz"
    if not npz_path.exists():
        snakemake_log(snakemake, f"  Missing: {npz_path.name}")
        continue
    data = np.load(str(npz_path), allow_pickle=True)
    kS_data[cmbk] = {
        "ells": data["ells"],
        "cls": data["cls"][0],
        "cl_sys_auto": data["cl_sys_auto"] if "cl_sys_auto" in data else None,
    }

if not fS_data and not kS_data:
    snakemake_log(snakemake, "No data found, creating placeholder")
    setup_theme()
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.text(0.5, 0.5, "No systematic spectra available",
            ha="center", va="center", fontsize=14, transform=ax.transAxes)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close()
    evidence_doc = {
        "id": f"systematic_spectra_{sysmap_name}",
        "generated": datetime.now(timezone.utc).isoformat(),
        "evidence": {"status": "no_data"},
    }
    save_evidence(evidence_doc, evidence_path, snakemake)
    raise SystemExit(0)

# ── Plot ──────────────────────────────────────────────────────────────────────
setup_theme()
sysmap_label = SYSMAP_LABELS.get(sysmap_name, sysmap_name)

# Small ell offsets to avoid overlapping markers
def ell_offset(ells, idx, n_total):
    """Multiplicative offset to separate overlapping points."""
    spread = 0.03  # ±3% in log space
    frac = (idx - (n_total - 1) / 2) / max(n_total - 1, 1)
    return ells * (1 + spread * frac)

fig, axes = plt.subplots(3, 1, figsize=(1.6 * FW, 3.2 * FH), sharex=True,
                         gridspec_kw={"hspace": 0.18})

# --- Panel 1: C^{fS} per method (tracer × systematic) ---
# Shear methods on left y-axis, density on right y-axis (different amplitudes).
ax_shear = axes[0]
shear_methods_present = [m for m in methods if m in fS_data and m != "density"]
density_present = "density" in fS_data

n_methods = len([m for m in methods if m in fS_data])
for i, method in enumerate(methods):
    if method not in fS_data:
        continue
    d = fS_data[method]
    ells = d["ells"]
    cl = d["cls"]
    prefactor = ells * (ells + 1) / (2 * np.pi)
    color = COLORS.get(method, "black")
    marker = MARKERS.get(method, "o")
    label = LABELS.get(method, method)
    y = prefactor * cl
    x = ell_offset(ells, i, n_methods)

    # Plot density on the right axis, shear on the left
    if method == "density" and shear_methods_present:
        if not hasattr(ax_shear, "_twin_ax"):
            ax_density = ax_shear.twinx()
            ax_shear._twin_ax = ax_density
        else:
            ax_density = ax_shear._twin_ax
        target_ax = ax_density
    else:
        target_ax = ax_shear

    if d["knox_var"] is not None:
        yerr = prefactor * np.sqrt(np.abs(d["knox_var"]))
        target_ax.errorbar(x, y, yerr=yerr, fmt=f"{marker}-",
                    color=color, ms=4, lw=0.8, elinewidth=0.5, capsize=1.5,
                    label=label)
    else:
        target_ax.plot(x, y, f"{marker}-", color=color, ms=4, lw=0.8, label=label)

ax_shear.axhline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
ax_shear.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{fS}\,/\,2\pi$ (shear)")
ax_shear.set_title(rf"Tracer $\times$ {sysmap_label}", fontsize=10, loc="left")

if density_present and shear_methods_present and hasattr(ax_shear, "_twin_ax"):
    ax_density = ax_shear._twin_ax
    ax_density.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{fS}\,/\,2\pi$ (density)",
                          color=COLORS.get("density", "green"))
    ax_density.tick_params(axis="y", labelcolor=COLORS.get("density", "green"))
    # Combined legend from both axes
    h1, l1 = ax_shear.get_legend_handles_labels()
    h2, l2 = ax_density.get_legend_handles_labels()
    ax_shear.legend(h1 + h2, l1 + l2, fontsize=8, loc="best", framealpha=0.9)
else:
    ax_shear.legend(fontsize=8, loc="best", framealpha=0.9)

# --- Panel 2: C^{κS} per CMB experiment ---
ax = axes[1]
n_cmbk = len(kS_data)
for i, (cmbk, d) in enumerate(kS_data.items()):
    ells = d["ells"]
    cl = d["cls"]
    prefactor = ells * (ells + 1) / (2 * np.pi)
    color = COLORS.get(cmbk, "black")
    marker = MARKERS.get(cmbk, "o")
    label = LABELS.get(cmbk, cmbk)
    x = ell_offset(ells, i, n_cmbk)
    ax.plot(x, prefactor * cl, f"{marker}-", color=color, ms=4, lw=1.0, label=label)

ax.axhline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
ax.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\kappa S}\,/\,2\pi$")
ax.set_title(rf"CMB $\kappa$ $\times$ {sysmap_label}", fontsize=10, loc="left")
ax.legend(fontsize=8, loc="best", framealpha=0.9)

# --- Panel 3: C^{SS} (systematic auto-spectrum per footprint) ---
ax = axes[2]
plotted_SS = False
for i, (cmbk, d) in enumerate(kS_data.items()):
    if d["cl_sys_auto"] is None:
        continue
    ells = d["ells"]
    cl_SS = d["cl_sys_auto"]
    prefactor = ells * (ells + 1) / (2 * np.pi)
    color = COLORS.get(cmbk, "gray")
    marker = MARKERS.get(cmbk, "o")
    label = f"{LABELS.get(cmbk, cmbk)} footprint"
    x = ell_offset(ells, i, n_cmbk)
    ax.plot(x, prefactor * cl_SS, f"{marker}-", color=color, ms=4, lw=1.0,
            label=label, alpha=0.8)
    plotted_SS = True

if not plotted_SS:
    ax.text(0.5, 0.5, r"$C^{SS}_\ell$ not available",
            ha="center", va="center", fontsize=11, transform=ax.transAxes,
            color="0.5")

ax.axhline(0, color="gray", ls="--", lw=0.5, alpha=0.5)
ax.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{SS}\,/\,2\pi$")
ax.set_xlabel(r"Multipole $\ell$")
ax.set_xscale("log")
ax.set_title(rf"{sysmap_label} auto-spectrum", fontsize=10, loc="left")
ax.legend(fontsize=8, loc="best", framealpha=0.9)

fig.suptitle(f"Contamination ingredients: {sysmap_label}",
             fontsize=12, fontweight="bold")

output_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_png, bbox_inches="tight", dpi=150)
plt.close(fig)
snakemake_log(snakemake, f"Saved: {output_png}")

# ── Evidence ──────────────────────────────────────────────────────────────────
method_summaries = {}
for method, d in fS_data.items():
    cl = d["cls"]
    kv = d["knox_var"]
    if kv is not None and np.any(kv > 0):
        snr = float(np.mean(cl / np.sqrt(np.abs(kv) + 1e-40)))
    else:
        snr = None
    method_summaries[str(method)] = {
        "mean_abs_cl_fS": round(float(np.mean(np.abs(cl))), 12),
        "knox_snr": round(snr, 4) if snr is not None else None,
    }

kS_summaries = {}
for cmbk, d in kS_data.items():
    cl = d["cls"]
    kS_summaries[str(cmbk)] = {
        "mean_abs_cl_kS": round(float(np.mean(np.abs(cl))), 12),
        "has_cl_SS": d["cl_sys_auto"] is not None,
    }

evidence_doc = {
    "id": f"systematic_spectra_{sysmap_name}",
    "generated": datetime.now(timezone.utc).isoformat(),
    "output": {"png": output_png.name},
    "evidence": {
        "sysmap": sysmap_name,
        "methods": list(methods),
        "per_method_fS": method_summaries,
        "per_cmbk_kS": kS_summaries,
    },
}

evidence_path.parent.mkdir(parents=True, exist_ok=True)
save_evidence(evidence_doc, evidence_path, snakemake)
snakemake_log(snakemake, f"Saved: {evidence_path}")
