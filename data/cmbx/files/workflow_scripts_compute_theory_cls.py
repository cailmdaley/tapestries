"""Compute theory C_ℓ predictions using eDR1like.Theory at fiducial cosmology.

Uses Planck 2018 best-fit parameters (from eDR1like defaults) and the
pipeline's n(z) distributions. Outputs theory C_ℓ for all 6 tomographic bins,
binned through the NaMaster bandpower windows to match the data.

For shear (lensmc/metacal):
  - Cross-spectra C_ℓ^{κE} (vanilla and NLA IA) — for theory comparison
  - Auto-spectra C_ℓ^{EE} per bin and C_ℓ^{κκ} — for Gaussian covariance guess

For density:
  - Cross-spectra C_ℓ^{κδ} with SPV3 galaxy bias — for theory comparison
  - Auto-spectra C_ℓ^{δδ} per bin and C_ℓ^{κκ} — for Gaussian covariance guess
  - No IA (intrinsic alignment only affects shear)

IA parameters read from config.yaml intrinsic_alignment section.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import astropy.io.fits as pf
import numpy as np

from dr1_cmbx.eDR1like.theory import Theory
from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log


snakemake = snakemake  # type: ignore # noqa: F821

cmbk = snakemake.wildcards.cmbk
method = snakemake.wildcards.method
nz_path = Path(snakemake.input.nz_file)
output_npz = Path(snakemake.output.npz)
is_density = method == "density"

snakemake_log(snakemake, f"Computing theory C_l for {method} x {cmbk}")
ia_config = snakemake.config.get("intrinsic_alignment", {})
ia_A_IA = ia_config.get("A_IA", 1.72)
ia_eta_IA = ia_config.get("eta_IA", -0.41)
ia_model = ia_config.get("model", "NLA-z")

snakemake_log(snakemake, f"  n(z) file: {nz_path}")
if is_density:
    snakemake_log(snakemake, "  Probe: density (κδ) — SPV3 galaxy bias, no IA")
else:
    snakemake_log(snakemake, f"  Probe: shear (κE) — IA model: {ia_model} (A_IA={ia_A_IA}, eta={ia_eta_IA})")

# --- Load n(z) ---
hdul = pf.open(str(nz_path))
tab = hdul["BIN_INFO"].data
z_array = np.linspace(0, 6, 3000)

dndz = {}
for i in range(len(tab["BIN_ID"])):
    bin_id = tab["BIN_ID"][i]
    dndz[f"bin{bin_id}"] = (z_array, tab["N_Z"][i].astype(np.float64))

# "all" bin: sum of individual bin n(z) distributions
nz_all = sum(nz for _, nz in dndz.values())
dndz["binall"] = (z_array, nz_all)

snakemake_log(snakemake, f"  Loaded {len(dndz)} n(z) bins (including 'all')")

# --- Compute theory C_l ---
lmax = int(snakemake.params.lmax)

# Tracer prefix: "L" for weak lensing, "G" for galaxy clustering
tracer_prefix = "G" if is_density else "L"

# Map bin_id → tracer index (Theory assigns indices by dndz insertion order)
bin_to_tracer_idx = {str(i): i - 1 for i in range(1, 7)}
bin_to_tracer_idx["all"] = 6


def compute_theory_shear(dndz, lmax, include_ia_bias=False):
    """Compute shear theory C_l with optional IA."""
    theory = Theory(
        dndz=dndz,
        include_lensing=True,
        include_weaklensing=True,
        theory_provider="ccl",
        lmax=lmax,
        bias_type="global",
        set_bias_cst=True,
        include_ia_bias=include_ia_bias,
    )
    return theory.get_theory_cls()


def compute_theory_density(dndz, lmax):
    """Compute density theory C_l with SPV3 galaxy bias."""
    theory = Theory(
        dndz=dndz,
        include_lensing=True,
        include_weaklensing=False,
        theory_provider="ccl",
        lmax=lmax,
        bias_type="global",
        set_bias_cst=False,  # Use SPV3 polynomial galaxy bias
        bias_model="SPV3",
    )
    return theory.get_theory_cls()


def extract_and_bin(cls_theory, method, lmax, tracer_prefix):
    """Extract κ×tracer spectra and bin through bandpower windows."""
    binned = {}
    unbinned = {}

    for bin_id in ["1", "2", "3", "4", "5", "6", "all"]:
        tracer_key = ("P", f"{tracer_prefix}{bin_to_tracer_idx[bin_id]}")
        if tracer_key not in cls_theory:
            continue

        cl_full = cls_theory[tracer_key]
        unbinned[bin_id] = cl_full

        data_npz_path = Path(snakemake.input.data_dir) / f"{method}_bin{bin_id}_x_{cmbk}_cls.npz"
        if data_npz_path.exists():
            data = np.load(str(data_npz_path), allow_pickle=True)
            bpws = data["bpws"]
            ells_data = data["ells"]
            bpw_ee = bpws[0, :, 0, :]

            n_ell_bpw = bpw_ee.shape[1]
            cl_padded = np.zeros(n_ell_bpw)
            n_copy = min(len(cl_full), n_ell_bpw)
            cl_padded[:n_copy] = cl_full[:n_copy]

            cl_binned = bpw_ee @ cl_padded
            binned[bin_id] = {"ells": ells_data, "cl_binned": cl_binned}

    return unbinned, binned


if is_density:
    # --- Density probe: single theory computation (no IA) ---
    snakemake_log(snakemake, f"  Initializing Theory (CCL, lmax={lmax}, SPV3 bias)...")
    cls_theory = compute_theory_density(dndz, lmax)
    snakemake_log(snakemake, f"  Available keys: {list(cls_theory.keys())}")
    unbinned, binned = extract_and_bin(cls_theory, method, lmax, "G")

    for bin_id, d in binned.items():
        snakemake_log(
            snakemake,
            f"  bin {bin_id}: {len(d['ells'])} bands, "
            f"max(ell*Cl)={np.max(d['ells'] * d['cl_binned']):.3e}",
        )

    # Extract auto-spectra: C_ℓ^{δδ} per bin and C_ℓ^{κκ}
    snakemake_log(snakemake, "  Extracting auto-spectra for covariance...")
    cl_kk = cls_theory.get(("P", "P"), None)
    if cl_kk is not None:
        snakemake_log(snakemake, f"  C_l^{{kk}}: len={len(cl_kk)}, max(l*Cl)={np.max(np.arange(len(cl_kk)) * cl_kk):.3e}")

    cl_auto = {}
    for bin_id in ["1", "2", "3", "4", "5", "6", "all"]:
        idx = bin_to_tracer_idx[bin_id]
        tracer_key = (f"G{idx}", f"G{idx}")
        cl = cls_theory.get(tracer_key, None)
        if cl is not None:
            cl_auto[bin_id] = cl
            snakemake_log(
                snakemake,
                f"  C_l^{{dd}} bin {bin_id}: max(l*Cl)={np.max(np.arange(len(cl)) * cl):.3e}",
            )

    # --- Save density theory ---
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    ell_theory = np.arange(lmax + 1)

    save_dict = {
        "ell_theory": ell_theory,
        "metadata": {
            "cmbk": cmbk,
            "method": method,
            "probe": "density",
            "lmax": lmax,
            "cosmology": "Planck2018_best_fit",
            "theory_provider": "ccl",
            "bias_model": "SPV3",
            "generated": datetime.now(timezone.utc).isoformat(),
        },
    }

    # Store theory (single variant — no IA for density)
    for bin_id, cl in unbinned.items():
        save_dict[f"cl_theory_bin{bin_id}"] = cl
    for bin_id, data in binned.items():
        save_dict[f"ells_binned_bin{bin_id}"] = data["ells"]
        save_dict[f"cl_binned_bin{bin_id}"] = data["cl_binned"]

    # Store auto-spectra for covariance
    if cl_kk is not None:
        save_dict["cl_kk_theory"] = cl_kk
    for bin_id, cl in cl_auto.items():
        save_dict[f"cl_dd_theory_bin{bin_id}"] = cl

else:
    # --- Shear probe: vanilla + NLA IA ---
    # Vanilla theory (lensing only)
    snakemake_log(snakemake, f"  Initializing Theory (CCL, lmax={lmax}, no IA)...")
    cls_vanilla = compute_theory_shear(dndz, lmax, include_ia_bias=False)
    snakemake_log(snakemake, f"  Available keys: {list(cls_vanilla.keys())}")
    vanilla_unbinned, vanilla_binned = extract_and_bin(cls_vanilla, method, lmax, "L")

    for bin_id, d in vanilla_binned.items():
        snakemake_log(
            snakemake,
            f"  bin {bin_id} (vanilla): {len(d['ells'])} bands, "
            f"max(ell*Cl)={np.max(d['ells'] * d['cl_binned']):.3e}",
        )

    # NLA IA theory
    snakemake_log(snakemake, f"  Computing theory with include_ia_bias=True ({ia_model} requested)...")
    snakemake_log(
        snakemake,
        "  NOTE: dr1_cmbx.eDR1like.Theory currently uses internal IA defaults "
        "(A_IA=1.72, eta_IA=-0.41) and does not accept ia_params from workflow config.",
    )
    cls_ia = compute_theory_shear(dndz, lmax, include_ia_bias=True)
    ia_unbinned, ia_binned = extract_and_bin(cls_ia, method, lmax, "L")

    for bin_id, d in ia_binned.items():
        snakemake_log(
            snakemake,
            f"  bin {bin_id} (NLA IA): {len(d['ells'])} bands, "
            f"max(ell*Cl)={np.max(d['ells'] * d['cl_binned']):.3e}",
        )

    # --- Extract auto-spectra for Gaussian covariance guess ---
    snakemake_log(snakemake, "  Extracting auto-spectra for covariance...")

    # CMB lensing auto: ("P", "P")
    cl_kk = cls_ia.get(("P", "P"), None)
    if cl_kk is not None:
        snakemake_log(snakemake, f"  C_l^{{kk}}: len={len(cl_kk)}, max(l*Cl)={np.max(np.arange(len(cl_kk)) * cl_kk):.3e}")
    else:
        snakemake_log(snakemake, "  WARNING: C_l^{kk} not found in theory output")

    # Shear auto per bin: ("L_i", "L_i") — uses IA theory (includes IA contribution)
    cl_ee_auto = {}
    for bin_id in ["1", "2", "3", "4", "5", "6", "all"]:
        idx = bin_to_tracer_idx[bin_id]
        tracer_key = (f"L{idx}", f"L{idx}")
        cl_auto = cls_ia.get(tracer_key, None)
        if cl_auto is not None:
            cl_ee_auto[bin_id] = cl_auto
            snakemake_log(
                snakemake,
                f"  C_l^{{EE}} bin {bin_id}: max(l*Cl)={np.max(np.arange(len(cl_auto)) * cl_auto):.3e}",
            )

    snakemake_log(snakemake, f"  Auto-spectra: {len(cl_ee_auto)} EE bins + {'1 kk' if cl_kk is not None else '0 kk'}")

    # --- Save shear theory ---
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    ell_theory = np.arange(lmax + 1)

    save_dict = {
        "ell_theory": ell_theory,
        "metadata": {
            "cmbk": cmbk,
            "method": method,
            "probe": "shear",
            "lmax": lmax,
            "cosmology": "Planck2018_best_fit",
            "theory_provider": "ccl",
            "bias": "constant_b1",
            "ia_model": ia_model,
            "ia_config_requested_A_IA": ia_A_IA,
            "ia_config_requested_eta_IA": ia_eta_IA,
            "ia_params_applied_from_config": False,
            "generated": datetime.now(timezone.utc).isoformat(),
        },
    }

    # Store vanilla theory
    for bin_id, cl in vanilla_unbinned.items():
        save_dict[f"cl_theory_bin{bin_id}"] = cl
    for bin_id, data in vanilla_binned.items():
        save_dict[f"ells_binned_bin{bin_id}"] = data["ells"]
        save_dict[f"cl_binned_bin{bin_id}"] = data["cl_binned"]

    # Store IA theory (prefixed with cl_ia_)
    for bin_id, cl in ia_unbinned.items():
        save_dict[f"cl_ia_theory_bin{bin_id}"] = cl
    for bin_id, data in ia_binned.items():
        save_dict[f"cl_ia_binned_bin{bin_id}"] = data["cl_binned"]

    # Store auto-spectra for covariance
    if cl_kk is not None:
        save_dict["cl_kk_theory"] = cl_kk
    for bin_id, cl in cl_ee_auto.items():
        save_dict[f"cl_ee_theory_bin{bin_id}"] = cl

np.savez(output_npz, **save_dict)
snakemake_log(snakemake, f"Saved: {output_npz}")
snakemake_log(snakemake, "Done!")
