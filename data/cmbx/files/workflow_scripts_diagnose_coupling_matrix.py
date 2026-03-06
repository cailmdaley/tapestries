"""Diagnose NaMaster coupling matrix conditioning.

Investigates whether mode-coupling matrix inversion is numerically stable
for our mask geometry, particularly for spin-2 fields at low ℓ. Motivated
by Giulio's flag (2026-02-18) and the suspicious SPT ℓ≈100 spike.

For each CMB experiment (ACT, SPT GMV), creates workspaces with the same
Euclid shear mask but different spins:
  - spin-0 × spin-0 (scalar × CMBκ) — control
  - spin-2 × spin-0 (shear × CMBκ) — test

Same mask geometry isolates the effect of spin on conditioning.

Diagnostics:
  - Coupling matrix condition numbers
  - Singular value spectrum
  - Roundtrip test: couple → decouple, check amplification per bandpower
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pickle

import healpy as hp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_cmbx.eDR1data.spectra import make_bins, make_mask, make_field
from dr1_notebooks.scratch.cdaley.nmt_utils import read_healpix_map

import pymaster as nmt

# --- Config ---
nside = snakemake.params.nside
lmin = snakemake.params.lmin
lmax = snakemake.params.lmax
nells = snakemake.params.nells
aposcale = snakemake.params.aposcale
n_iter = snakemake.params.n_iter

output_npz = Path(snakemake.output.npz)
output_png = Path(snakemake.output.png)
output_evidence = Path(snakemake.output.evidence)

# --- Load shear weight map for Euclid mask ---
snakemake_log(snakemake, "Loading shear weight map...")
with open(snakemake.input.shear_pkl, "rb") as f:
    shear_data = pickle.load(f)
bin_key = next(k for k in shear_data if k.endswith("bin"))
bd = shear_data[bin_key][f"bin{snakemake.params.all_bin_idx}"]
npix = hp.nside2npix(nside)
shear_weight = np.zeros(npix)
shear_weight[bd["ipix"]] = bd["sum_weights"]
euclid_mask = make_mask(shear_weight, aposcale=aposcale, weights=[shear_weight])
fsky_euclid = float(np.mean(euclid_mask > 0))
snakemake_log(snakemake, f"  Euclid mask fsky: {fsky_euclid:.4f}")

# --- Binning ---
bins, bpw_edges = make_bins(lmin, lmax, nells)
nbins = bins.get_n_bands()
ell_eff = bins.get_effective_ells()
snakemake_log(snakemake, f"Binning: {nbins} bins, ell=[{bpw_edges[0]}, {bpw_edges[-1]}]")

# --- Create zero-signal Euclid fields (mask geometry only) ---
npix = 12 * nside**2
zeros = np.zeros(npix)

scalar_field = make_field(zeros, euclid_mask, spin=0, lmax=lmax, n_iter=n_iter)
shear_field = make_field([zeros, zeros], euclid_mask, spin=2, pol_conv="EUCLID", lmax=lmax, n_iter=n_iter)

# --- Analyze each CMB experiment ---
results = {}
cmbk_names = snakemake.params.cmbk_names

for cmbk_name, cmbk_mask_path in zip(cmbk_names, snakemake.input.cmbk_masks):
    snakemake_log(snakemake, f"\n{'='*60}")
    snakemake_log(snakemake, f"  {cmbk_name}")
    snakemake_log(snakemake, f"{'='*60}")

    cmbk_mask_raw = read_healpix_map(str(cmbk_mask_path))
    cmbk_mask = nmt.mask_apodization(cmbk_mask_raw, aposcale, "C2")
    fsky_cmb = float(np.mean(cmbk_mask > 0))
    snakemake_log(snakemake, f"  CMB mask fsky: {fsky_cmb:.4f}")

    cmbk_field = make_field(zeros, cmbk_mask, spin=0, lmax=lmax, n_iter=n_iter)
    cmbk_results = {}

    for field_name, euclid_field, spin_label in [
        ("spin0", scalar_field, "spin-0 × spin-0"),
        ("spin2", shear_field, "spin-2 × spin-0"),
    ]:
        snakemake_log(snakemake, f"\n  --- {spin_label} ---")

        # Create workspace (no cache — we need fresh computation)
        wksp = nmt.NmtWorkspace()
        wksp.compute_coupling_matrix(euclid_field, cmbk_field, bins)

        # Extract coupling matrix
        mcm = wksp.get_coupling_matrix()
        snakemake_log(snakemake, f"  Coupling matrix shape: {mcm.shape}")

        # Condition number (2-norm)
        try:
            cond = float(np.linalg.cond(mcm))
        except np.linalg.LinAlgError:
            cond = float("inf")
        snakemake_log(snakemake, f"  Condition number: {cond:.2e}")

        # Top-k singular values (full SVD too expensive for 6002×6002)
        # Use randomized SVD for the top/bottom singular values
        n_total = mcm.shape[0]
        snakemake_log(snakemake, f"  Computing SVD ({n_total}×{n_total})...")
        try:
            sv = np.linalg.svd(mcm, compute_uv=False)
            sv_max = float(sv[0])
            sv_min = float(sv[-1])
            # Keep every 10th for storage
            sv_sparse = sv[::max(1, len(sv) // 200)].copy()
            snakemake_log(snakemake, f"  SV max: {sv_max:.4e}, min: {sv_min:.4e}")
            snakemake_log(snakemake, f"  SV[10]: {float(sv[min(9, len(sv)-1)]):.4e}")
        except np.linalg.LinAlgError:
            sv_sparse = np.array([])
            sv_max = sv_min = float("nan")
            snakemake_log(snakemake, "  SVD failed!")

        # Roundtrip amplification test
        n_cls_field = 2 if field_name == "spin2" else 1
        amplification = np.zeros(nbins)
        for b in range(nbins):
            test_cl = np.zeros((n_cls_field, lmax + 1))
            for ell in bins.get_ell_list(b):
                if ell <= lmax:
                    test_cl[0, ell] = 1.0
            coupled = wksp.couple_cell(test_cl)
            decoupled = wksp.decouple_cell(coupled)
            input_binned = bins.bin_cell(test_cl)
            if input_binned[0, b] > 0:
                amplification[b] = decoupled[0, b] / input_binned[0, b]
            else:
                amplification[b] = float("nan")

        snakemake_log(snakemake, f"  Amplification (first 5): {amplification[:5].round(6)}")
        snakemake_log(snakemake, f"  Amplification range: [{np.nanmin(amplification):.6f}, {np.nanmax(amplification):.6f}]")

        # Bandpower windows
        bpws = wksp.get_bandpower_windows()
        snakemake_log(snakemake, f"  Bandpower windows shape: {bpws.shape}")

        cmbk_results[field_name] = {
            "condition_number": cond,
            "sv_max": sv_max,
            "sv_min": sv_min,
            "sv_sparse": sv_sparse.tolist(),
            "amplification": amplification.tolist(),
        }

        del wksp, mcm

    results[cmbk_name] = cmbk_results

# --- Save NPZ ---
output_npz.parent.mkdir(parents=True, exist_ok=True)
save_dict = {"ell_eff": ell_eff, "bpw_edges": bpw_edges}
for cmbk in results:
    for ft in results[cmbk]:
        save_dict[f"{cmbk}_{ft}_amplification"] = np.array(results[cmbk][ft]["amplification"])
        save_dict[f"{cmbk}_{ft}_sv"] = np.array(results[cmbk][ft]["sv_sparse"])
np.savez(output_npz, **save_dict)
snakemake_log(snakemake, f"\nSaved: {output_npz}")

# --- Plot ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from dr1_notebooks.scratch.cdaley.plot_utils import FW, FH, setup_theme

sns.set_theme(style="whitegrid", font_scale=1.1)
fig, axes = plt.subplots(2, 2, figsize=(2*FW, 2*FH))

field_labels = {
    "spin0": r"spin-0 $\times$ spin-0",
    "spin2": r"spin-2 $\times$ spin-0",
}
colors = {"spin0": "#4878d0", "spin2": "#ee854a"}

# Top row: singular value spectra
for col, cmbk in enumerate(cmbk_names[:2]):
    ax = axes[0, col]
    for ft in ["spin0", "spin2"]:
        sv = np.array(results[cmbk][ft]["sv_sparse"])
        if len(sv) > 0 and sv[0] > 0:
            ax.semilogy(sv / sv[0], label=field_labels[ft], color=colors[ft], alpha=0.8)
    ax.set_xlabel("Index (subsampled)")
    ax.set_ylabel("Normalized singular value")
    cond_s0 = results[cmbk]["spin0"]["condition_number"]
    cond_s2 = results[cmbk]["spin2"]["condition_number"]
    cond_s0_str = f"{cond_s0:.1e}" if np.isfinite(cond_s0) else "singular"
    cond_s2_str = f"{cond_s2:.1e}" if np.isfinite(cond_s2) else "singular"
    ax.set_title(f"{cmbk.replace('_', ' ').upper()}\ncond(s0)={cond_s0_str}  cond(s2)={cond_s2_str}")
    ax.legend(fontsize=9)

# Bottom row: roundtrip amplification
for col, cmbk in enumerate(cmbk_names[:2]):
    ax = axes[1, col]
    for ft in ["spin0", "spin2"]:
        amp = np.array(results[cmbk][ft]["amplification"])
        ax.plot(ell_eff, amp, "o-", label=field_labels[ft], color=colors[ft],
                alpha=0.8, markersize=4)
    ax.axhline(1.0, color="k", ls="--", alpha=0.3)
    ax.set_xlabel(r"$\ell_{\rm eff}$")
    ax.set_ylabel("Roundtrip amplification")
    ax.set_title(f"{cmbk.replace('_', ' ').upper()} — couple → decouple")
    ax.legend(fontsize=9)
    # Zoom to show deviations from 1
    amp_all = np.concatenate([np.array(results[cmbk][ft]["amplification"]) for ft in ["spin0", "spin2"]])
    amp_range = max(abs(np.nanmax(amp_all) - 1), abs(1 - np.nanmin(amp_all)), 0.01)
    ax.set_ylim(1 - 2 * amp_range, 1 + 2 * amp_range)

fig.suptitle("NaMaster coupling matrix conditioning\n(same Euclid mask, different spins)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(output_png, bbox_inches="tight")
snakemake_log(snakemake, f"Saved: {output_png}")

# --- Evidence ---
evidence = {
    "id": "diagnose_coupling_matrix",
    "generated": datetime.now(timezone.utc).isoformat(),
    "output": {"npz": output_npz.name, "png": output_png.name},
    "params": {
        "nside": nside, "lmin": lmin, "lmax": lmax,
        "nells": nells, "aposcale": aposcale,
        "fsky_euclid": fsky_euclid,
    },
    "evidence": {},
}
for cmbk in results:
    for ft in results[cmbk]:
        r = results[cmbk][ft]
        evidence["evidence"][f"{cmbk}_{ft}"] = {
            "condition_number": r["condition_number"],
            "sv_max": r["sv_max"],
            "sv_min": r["sv_min"],
            "amplification_min": float(np.nanmin(r["amplification"])),
            "amplification_max": float(np.nanmax(r["amplification"])),
            "amplification_first_bin": r["amplification"][0] if r["amplification"] else None,
        }

output_evidence.parent.mkdir(parents=True, exist_ok=True)
with open(output_evidence, "w") as f:
    json.dump(evidence, f, indent=2)
snakemake_log(snakemake, f"Saved: {output_evidence}")
