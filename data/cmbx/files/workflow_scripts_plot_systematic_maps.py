"""VMPZ-ID systematic maps on the RR2 footprint — one figure per map.

Each systematic map gets its own 1×2 figure (north/south patches).
Produces 11 individual PNGs (~1MB each) instead of one 13MB composite.

This is the root node of the contamination analysis: every map shown here
is a potential contaminant. The reader sees the spatial structure before
any null test or metric is computed.
"""

import pickle
from datetime import datetime, timezone
from pathlib import Path

import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skyproj
from matplotlib.colors import Normalize

from dr1_notebooks.scratch.cdaley.rr2 import RR2Coverage
from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_log
from dr1_notebooks.scratch.cdaley.nmt_utils import load_systematic_map
from dr1_notebooks.scratch.cdaley.plot_utils import setup_theme, save_evidence


snakemake = snakemake  # type: ignore # noqa: F821

output_dir = Path(snakemake.output.evidence).parent
evidence_path = Path(snakemake.output.evidence)

nside = int(snakemake.config["nside"])
sys_map_keys = list(snakemake.params.sys_map_keys)

# Map display labels and colormaps
MAP_DISPLAY = {
    "stellar_density":      {"label": r"Stellar density (stars/deg$^2$)",  "cmap": "rocket"},
    "galactic_extinction":  {"label": r"Galactic extinction $E(B-V)$",     "cmap": "mako"},
    "exposures":            {"label": "Exposures",                          "cmap": "mako"},
    "zodiacal_light":       {"label": "Zodiacal light",                     "cmap": "rocket"},
    "saa":                  {"label": "SAA passages",                       "cmap": "rocket"},
    "psf":                  {"label": "PSF",                                "cmap": "mako"},
    "noise":                {"label": "Noise",                              "cmap": "rocket"},
    "persistence":          {"label": "Persistence",                        "cmap": "rocket"},
    "star_brightness_mean": {"label": "Star brightness (mean)",             "cmap": "mako"},
    "galaxy_number_density":{"label": "Galaxy number density",              "cmap": "mako"},
    "phz_quality":          {"label": r"Photo-$z$ quality (SEVENTY mean)",  "cmap": "mako"},
}

snakemake_log(snakemake, f"Building {len(sys_map_keys)} individual systematic map figures")

# --- Load shear weights for footprint ---
snakemake_log(snakemake, "Loading shear weight map...")
with open(snakemake.input.shear_pkl, "rb") as f:
    shear_data = pickle.load(f)
bin_key = next(k for k in shear_data if k.endswith("bin"))
bd = shear_data[bin_key][f"bin{snakemake.params.all_bin_idx}"]
npix = hp.nside2npix(nside)
shear_weights = np.zeros(npix)
shear_weights[bd["ipix"]] = bd["sum_weights"]
shear_mask = (shear_weights > 0).astype(float)
coverage = RR2Coverage.from_mask(nside, shear_mask, sparse=True)

ordered_maps = list(zip(sys_map_keys, map(str, snakemake.input.sys_maps)))

proj_params = [
    {"lon_0": 70, "lat_0": -33, "extent": [63.8, 77.2, -37, -26.5]},   # north
    {"lon_0": 45, "lat_0": -64, "extent": [32, 70, -44, -63.5]},         # south
]

LABEL_SIZE = 12
TICK_SIZE = 9
CBAR_SIZE = 9

map_stats = {}
output_pngs = {}

for row_idx, (key, path) in enumerate(ordered_maps):
    snakemake_log(snakemake, f"  Loading {key} ({Path(path).name})...")
    try:
        m = load_systematic_map(path, nside, log_fn=lambda msg: snakemake_log(snakemake, msg))
        m_masked = m * shear_mask
        in_fp = m_masked[shear_mask > 0]
        in_fp_nonzero = in_fp[in_fp != 0]

        if len(in_fp_nonzero) > 0:
            map_stats[key] = {
                "median": float(np.median(in_fp_nonzero)),
                "p10":    float(np.percentile(in_fp_nonzero, 10)),
                "p90":    float(np.percentile(in_fp_nonzero, 90)),
                "frac_nonzero": float(len(in_fp_nonzero) / len(in_fp)),
            }
            snakemake_log(snakemake, f"    median={map_stats[key]['median']:.4g}, "
                          f"p90={map_stats[key]['p90']:.4g}, "
                          f"frac_nonzero={map_stats[key]['frac_nonzero']:.3f}")
        else:
            map_stats[key] = {"median": 0, "p10": 0, "p90": 0, "frac_nonzero": 0}
            snakemake_log(snakemake, "    all zero in footprint")

        sparse_data = coverage.sparsify(m_masked)
        del m, m_masked, in_fp, in_fp_nonzero
    except Exception as e:
        snakemake_log(snakemake, f"  WARNING: failed to load {key}: {e}")
        sparse_data = np.zeros(len(coverage.ipix_sparse))
        map_stats[key] = {"error": str(e)}

    finite = sparse_data[np.isfinite(sparse_data) & (sparse_data != 0)]
    if len(finite) > 0:
        vmin = float(np.percentile(finite, 2))
        vmax = float(np.percentile(finite, 98))
    else:
        vmin, vmax = 0, 1
    norm = Normalize(vmin, vmax)

    disp = MAP_DISPLAY.get(key, {"label": key.replace("_", " ").title(), "cmap": "mako"})

    # --- One figure per map: 1 row × 2 columns ---
    setup_theme("whitegrid")
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 4.5),
        gridspec_kw={"width_ratios": [46.5, 53.5], "wspace": 0.05},
    )

    for patch_idx, pp in enumerate(proj_params):
        ax = axes[patch_idx]
        sp = skyproj.GnomonicSkyproj(ax, **pp)
        sp.draw_hpxpix(
            nside, coverage.ipix_sparse, sparse_data,
            nest=coverage.nest, zoom=False,
            cmap=disp["cmap"], norm=norm,
        )
        ax.tick_params(labelsize=TICK_SIZE)
        for spine_label in ax.get_xticklabels() + ax.get_yticklabels():
            spine_label.set_fontsize(TICK_SIZE)
        ax.xaxis.label.set_size(LABEL_SIZE)
        ax.yaxis.label.set_size(LABEL_SIZE)
        if patch_idx == 1:
            result = sp.draw_inset_colorbar(loc="upper right")
            cb = result[0] if isinstance(result, tuple) else result
            if cb is not None:
                cb.ax.tick_params(labelsize=CBAR_SIZE)
                for lbl in cb.ax.get_yticklabels():
                    lbl.set_fontsize(CBAR_SIZE)

    fig.suptitle(disp["label"], fontsize=LABEL_SIZE + 2, fontweight="bold", y=1.02)

    png_name = f"systematic_map_{key}.png"
    png_path = output_dir / png_name
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    output_pngs[key] = png_name
    snakemake_log(snakemake, f"  Saved: {png_path}")
    del sparse_data

# --- Evidence ---
output_dict = {f"systematic_map_{k}": v for k, v in output_pngs.items()}
output_dict["evidence"] = "evidence.json"

evidence_doc = {
    "id": "plot_systematic_maps",
    "generated": datetime.now(timezone.utc).isoformat(),
    "input": {
        "sys_maps": list(snakemake.input.sys_maps),
        "shear_pkl": str(snakemake.input.shear_pkl),
    },
    "output": output_dict,
    "params": {
        "nside": nside,
        "n_maps": len(sys_map_keys),
        "map_keys": sys_map_keys,
    },
    "evidence": {
        "maps": {k: map_stats[k] for k in sys_map_keys},
    },
}

evidence_path.parent.mkdir(parents=True, exist_ok=True)
save_evidence(evidence_doc, evidence_path, snakemake)
snakemake_log(snakemake, f"Saved evidence: {evidence_path}")
