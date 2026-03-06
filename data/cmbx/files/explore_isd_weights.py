"""Generate ISD weight iterations GIFs and product of weights plots."""

import fitsio
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import io
import seaborn as sns
from IPython import get_ipython

from dr1_notebooks.scratch.cdaley.rr2 import RR2Coverage

sns.set_palette("husl", 6)

ipython = get_ipython()
if ipython is not None:
    import yaml
    from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_interactive
    PROJECT_ROOT = Path("/leonardo_work/EUHPC_E05_083/cdaley00/cmbx")
    with (PROJECT_ROOT / "config/config.yaml").open() as fp:
        config_preview = yaml.safe_load(fp)
    out_dir = Path(config_preview["paths"]["out_dir"])
    default_target = str(out_dir / f"{config_preview['catalog_version']}-{config_preview['pipeline_version']}" / "explorations" / "isd_analysis" / "isd_product_bin_1.png")
    snakemake = snakemake_interactive(default_target, workdir=str(PROJECT_ROOT))
else:
    from snakemake.script import snakemake


def cubehelix(start):
    return sns.cubehelix_palette(start=start, rot=0.5, as_cmap=True)


def load_isd_coverage(bin_id):
    """Load ISD coverage object using ISD footprint pixel mapping."""
    catalog_dir = Path(snakemake.input["catalog"]).parent
    footprint_file = catalog_dir / "correction_weights_footprint_pixels_1_bias2_PHZPTBIDCUT.npy"
    footprint_pixels = np.load(footprint_file)

    # Load ISD weights
    weights_file = next(f for f in snakemake.input["isd_weights"] if f"_{bin_id}_" in f)
    weights = np.load(weights_file)

    # Handle 1-pixel mismatch: bins 2-6 have 1 extra pixel
    # For bins 2-6, append the missing visibility pixel to footprint
    nside = 4096
    if weights.shape[1] > len(footprint_pixels):
        # Load visibility to find the extra pixel
        vis_path = catalog_dir / f"visibility_tombinid_{bin_id}.fits"
        vis_data = fitsio.read(vis_path)
        vis_pixels = vis_data['PIXEL']
        # Find pixel in visibility but not in footprint
        extra_pixel = list(set(vis_pixels) - set(footprint_pixels))[0]
        pixels = np.append(footprint_pixels, extra_pixel)
    else:
        pixels = footprint_pixels

    # Convert to RA/DEC and create coverage object
    ra, dec = hp.pix2ang(nside, pixels, lonlat=True, nest=True)
    return RR2Coverage(nside=nside, ra=ra, dec=dec, w=np.ones(len(ra)), sparse=True, nest=True), weights


def isd_iteration_frames(coverage_obj, weights, bin_id):
    """Generator yielding PIL Images for each ISD iteration (inverse weights)."""
    for i in range(weights.shape[0]):
        fig, axes = coverage_obj.plot(
            1.0 / weights[i],
            cmap=cubehelix(2.0),
            vlim="asym",
            vlim_percentiles=(5, 95),
            vscale="linear"
        )
        fig.suptitle(rf"ISD weight$^{{-1}}$ - Bin {bin_id}, Iteration {i+1}/30", fontsize=16, y=0.85)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        yield Image.open(buf)


def create_gif(frames, output_path, duration=375):
    """Create animated GIF from frame list."""
    frames = list(frames)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)

# Generate outputs
output_dir = Path(snakemake.params["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

for bin_id in range(1, 7):
    print(f"\n=== Processing tomographic bin {bin_id} ===")

    coverage_obj, weights = load_isd_coverage(bin_id)

    # Compute product of all weight iterations
    weight_product = np.prod(weights, axis=0)

    # Create systematic indicator plot (inverse of weight product)
    fig_product, ax_product = coverage_obj.plot(
        1.0 / weight_product,
        cmap=cubehelix(2.0),
        vlim="asym",
        vlim_percentiles=(5, 95),
        vscale="linear"
    )
    fig_product.suptitle(rf"ISD $(\Pi {{\rm weights}})^{{-1}}$ - Bin {bin_id}", fontsize=14, y=0.85)
    fig_product.savefig(output_dir / f"isd_product_bin_{bin_id}.png", bbox_inches="tight")
    plt.close(fig_product)

    # Create GIF showing evolution across iterations
    frames = isd_iteration_frames(coverage_obj, weights, bin_id)
    gif_path = output_dir / f"isd_weights_iterations_bin_{bin_id}.gif"
    create_gif(frames, gif_path)

    print(f"Generated: {gif_path.name} and isd_product_bin_{bin_id}.png")
