# %%
"""Compare theoretical and empirical N(z) distributions.

Simplified script for comparing two N(z) estimates:
1. Theoretical N(z): Forecasted Euclid distributions
2. Empirical N(z): Simple weighted histogram from catalog
"""

from pathlib import Path

import fitsio
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from dr1_notebooks.scratch.cdaley.plot_utils import FW, FH, setup_theme
from IPython import get_ipython

sns.set_palette("husl", 6)
PROJECT_ROOT = Path("/leonardo_work/EUHPC_E05_083/cdaley00/cmbx")


# %%
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

    import yaml

    from dr1_notebooks.scratch.cdaley.snakemake_helpers import snakemake_interactive

    with (PROJECT_ROOT / "config/config.yaml").open() as fp:
        config_preview = yaml.safe_load(fp)

    out_dir = Path(config_preview["paths"]["out_dir"])
    default_target = str(
        out_dir
        / f"{config_preview['catalog_version']}-{config_preview['pipeline_version']}"
        / "explorations"
        / "nz_comparison"
        / "nz_comparison_stats.json"
    )

    snakemake = snakemake_interactive(default_target, workdir=str(PROJECT_ROOT))
else:  # pragma: no cover - executed by Snakemake runner
    from snakemake.script import snakemake  # type: ignore


# %%
def load_theoretical_nz(file_path: Path) -> tuple[np.ndarray, dict]:
    """Load theoretical N(z) distributions from FITS file.

    Returns:
        z_grid: Redshift grid array
        nz_data: Dict with bin_id -> {'mean_z': float, 'nz': array}
    """
    print("Loading theoretical N(z) distributions...")
    data = fitsio.read(str(file_path), ext=1)

    # Use predefined redshift grid (z = 0 to 6 with 3000 bins)
    z_grid = np.linspace(0, 6, 3000)

    nz_data = {}
    for row in data:
        bin_id = int(row['BIN_ID'])
        nz_data[bin_id] = {
            'mean_z': float(row['MEAN_REDSHIFT']),
            'nz': row['N_Z'].copy()
        }

    print(f"  Loaded {len(nz_data)} theoretical N(z) bins")
    return z_grid, nz_data


def construct_empirical_nz(catalog_path: Path, phz_column: str, z_min: float = 0.0, z_max: float = 3.0, dz: float = 0.01, weighted: bool = True) -> dict:
    """Construct empirical N(z) from catalog using specified binning parameters.

    Args:
        catalog_path: Path to catalog parquet file
        phz_column: Column name for photometric redshift (e.g., 'phz_median', 'phz_mode_1')
        z_min: Minimum redshift for binning
        z_max: Maximum redshift for binning
        dz: Bin width
        weighted: If True, use phz_weight; if False, unweighted

    Returns:
        nz_data: Dict with bin_id -> {'nz': array, 'z_centers': array}
    """
    weight_type = "weighted" if weighted else "unweighted"
    print(f"Constructing {weight_type} empirical N(z) from catalog using {phz_column}...")
    print(f"  Binning: z=[{z_min:.2f}, {z_max:.2f}], dz={dz:.3f}")

    # Load catalog data
    df = pl.scan_parquet(catalog_path).select(["tom_bin_id", phz_column, "phz_weight"]).collect()
    print(f"  Using {len(df):,} catalog objects for empirical N(z)")

    # Create binning grid
    z_bin_edges = np.arange(z_min, z_max + dz/2, dz)
    z_centers = z_bin_edges[:-1] + dz/2

    nz_data = {}
    for bin_id in range(1, 7):
        bin_data = df.filter(pl.col("tom_bin_id") == bin_id)

        # Create histogram (weighted or unweighted)
        z_phot = bin_data[phz_column].to_numpy()
        if weighted:
            weights = bin_data['phz_weight'].to_numpy()
        else:
            weights = None

        # Use histogram with specified spacing
        hist, _ = np.histogram(z_phot, bins=z_bin_edges, weights=weights)

        # Normalize to unit area
        if np.sum(hist) > 0:
            hist_norm = hist / np.sum(hist) / dz
        else:
            hist_norm = hist

        # Store result with bin centers for plotting
        nz_data[bin_id] = {'nz': hist_norm, 'z_centers': z_centers}
        print(f"  Bin {bin_id}: {len(bin_data):,} objects")

    return nz_data


# %%
# Load all N(z) estimates
print("Loading N(z) data from all sources...")

z_grid, theoretical_nz = load_theoretical_nz(Path(snakemake.input["theoretical_nz"]))

# Get binning width and phz column from Snakemake parameters
catalog_path = Path(snakemake.input["catalog"])
dz = snakemake.params["binning_width"]
phz_column = snakemake.params["phz_column"]

# Create empirical N(z) with specified binning
empirical_weighted_nz = construct_empirical_nz(catalog_path, phz_column, z_min=0.0, z_max=3.0, dz=dz, weighted=True)
empirical_unweighted_nz = construct_empirical_nz(catalog_path, phz_column, z_min=0.0, z_max=3.0, dz=dz, weighted=False)

print(f"Redshift grid: {len(z_grid)} bins from z={z_grid[0]:.1f} to z={z_grid[-1]:.1f}")


# %%
output_dir = Path(snakemake.params["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)


# %%
def create_two_pane_plot(empirical_data, empirical_label, binning_suffix, phz_col_label, output_filename):
    """Create a two-pane N(z) comparison plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2*FW, 2*FH), sharex=True)

    # Get bin colors
    bin_colors = sns.color_palette("husl", 6)

    # Define normalization function
    def normalize_to_unit_area(nz, z_grid):
        """Normalize distributions to have area = 1."""
        dz = z_grid[1] - z_grid[0]
        integral = np.sum(nz) * dz
        return nz / integral if integral > 0 else np.zeros_like(nz)

    # Plot all bins
    for i, bin_id in enumerate(range(1, 7)):
        color = bin_colors[i]
        label = f'bin {bin_id}'

        # Theoretical N(z) - no smoothing
        theo_norm = normalize_to_unit_area(theoretical_nz[bin_id]['nz'], z_grid)

        # Empirical N(z) - use its own grid, no smoothing
        emp_nz = empirical_data[bin_id]['nz']  # Already normalized
        emp_z = empirical_data[bin_id]['z_centers']

        # TOP PANE: Theoretical N(z)
        ax1.fill_between(z_grid, 0, theo_norm, color=color, alpha=0.3)
        ax1.plot(z_grid, theo_norm, color=color, linewidth=2, alpha=1, label=label)

        # BOTTOM PANE: Empirical N(z) - use step-filled histogram
        ax2.fill_between(emp_z, 0, emp_nz, step='mid', color=color, alpha=0.3)
        ax2.step(emp_z, emp_nz, where='mid', color=color, linewidth=2, alpha=1.0, label=label)

    # Format all axes with appropriate y-axis limits
    # Determine appropriate y-limits from the data
    all_theo = [normalize_to_unit_area(theoretical_nz[bid]['nz'], z_grid).max() for bid in range(1, 7)]
    all_emp = [empirical_data[bid]['nz'].max() for bid in range(1, 7)]

    max_theo = max(all_theo) * 1.05  # Add 5% margin
    max_emp = max(all_emp) * 1.05

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 2.5)
        ax.grid(True, alpha=0.3)

    # Set individual y-limits based on data ranges
    ax1.set_ylim(0, max_theo)
    ax2.set_ylim(0, max_emp)

    # Labels and titles
    ax1.set_ylabel('Forecasted n(z)', fontsize=14)
    ax1.set_title(f'Shear n(z): forecasted vs empirical ({empirical_label}, {phz_col_label})', fontsize=16, pad=20)

    ax2.set_ylabel(f'Empirical n(z) ({empirical_label})', fontsize=14)
    ax2.set_xlabel('Redshift z', fontsize=14)

    # Add legends
    ax1.legend(loc='upper right', fontsize=10, ncol=2, framealpha=0.8)
    ax2.legend(loc='upper right', fontsize=10, ncol=2, framealpha=0.8)

    # Annotation for empirical panel
    ax2.text(0.02, 0.95, f'{binning_suffix}',
             transform=ax2.transAxes, ha='left', va='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / output_filename, bbox_inches='tight')
    plt.close()


# Create plots for the specified binning width
print(f"Generating weighted empirical N(z) comparison (dz={dz:.3f} binning, {phz_column})...")
create_two_pane_plot(empirical_weighted_nz, "weighted", f"dz={dz:.3f}", phz_column, Path(snakemake.output["weighted"]).name)

print(f"Generating unweighted empirical N(z) comparison (dz={dz:.3f} binning, {phz_column})...")
create_two_pane_plot(empirical_unweighted_nz, "unweighted", f"dz={dz:.3f}", phz_column, Path(snakemake.output["unweighted"]).name)


print(f"N(z) comparison analysis complete. Results saved to {output_dir}")