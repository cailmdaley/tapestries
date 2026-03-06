"""Notebook-extracted plotting utilities for unblinding ceremony figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from getdist import plots
from matplotlib import scale as mscale
from scipy.interpolate import interp1d

from plotting_utils import PAPER_MPLSTYLE, SquareRootScale

# Extracted plotting environment from Sasha notebooks/scripts.
mscale.register_scale(SquareRootScale)
plt.style.use(PAPER_MPLSTYLE)
plt.rc("text", usetex=True)


@dataclass(frozen=True)
class ChainSpec:
    root: str
    label: str
    color: str
    base_dir: Path
    alpha: float = 1.0


def _load_xi_table(path: Path | str, nrows: int = 20) -> np.ndarray:
    rows: list[np.ndarray] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = np.fromstring(stripped, sep=" ")
            if values.size < 9:
                continue
            rows.append(values)
            if len(rows) >= nrows:
                break
    if len(rows) < nrows:
        raise ValueError(f"Expected at least {nrows} xi rows in {path}, found {len(rows)}")
    return np.vstack(rows)


def ensure_getdist_chain(base_dir: Path, root: str) -> Path:
    """MAKE PARAMNAMES FILE + READ CHAIN conversion (from notebooks)."""
    chain_dir = base_dir / root
    samples_path = chain_dir / f"samples_{root}.txt"
    gd_samples_path = chain_dir / f"getdist_{root}.txt"
    paramnames_path = chain_dir / f"getdist_{root}.paramnames"

    if not samples_path.exists():
        return gd_samples_path

    with samples_path.open("r", encoding="utf-8") as file:
        params = file.readline()[1:].split("\t")[:-4]

    with paramnames_path.open("w", encoding="utf-8") as file:
        for param in params:
            if len(param.split("--")) > 1:
                file.write(param.split("--")[1] + "\n")
            else:
                file.write(param.split("--")[0] + "\n")

    samples = np.loadtxt(samples_path)
    if "nautilus" in root:
        samples = np.column_stack((np.exp(samples[:, -3]), samples[:, -1] - samples[:, -2], samples[:, 0:-3]))
    else:
        samples = np.column_stack((samples[:, -1], samples[:, -3], samples[:, 0:-4]))
    np.savetxt(gd_samples_path, samples)
    return gd_samples_path


def _build_plotter(
    width_inch: float,
    axes_fontsize: float,
    axes_labelsize: float,
    legend_fontsize: float,
):
    g = plots.get_subplot_plotter(width_inch=width_inch)
    g.settings.axes_fontsize = axes_fontsize
    g.settings.axes_labelsize = axes_labelsize
    g.settings.alpha_filled_add = 0.7
    g.settings.legend_fontsize = legend_fontsize
    return g


def _set_param_labels(chain) -> None:
    # Extracted from plot_contours_blind / plot_contours_cl_vs_xi / plot_whisker.
    name_list = ["OMEGA_M", "ombh2", "h0", "n_s", "SIGMA_8", "S_8", "s_8_input", "logt_agn", "a", "m1", "bias_1"]
    label_list = [
        r"\Omega_{\rm m}",
        r"\omega_{\rm b} h^2",
        r"h_0",
        r"n_{\rm s}",
        r"\sigma_8",
        r"S_8",
        r"S_8",
        r"\log T_{\rm AGN}",
        r"A_{\rm IA}",
        r"m_1",
        r"\Delta z_1",
    ]

    param_names = chain.getParamNames()
    for name, label in zip(name_list, label_list):
        try:
            param_names.parWithName(name).label = label
        except Exception:
            pass

    # UNIONS chains expose s_8_input; contour notebooks use S_8 in some panels.
    try:
        param_names.parWithName("S_8")
    except Exception:
        try:
            s8_input = chain.getParams().s_8_input
            chain.addDerived(s8_input, name="S_8", label=r"S_8")
        except Exception:
            pass


def _adjust_paramname_chain(chain, current_name: str, target_name: str, label: str) -> None:
    # Extracted from plot_whisker.
    try:
        param_names = chain.getParamNames()
        par = param_names.parWithName(current_name)
        par.label = label
        par.name = target_name
        chain.setParamNames(param_names)
    except Exception:
        pass


def _derive_parameter_s8(chain):
    # Extracted from plot_whisker.
    if "S_8" in chain.getParamNames().list():
        return chain
    omega_m = chain.getParams().OMEGA_M
    sigma_8 = chain.getParams().SIGMA_8
    s_8 = sigma_8 * (omega_m / 0.3) ** 0.5
    chain.addDerived(s_8, name="S_8", label=r"S_8")
    return chain


def _harmonize_external_chain(chain, root: str) -> None:
    if root in {"Planck18", "KiDS-1000", "HSC_Y3", "HSC_Y3_cell", "DES+KiDS", "DES_Y3"}:
        _adjust_paramname_chain(chain, "omega_m", "OMEGA_M", r"\Omega_{\rm m}")
    if root == "DES_Y3":
        _derive_parameter_s8(chain)


def load_getdist_chains(
    chain_specs: list[ChainSpec],
    width_inch: float,
    axes_fontsize: float,
    axes_labelsize: float,
    legend_fontsize: float,
):
    g = _build_plotter(width_inch=width_inch, axes_fontsize=axes_fontsize, axes_labelsize=axes_labelsize, legend_fontsize=legend_fontsize)

    chains = []
    for spec in chain_specs:
        ensure_getdist_chain(spec.base_dir, spec.root)
        chain = g.samples_for_root(
            str(spec.base_dir / spec.root / f"getdist_{spec.root}"),
            cache=False,
            settings={"ignore_rows": 0, "smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5},
        )
        _set_param_labels(chain)
        if spec.base_dir.name == "ext_data" or spec.root in {"Planck18", "DES_Y3", "KiDS-1000", "DES+KiDS", "HSC_Y3", "HSC_Y3_cell"}:
            _harmonize_external_chain(chain, spec.root)
        chains.append(chain)

    return g, chains


def plot_triangle(
    chain_specs: list[ChainSpec],
    param_names: list[str],
    output_path: Path,
    width_inch: float = 20.0,
    axes_fontsize: float = 35.0,
    axes_labelsize: float = 50.0,
    legend_fontsize: float = 40.0,
    legend_loc: str = "upper right",
) -> None:
    """Extracted triangle_plot pattern from contour notebooks."""
    g, chains = load_getdist_chains(chain_specs, width_inch, axes_fontsize, axes_labelsize, legend_fontsize)

    colours = [spec.color for spec in chain_specs]
    linestyle = ["solid" for _ in chain_specs]
    line_args = [dict(color=col, ls=ls) for col, ls in zip(colours, linestyle)]

    g.triangle_plot(
        chains,
        param_names,
        legend_labels=[spec.label for spec in chain_specs],
        line_args=line_args,
        contour_colors=colours,
        legend_loc=legend_loc,
        filled=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.export(str(output_path))
    plt.close(g.fig)


def plot_xipm_data_vector(xipm_path: str, output_path: Path) -> None:
    """Extracted from 2D_cosmic_shear_paper_plots/corr_func.ipynb (xi+ / xi- blocks)."""
    xipm = _load_xi_table(xipm_path, nrows=20)
    theta = xipm[:, 1]
    xip = xipm[:, 3]
    xim = xipm[:, 4]
    varxip = xipm[:, 7]
    varxim = xipm[:, 8]

    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 4.5))

    ax1.tick_params(axis="both", which="both", direction="in", length=6, width=1, top=True, bottom=True, left=True, right=True)
    ax1.yaxis.minorticks_on()
    ax1.plot(theta, xip * 1e4, marker="o", markersize=4, ls="solid", lw=1.8, color="royalblue")
    ax1.fill_between(theta, (xip - varxip) * 1e4, (xip + varxip) * 1e4, color="powderblue", alpha=0.7)
    ax1.text(0.85, 0.88, "1-1", transform=ax1.transAxes, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.5))
    ax1.axvspan(0, 10, color="gray", alpha=0.3)
    ax1.axvspan(150, 200, color="gray", alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\theta$ [arcmin]")
    ax1.set_ylabel(r"$\xi_+\times 10^4$")

    ax2.tick_params(axis="both", which="both", direction="in", length=6, width=1, top=True, bottom=True, left=True, right=True)
    ax2.yaxis.minorticks_on()
    ax2.plot(theta, xim * 1e4, marker="o", markersize=4, ls="solid", lw=1.8, color="orangered")
    ax2.fill_between(theta, (xim - varxim) * 1e4, (xim + varxim) * 1e4, color="pink", alpha=0.7)
    ax2.text(0.85, 0.88, "1-1", transform=ax2.transAxes, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.5))
    ax2.axvspan(0, 10, color="gray", alpha=0.3)
    ax2.axvspan(150, 200, color="gray", alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\theta$ [arcmin]")
    ax2.set_ylabel(r"$\xi_-\times 10^4$")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cell_ee_data_vector(pseudo_cl_path: str, pseudo_cl_cov_path: str, output_path: Path) -> None:
    """Extracted from 2025_10_08_plot_data_vectors.py (EE panel logic)."""
    cell = fits.getdata(pseudo_cl_path)
    cov_cell = fits.open(pseudo_cl_cov_path)

    ell = cell["ell"]
    cl_ee = cell["EE"]
    cov_cl_ee = cov_cell["COVAR_EE_EE"].data
    cov_cell.close()

    fig, ax0 = plt.subplots(ncols=1, nrows=1, figsize=(7, 5))
    ax0.errorbar(
        ell,
        cl_ee * ell,
        yerr=np.sqrt(np.diag(cov_cl_ee)) * ell,
        label=r"$C_\ell^{EE}$",
        color="royalblue",
        fmt="o",
        capsize=2,
    )

    ax0.set_xscale("squareroot")
    ax0.set_xticks(np.array([100, 400, 900, 1600]))
    ax0.minorticks_on()
    ax0.tick_params(axis="x", which="minor", length=2, width=0.8)
    minor_ticks = [i * 10 for i in range(1, 10)] + [i * 100 for i in range(1, 21)]
    ax0.set_xticks(minor_ticks, minor=True)
    ax0.legend()

    ax0.set_xlabel(r"$\ell$")
    ax0.set_ylabel(r"$\ell \, C_\ell^{EE}$")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_xipm_bestfit_with_bmodes(
    xi_data_path: Path | str,
    pure_eb_data_path: Path,
    bestfit_dir: Path | None,
    output_path: Path,
    scale_cut_xip: tuple[float, float] = (12.0, 83.0),
    scale_cut_xim: tuple[float, float] = (12.0, 83.0),
) -> None:
    """Extracted from 2D_cosmic_shear_paper_plots/workflow/scripts/plot_xi_bestfit.py.

    If bestfit_dir is None, plots data + B-modes only (no theory curves).

    xi_data_path can be:
    - A CosmoSIS FITS file with XI_PLUS/XI_MINUS HDUs and COVMAT (preferred)
    - A plain-text TreeCorr output table (legacy)
    """
    xi_path = Path(xi_data_path)
    if xi_path.suffix == ".fits":
        xip_hdu = fits.getdata(str(xi_path), "XI_PLUS")
        xim_hdu = fits.getdata(str(xi_path), "XI_MINUS")
        cov = fits.getdata(str(xi_path), "COVMAT")
        theta_data = xip_hdu["ANG"]
        xip_data = xip_hdu["VALUE"]
        xim_data = xim_hdu["VALUE"]
        n = len(xip_data)
        sigma_xip = np.sqrt(np.diag(cov[:n, :n]))
        sigma_xim = np.sqrt(np.diag(cov[n : 2 * n, n : 2 * n]))
    else:
        data = _load_xi_table(xi_path, nrows=20)
        theta_data = data[:, 1]
        xip_data = data[:, 3]
        xim_data = data[:, 4]
        sigma_xip = data[:, 7]
        sigma_xim = data[:, 8]

    eb_data = np.load(pure_eb_data_path)
    theta_eb = eb_data["theta"]
    xip_B = eb_data["xip_B"]
    xim_B = eb_data["xim_B"]
    cov_pure_eb = eb_data["cov_pure_eb"]

    nbins = len(theta_eb)
    sigma_xip_B = np.sqrt(np.diag(cov_pure_eb[2 * nbins : 3 * nbins, 2 * nbins : 3 * nbins]))
    sigma_xim_B = np.sqrt(np.diag(cov_pure_eb[3 * nbins : 4 * nbins, 3 * nbins : 4 * nbins]))

    min_sep, max_sep = 1.0, 250.0
    bin_edges = np.geomspace(min_sep, max_sep, nbins + 1)
    bin_centers_nominal = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    def get_bin_edge_cuts(centers, edges, scale_cut):
        mask = (centers >= scale_cut[0]) & (centers <= scale_cut[1])
        idx_first = np.where(mask)[0][0]
        idx_last = np.where(mask)[0][-1]
        return edges[idx_first], edges[idx_last + 1]

    edge_cut_xip = get_bin_edge_cuts(bin_centers_nominal, bin_edges, scale_cut_xip)
    edge_cut_xim = get_bin_edge_cuts(bin_centers_nominal, bin_edges, scale_cut_xim)

    has_theory = bestfit_dir is not None
    if has_theory:
        theta_theory_rad = np.loadtxt(bestfit_dir / "shear_xi_plus" / "theta.txt", comments="#")
        theta_theory = np.rad2deg(theta_theory_rad) * 60
        xip_theory = np.loadtxt(bestfit_dir / "shear_xi_plus" / "bin_1_1.txt", comments="#")
        xim_theory = np.loadtxt(bestfit_dir / "shear_xi_minus" / "bin_1_1.txt", comments="#")

        theta_sys_rad = np.loadtxt(bestfit_dir / "xi_sys" / "theta.txt", comments="#")
        theta_sys = np.rad2deg(theta_sys_rad) * 60
        xip_sys = np.loadtxt(bestfit_dir / "xi_sys" / "shear_xi_plus.txt", comments="#")
        xim_sys = np.loadtxt(bestfit_dir / "xi_sys" / "shear_xi_minus.txt", comments="#")

    theta_fine = np.geomspace(0.5, 300, 500)
    if has_theory:
        xip_th_interp = interp1d(theta_theory, xip_theory, kind="cubic", fill_value="extrapolate")(theta_fine)
        xim_th_interp = interp1d(theta_theory, xim_theory, kind="cubic", fill_value="extrapolate")(theta_fine)
        xip_sys_interp = interp1d(theta_sys, xip_sys, kind="cubic", fill_value="extrapolate")(theta_fine)
        xim_sys_interp = interp1d(theta_sys, xim_sys, kind="cubic", fill_value="extrapolate")(theta_fine)

    scale_factor = 1e-4
    xlim = [1, 250]
    ylim = [-0.15, 1.25]

    ms_data = 3
    ms_bmode = 3
    capsize = 1.5
    elinewidth = 0.8

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    plot_configs = [
        (axes[0], xip_data, sigma_xip, xip_B, sigma_xip_B,
         xip_th_interp if has_theory else None, xip_sys_interp if has_theory else None,
         edge_cut_xip, r"$\xi_+$", "+"),
        (axes[1], xim_data, sigma_xim, xim_B, sigma_xim_B,
         xim_th_interp if has_theory else None, xim_sys_interp if has_theory else None,
         edge_cut_xim, r"$\xi_-$", "-"),
    ]

    for idx, (ax, xi_data_arr, sigma_xi, xi_B, sigma_B, xi_th, xi_sys_arr, edge_cut, label, _pm) in enumerate(plot_configs):
        show_legend = idx == 1

        ax.axvspan(xlim[0], edge_cut[0], color="0.90", zorder=0, alpha=0.7)
        ax.axvspan(edge_cut[1], xlim[1], color="0.90", zorder=0, alpha=0.7)

        if has_theory:
            ax.plot(
                theta_fine,
                theta_fine * (xi_th + xi_sys_arr) / scale_factor,
                "-",
                color="k",
                lw=1.5,
                label=r"Best-fit $\xi^{\mathrm{th}}_\pm + \xi^{\mathrm{sys}}_\pm$" if show_legend else None,
                zorder=2,
            )
            ax.plot(
                theta_fine,
                theta_fine * xi_sys_arr / scale_factor,
                "-",
                color="C0",
                lw=1.2,
                label=r"Best-fit $\xi^{\mathrm{sys}}_\pm$" if show_legend else None,
                zorder=2,
            )

        ax.errorbar(
            theta_data,
            theta_data * xi_data_arr / scale_factor,
            yerr=theta_data * sigma_xi / scale_factor,
            fmt="o",
            color="k",
            markersize=ms_data,
            capsize=capsize,
            elinewidth=elinewidth,
            label=r"$\xi_\pm$" if show_legend else None,
            zorder=3,
        )

        theta_eb_offset = theta_eb * 1.03
        ax.errorbar(
            theta_eb_offset,
            theta_eb_offset * xi_B / scale_factor,
            yerr=theta_eb_offset * sigma_B / scale_factor,
            fmt="o",
            color="C3",
            markersize=ms_bmode,
            capsize=capsize,
            elinewidth=elinewidth,
            alpha=0.85,
            label=r"$\xi^B_\pm$" if show_legend else None,
            zorder=3,
        )

        ax.axhline(0, color="gray", linestyle="--", alpha=0.8, linewidth=0.8, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$\theta$ (arcmin)")
        ax.set_title(label)
        if show_legend:
            ax.legend(loc="upper left")

    axes[0].set_ylabel(r"$\theta\xi \times 10^4$")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _get_stats_row(chain, label: str, color: str) -> list[str | float]:
    # Extracted from plot_whisker param_values build.
    margestats = chain.getMargeStats()
    try:
        s8_stats = margestats.parWithName("S_8")
    except Exception:
        s8_stats = margestats.parWithName("s_8_input")
    sigma8_stats = margestats.parWithName("SIGMA_8")
    omegam_stats = margestats.parWithName("OMEGA_M")
    return [
        label,
        color,
        s8_stats.mean,
        s8_stats.mean - s8_stats.limits[0].lower,
        s8_stats.limits[0].upper - s8_stats.mean,
        sigma8_stats.mean,
        sigma8_stats.mean - sigma8_stats.limits[0].lower,
        sigma8_stats.limits[0].upper - sigma8_stats.mean,
        omegam_stats.mean,
        omegam_stats.mean - omegam_stats.limits[0].lower,
        omegam_stats.limits[0].upper - omegam_stats.mean,
    ]


def get_sigma_tension(mean1, low1, high1, mean2, low2, high2):
    # Extracted from plot_whisker.
    sigma1 = 0.5 * (high1 + low1)
    sigma2 = 0.5 * (high2 + low2)
    delta_mean = np.abs(mean1 - mean2)
    sigma_tension = delta_mean / np.sqrt(sigma1**2 + sigma2**2)
    sign = 1 if mean1 > mean2 else -1
    return sigma_tension * sign


def plot_cell_ee_with_bestfit(
    cosmosis_data_path: str,
    bestfit_specs: list[tuple[str, str, dict]],
    output_folder: str,
    output_path: Path,
    ell_min: float = 10.0,
    ell_max: float = 2048.0,
    label_data: str = "Fiducial data",
) -> None:
    """Extracted from get_chi2_cell.ipynb plot_best_fit() (cell 21-22).

    Parameters
    ----------
    cosmosis_data_path : str
        Path to CosmoSIS FITS file with CELL_EE and COVMAT HDUs.
    bestfit_specs : list of (label, root, line_args_dict)
        Each entry is (legend label, chain root name, dict of plot kwargs).
    output_folder : str
        Base path to best_fit directories (e.g. /n09data/guerrini/output_chains/).
    output_path : Path
        Where to save the figure.
    """
    data = fits.getdata(cosmosis_data_path, "CELL_EE")
    cov_mat = fits.getdata(cosmosis_data_path, "COVMAT")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ell = data["ANG"]
    cell = data["VALUE"]
    ax.errorbar(
        ell,
        ell * cell,
        yerr=ell * np.sqrt(np.diag(cov_mat)),
        fmt="o",
        label=label_data,
        color="black",
        capsize=2,
    )

    for label, root, line_kw in bestfit_specs:
        ell_th = np.loadtxt(f"{output_folder}/best_fit/{root}/shear_cl/ell.txt")
        shear_cl = np.loadtxt(f"{output_folder}/best_fit/{root}/shear_cl/bin_1_1.txt")
        mask = (ell_th > ell_min) & (ell_th < ell_max)
        ax.plot(ell_th[mask], ell_th[mask] * shear_cl[mask], label=label, **line_kw)

    # Scale cut annotations (from notebook cell 21)
    ax.axvline(x=1800, color="black", linestyle="--", alpha=0.5)
    ax.axvline(x=2048, color="black", linestyle="--", alpha=1.0)
    ax.axvline(x=500, color="black", linestyle="--", alpha=0.3)

    ax.text(
        1740, 0.90, r"$k_\mathrm{max} = 3 h$ Mpc$^{-1}$",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=10, rotation=90,
    )
    ax.text(
        1978, 0.90, r"$k_\mathrm{max} = 5 h$ Mpc$^{-1}$",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=10, rotation=90,
    )
    ax.text(
        470, 0.90, r"$k_\mathrm{max} = 1 h$ Mpc$^{-1}$",
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=10, rotation=90,
    )

    ax.set_ylabel(r"$\ell C_\ell$", fontsize=16)
    ax.set_xlabel(r"$\ell$", fontsize=16)
    ax.set_xlim(ell.min() - 10, ell.max() + 100)
    ax.set_xscale("squareroot")
    ax.set_xticks(np.array([100, 400, 900, 1600]))
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", length=2, width=0.8)
    minor_ticks = [i * 10 for i in range(1, 10)] + [i * 100 for i in range(1, 21)]
    ax.xaxis.set_ticks(minor_ticks, minor=True)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=10)
    ax.yaxis.get_offset_text().set_fontsize(14)

    plt.legend(loc="lower center", bbox_to_anchor=(0.685, 0.70), fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_s8_whisker(
    chain_specs: list[ChainSpec],
    output_path: Path,
    reference_labels: list[str] | None = None,
    reference_colors: list[str] | None = None,
    reference_label: str | None = None,
) -> None:
    """Extracted from 2025_10_28_plot_whisker.ipynb (cells 5-11).

    Supports multiple reference bands: pass reference_labels and
    reference_colors as parallel lists. Each reference gets its own
    shaded band in the corresponding color. For backwards compatibility,
    a single reference_label still works.
    """
    # Backwards compatibility: single reference_label → list
    if reference_labels is None and reference_label is not None:
        reference_labels = [reference_label]
        reference_colors = reference_colors or [None]

    g, chains = load_getdist_chains(chain_specs, width_inch=30, axes_fontsize=60, axes_labelsize=60, legend_fontsize=60)
    plt.close(g.fig)

    labels = [spec.label for spec in chain_specs]
    colours = [spec.color for spec in chain_specs]
    alphas = [spec.alpha for spec in chain_specs]

    param_values = np.array(
        [["# Expt", "Colour", "S8_Mean", "S8_low", "S8_high", "sigma_8_Mean", "sigma_8_low", "sigma_8_high", "Omega_m_Mean", "Omega_m_low", "Omega_m_high"]],
        dtype=object,
    )

    escaped_labels = np.char.replace(np.array(labels), "\\", "\\\\")
    for i, chain in enumerate(chains):
        row = _get_stats_row(chain, escaped_labels[i], colours[i])
        param_values = np.vstack((param_values, row))

    expt = np.char.replace(param_values[1:, 0].astype(str), "\\\\", "\\")
    colours_arr = param_values[1:, 1].astype(str)
    s8_mean = param_values[1:, 2].astype(np.float64)
    s8_low = param_values[1:, 3].astype(np.float64)
    s8_high = param_values[1:, 4].astype(np.float64)
    sigma8_mean = param_values[1:, 5].astype(np.float64)
    sigma8_low = param_values[1:, 6].astype(np.float64)
    sigma8_high = param_values[1:, 7].astype(np.float64)
    omegam_mean = param_values[1:, 8].astype(np.float64)
    omegam_low = param_values[1:, 9].astype(np.float64)
    omegam_high = param_values[1:, 10].astype(np.float64)

    # Resolve reference indices
    ref_indices = []
    for rl in (reference_labels or []):
        matches = np.where(expt == rl)[0]
        if len(matches):
            ref_indices.append(matches[0])
    ref_label_set = set(reference_labels or [])

    from matplotlib.gridspec import GridSpec

    n_rows = len(expt)
    fig_height = max(6, 0.5 * n_rows + 1)
    fig = plt.figure(figsize=(10, fig_height))
    gs = GridSpec(1, 3, width_ratios=[1, 0.5, 0.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    ax3 = fig.add_subplot(gs[2], sharey=ax1)

    axs = [ax1, ax2, ax3]

    params = [
        (s8_mean, s8_low, s8_high, r"$S_8$"),
        (sigma8_mean, sigma8_low, sigma8_high, r"$\sigma_8$"),
        (omegam_mean, omegam_low, omegam_high, r"$\Omega_{\rm m}$"),
    ]

    row_spacing = 0.1
    y = np.arange(len(expt))

    for ax, param in zip(axs, params):
        means, lows, highs, label = param
        for i, mean, low, high, color, alpha in zip(y, means, lows, highs, colours_arr, alphas):
            ax.errorbar(mean, 0.05 + i * row_spacing, xerr=np.array([low, high])[:, None], fmt="o", color=color, ecolor=color, elinewidth=2, capsize=3, alpha=alpha)
        ax.set_xlabel(label, fontsize=14)

        # Draw reference bands — one per reference label
        for ri, ref_idx in enumerate(ref_indices):
            band_color = (reference_colors[ri] if reference_colors and ri < len(reference_colors) else colours_arr[ref_idx]) or colours_arr[ref_idx]
            ax.axvspan(means[ref_idx] - lows[ref_idx], means[ref_idx] + highs[ref_idx], color=band_color, alpha=0.15, zorder=0)

        ax.grid(False)
        ax.tick_params(axis="y", left=False, labelleft=False)
        if label == r"$S_8$":
            ax.set_xlim(0.25, 1.05)
        elif label == r"$\sigma_8$":
            ax.set_xlim(0.5, 1.2)
        elif label == r"$\Omega_{\rm m}$":
            ax.set_xlim(0.1, 0.5)

    axs[0].set_yticks(0.05 + y * row_spacing)
    axs[0].set_yticklabels([])
    for label, color, alpha in zip(expt, colours_arr, alphas):
        idx = np.where(expt == label)[0][0]
        yloc = 0.05 + row_spacing * idx
        axs[0].text(0.26, yloc, label, fontsize=12, ha="left", va="center", color=color, alpha=alpha)
        # Sigma tension relative to first reference (if not a reference itself)
        if label not in ref_label_set and ref_indices:
            ri0 = ref_indices[0]
            s8_tension = get_sigma_tension(s8_mean[idx], s8_low[idx], s8_high[idx], s8_mean[ri0], s8_low[ri0], s8_high[ri0])
            sign_str = "+" if s8_tension > 0 else "-"
            axs[0].text(1.045, yloc, rf"${sign_str}{np.abs(s8_tension):.2f}" + r"\, \sigma$", fontsize=10, ha="right", va="center", color=color, alpha=alpha)

    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
