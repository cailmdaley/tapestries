from functools import cached_property

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import skyproj
from matplotlib.colors import LogNorm, Normalize


# %%
class HealpixCoverage:
    def __init__(self, nside, ra, dec, w=None, nest=False, verbose=False, sparse=True):
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.nest = nest
        self.sparse = sparse
        self.verbose = verbose

        verbose and print(f"Calculating healpix coverage for {len(ra):,} galaxies...")

        self.ipix = hp.ang2pix(nside, ra, dec, lonlat=True, nest=nest)
        self.w = w

        if self.sparse:
            self.ipix_sparse, self.ipix_catalog = np.unique(
                self.ipix, return_inverse=True
            )

            self.verbose and print(
                f"Sparse mode: {len(self.ipix_sparse) / self.npix * 100:.2f}% of NSIDE={self.nside} pixels used."
            )

    def bin_map(self, vals=None, norm=False):
        if self.sparse:
            m = np.bincount(self.ipix_catalog, weights=vals)
        else:
            m = np.bincount(self.ipix, weights=vals, minlength=self.npix)

        if norm:
            m[self.weightmap > 0] /= self.weightmap[self.weightmap > 0]

        return m

    @cached_property
    def weightmap(self):
        return self.bin_map(self.w)

    def densify(self, m):
        """Convert a sparse map to a dense map."""
        assert len(m) == len(
            self.ipix_sparse
        ), "Map length must match unique pixel indices length."
        dense_map = np.zeros(self.npix, dtype=m.dtype)
        dense_map[self.ipix_sparse] = m
        return dense_map

    def sparsify(self, m):
        """Convert a dense map to a sparse map."""
        assert len(m) == self.npix, "Map length must match npix."
        return m[self.ipix_sparse]

    def smooth_map(self, m, fwhm=1.0):
        """Smooth a map using a Gaussian kernel."""
        if self.sparse:
            m = self.densify(m)
        m_smooth = hp.smoothing(m, fwhm=fwhm)
        return self.sparsify(m_smooth)


class RR2Coverage(HealpixCoverage):
    @classmethod
    def from_mask(cls, nside, mask_maps, nest=False, sparse=True):
        """Create RR2Coverage instance from HEALPix mask maps.

        Parameters
        ----------
        nside : int
            HEALPix NSIDE parameter
        mask_maps : array-like or list of array-like
            Single mask or list of HEALPix mask maps
        nest : bool, optional
            HEALPix ordering scheme (default: False for RING)
        sparse : bool, optional
            Use sparse representation (default: True)

        Returns
        -------
        RR2Coverage
            Coverage instance with pixel set from mask union
        """
        # Handle single mask input
        if not isinstance(mask_maps, (list, tuple)):
            mask_maps = [mask_maps]

        # Sum all masks and find non-zero pixels
        combined_mask = np.sum(mask_maps, axis=0)
        ipix_mask = np.where(combined_mask > 0)[0]

        # Generate dummy coordinates at pixel centers
        ra, dec = hp.pix2ang(nside, ipix_mask, lonlat=True, nest=nest)

        # Create instance using standard constructor
        return cls(nside=nside, ra=ra, dec=dec, nest=nest, sparse=sparse)

    def plot(
        self, m, cmap="Blues", vlim="asym", vscale="linear", vlim_percentiles=(0, 100)
    ):
        if vscale not in ["log", "linear"]:
            raise ValueError("`vscale` should be 'log' or 'linear'")
        if vlim == "sym":
            vmax = np.percentile(np.abs(m[np.isfinite(m)]), vlim_percentiles[1])
            vmin = -vmax
        elif vlim == "asym":
            (vmin, vmax) = np.percentile(m[np.isfinite(m)], vlim_percentiles)
        elif isinstance(vlim, tuple):
            (vmin, vmax) = vlim
        else:
            (vmin, vmax) = -vlim, vlim
        norm = LogNorm(vmin, vmax) if vscale == "log" else Normalize(vmin, vmax)

        fig, axes = plt.subplots(
            1, 2, figsize=(20, 10), 
            gridspec_kw={"width_ratios": [46.5, 53.5]},
            layout="constrained"
        )

        sp = skyproj.GnomonicSkyproj(
            axes[0], lon_0=70, lat_0=-33, extent=[63.8, 77.2, -37, -26.5]
        )
        sp.draw_hpxpix(
            self.nside,
            self.ipix_sparse,
            m,
            nest=self.nest,
            zoom=False,
            cmap=cmap,
            norm=norm,
        )
        sp.draw_inset_colorbar(loc="upper right")

        sp = skyproj.GnomonicSkyproj(
            axes[1], lon_0=45, lat_0=-64, extent=[32, 70, -44, -63.5]
        )
        sp.draw_hpxpix(
            self.nside,
            self.ipix_sparse,
            m,
            nest=self.nest,
            zoom=False,
            cmap=cmap,
            norm=norm,
        )
        sp.draw_inset_colorbar(loc="upper right")

        return fig, axes
