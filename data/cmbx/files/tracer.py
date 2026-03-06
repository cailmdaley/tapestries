from dataclasses import dataclass, field
from functools import cached_property
import numpy as np
import healpy as hp
import pymaster as nmt


@dataclass(eq=False)
class Tracer:
    """
    Base class (not useable on its own) representing a field defined on the sky.
    """

    name: str
    beam: np.array = field(repr=False, default=None, kw_only=True)
    dndz: tuple[np.array, np.array] = field(repr=False, default=None, kw_only=True)
    spin: int = field(init=False)
    is_cat_field: bool = field(init=False, repr=False)  # easy way to test if field is constructed from a catalog instead of a map


@dataclass
class MapTracer(Tracer):
    """
    A class representing a map-based field defined on the sky.

    Required arguments:
    name: str - a name describing the tracer
    maps: list of arrays - maps (1 for spin-0 fields or 2 for spin-2 fields) defining the field values on the sky
    mask: array - sky mask

    Optional arguments:
    beam: array - instrument beam or smoothing that has been applied to the field
    dndz: tuple of arrays - (z, dndz), redshift distribution of the tracer
    masked_on_input: bool 

    Attributes:
    nside: int - the healpix nside parameter for this tracer
    spin: int - spin of this tracer
    field: NmtField - namaster field object for this tracer
    """

    nside: int = field(init=False)
    maps: list[np.array] = field(repr=False)
    mask: np.array = field(repr=False)
    masked_on_input: bool = field(repr=False, default=False, kw_only=True)

    def __post_init__(self):
        self.is_cat_field = False
        for m in self.maps:
            assert len(m) == len(self.mask), "Maps and masks must all be the same size"
        self.nside = hp.npix2nside(len(self.mask))
        if self.beam is None:
            self.beam = np.ones(3*self.nside)
        assert len(self.beam) == 3*self.nside, "Beam is incorrect size for given nside"
        if len(self.maps) == 1:
            self.spin = 0
        elif len(self.maps) == 2:
            self.spin = 2
        else:
            raise ValueError("Only spin-0 or spin-2 supported")

    @cached_property
    def field(self):
        # namaster field
        return nmt.NmtField(self.mask, self.maps, spin=self.spin, beam=self.beam, masked_on_input=self.masked_on_input)


@dataclass
class CatalogTracer(Tracer):
    """
    A class representing a either a field sampled at discrete points on the sky or
      a field representing the clustering of discrete points on the sky.

    Required arguments:
    name: str
    pos: np.array
    weights: np.array
    lmax: int

    Required arguments for a sampled field:
    fields: np.array

    Required arguments for a clustering field:
    pos_rand: np.array
    weights_rand: np.array

    Optional arguments:
    lonlat: bool
    field_is_weighted: bool
    """

    pos: np.array = field(repr=False)
    weights: np.array = field(repr=False)
    lmax: int
    lonlat: bool = field(repr=False, default=True, kw_only=True)

    # only for NmtFieldCatalog:
    fields: list[np.array] = field(repr=False, default=None, kw_only=True)
    field_is_weighted: bool = field(repr=False, default=False, kw_only=True)

    # only for NmtFieldCatalogClustering:
    pos_rand: np.array = field(repr=False, default=None, kw_only=True)
    weights_rand: np.array = field(repr=False, default=None, kw_only=True)


    def __post_init__(self):
        self.is_cat_field = True
        if self.fields is None and self.pos_rand is None:
            raise ValueError("Must provide either field values or a randoms catalog")

        if self.fields is not None:
            # create a NmtFieldCatalog object
            if len(self.fields) == 1:
                self.spin = 0
            elif len(self.fields) == 2:
                self.spin = 2
            else:
                raise ValueError("Must provide either 1 or 2 fields")
            if self.beam is None:
                self.beam = np.ones(self.lmax+1)
            assert len(self.beam) >= self.lmax+1, "beam is incorrect size for given lmax"
        else:
            # create a NmtFieldCatalogClustering object
            self.spin = 0

    @cached_property
    def field(self):
        if self.fields is not None:
            # Field sampled at catalog locations
            return nmt.NmtFieldCatalog(self.pos, self.weights, self.fields, self.lmax, spin=self.spin, beam=self.beam,
                                       field_is_weighted=self.field_is_weighted, lonlat=self.lonlat)
        # clustering of catalog
        return nmt.NmtFieldCatalogClustering(self.pos, self.weights, self.pos_rand, self.weights_rand, self.lmax, lonlat=self.lonlat)
