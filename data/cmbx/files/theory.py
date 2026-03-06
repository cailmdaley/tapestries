import numpy as np
from scipy.interpolate import interp1d
import camb
from camb.sources import SplinedSourceWindow

import pyccl as ccl

from .defaults import params_default_camb, params_default_ccl
import copy



class Theory:
    def __init__(
        self,
        input_file=None,
        dndz=None,
        dndz_marginalization = False,
        dndz_marg_type="global",
        include_lensing=False,
        include_weaklensing=False,
        theory_provider="camb",
        theory_params=None,
        extra_params=None,
        include_nl_bias=False,
        higher_order_bias=False,
        higher_order_bias_bs= False,
        include_mag_bias=False,
        include_rsd=False,
        include_ia_bias=False,
        mag_bias_z=None,
        zmin=None,
        zmax=None,
        lmax=None,
        bias_type="global",
        set_bias_cst=False,
        dndz_threshold=0,
        bias_model='SPV3',  # 'SPV3' or 'SGB'
    ):

        assert input_file != None or dndz != None, "Provide either input_file or dndz"
        if input_file != None:
            data = np.load(input_file, allow_pickle=True)
            self.dndz = data["dndz"]

        elif dndz != None:
            self.dndz = dndz
        
        self.dndz_marginalization = dndz_marginalization
        self.dndz_marg_type = dndz_marg_type
        self.include_lensing = include_lensing
        self.include_weaklensing = include_weaklensing
        self.include_nl_bias = include_nl_bias
        self.higher_order_bias = higher_order_bias
        self.higher_order_bias_bs = higher_order_bias_bs
        ####
        self.include_mag_bias = include_mag_bias
        self.include_ia_bias = include_ia_bias
        self.include_rsd = include_rsd
        self.bias_type = bias_type
        self.nbin = len(self.dndz)
        self.set_bias_cst = set_bias_cst
        self.name_bins = list(self.dndz.keys())

        self.theory_provider = theory_provider

        if extra_params == None:
            self.extra_params = None
        if isinstance(extra_params, dict):
            self.extra_params = extra_params

        if zmin == None:
            self.izmin = 0
        else:
            assert zmin >= self.dndz[self.name_bins[0]][0][0], "provided zmin too small"
            self.izmin = np.argmin(np.abs(self.dndz[self.name_bins[0]][0] - zmin))

        self.izmax = {}

        if zmax == None:
            for i, n in enumerate(self.name_bins):
                self.izmax[n] = len(self.dndz[self.name_bins[0]][0])
        
        elif type(zmax) == list:
            self.izmax = {}
            for i, n in enumerate(self.name_bins):
                assert (
                    zmax[i] <= self.dndz[self.name_bins[i]][0][-1]
                ), "provided zmax too large"
                self.izmax[n] = np.argmin(np.abs(self.dndz[self.name_bins[i]][0] - zmax[i]))

        else:
            assert (
                zmax <= self.dndz[self.name_bins[0]][0][-1]
            ), "provided zmax too large"
            for i, n in enumerate(self.name_bins):
                self.izmax[n] = np.argmin(np.abs(self.dndz[self.name_bins[0]][0] - zmax))
        
        self.dndz_marg = {}
        # Normalize dndz to unity
        for n in self.name_bins:
            self.dndz[n] = list(self.dndz[n]) # FIX to transform tuple to list
            self.dndz[n][1] = (
                self.dndz[n][1] / np.trapz(self.dndz[n][1], self.dndz[n][0])
            )
        # Set dndz to zero if below threshold
        if dndz_threshold > 0:
            for n in self.name_bins:
                self.dndz[n][1][np.argwhere(self.dndz[n][1]<dndz_threshold)] = 0

        self.bias = {}
        for n in self.name_bins:
            if self.set_bias_cst:
                self.bias[n] = np.ones_like( self.dndz[n][0][self.izmin : self.izmax[n] + 1])
            else:
                if bias_model =='SPV3':
                    self.bias[n] = (
                        0.8307
                        + 1.1905 * self.dndz[n][0][self.izmin : self.izmax[n] + 1]
                        - 0.9283 * self.dndz[n][0][self.izmin : self.izmax[n] + 1] ** 2
                        + 0.4233 * self.dndz[n][0][self.izmin : self.izmax[n] + 1] ** 3
                    )
                elif bias_model == 'SGB':
                    self.bias[n] = (
                        1.47290623
                        - 1.03782755 * self.dndz[n][0][self.izmin : self.izmax[n] + 1]
                        + 1.31617606 * self.dndz[n][0][self.izmin : self.izmax[n] + 1] ** 2
                        - 0.22822347 * self.dndz[n][0][self.izmin : self.izmax[n] + 1] ** 3
                    )

        if self.include_nl_bias:
            self.bias2 = {}
            for n in self.name_bins:
                if self.set_bias_cst:
                    self.bias2[n] = np.ones_like( self.dndz[n][0][self.izmin : self.izmax[n] + 1])
                else:
                    self.bias2[n] = (
                        -0.69682803
                        + 1.60320679 * self.dndz[n][0][self.izmin : self.izmax[n] + 1]
                        - 1.31676159 * self.dndz[n][0][self.izmin : self.izmax[n] + 1] ** 2
                        + 0.70271383 * self.dndz[n][0][self.izmin : self.izmax[n] + 1] ** 3
                    )
        if self.include_mag_bias:
            self.mag_bias = {}
            for n in self.name_bins:
                if mag_bias_z is None:
                    self.mag_bias[n] = (
                        -1.50685
                        + 1.35034 * self.dndz[n][0][self.izmin : self.izmax[n] + 1]
                        + 0.08321 * self.dndz[n][0][self.izmin : self.izmax[n] + 1] ** 2
                        + 0.04279 * self.dndz[n][0][self.izmin : self.izmax[n] + 1] ** 3
                        # -1.41
                        # + 0.88 * self.dndz[n][0][self.izmin : self.izmax[n] + 1]
                        # + 0.45 * self.dndz[n][0][self.izmin : self.izmax[n] + 1] ** 2
                    )

                else:
                    self.mag_bias[n] = mag_bias_z[n] * np.ones_like(
                        self.dndz[n][0][self.izmin : self.izmax[n] + 1]
                    )

        if self.include_ia_bias:
            self.A_IA = 1.72
            self.C_IA = 0.013
            self.eta_IA = -0.41
            self.beta_IA = 2.17

        if self.theory_provider == "camb":
            if theory_params == None:
                theory_params = params_default_camb

            sources = []
            for n in self.name_bins:
                sources.append(
                    SplinedSourceWindow(
                        bias_z=self.bias[n],
                        z=self.dndz[n][0][self.izmin : self.izmax[n] + 1],
                        W=self.dndz[n][1][self.izmin : self.izmax[n] + 1],
                        source_type="counts",
                    )
                )

            if self.include_weaklensing:
                for n in self.name_bins:
                    sources.append(
                        SplinedSourceWindow(
                            z=self.dndz[n][0][self.izmin : self.izmax[n] + 1],
                            W=self.dndz[n][1][self.izmin : self.izmax[n] + 1],
                            source_type="lensing",
                        )
                    )

            theory_params.SourceWindows = sources

            if lmax == None:
                self.lmax = theory_params.max_l
            else:
                self.lmax = lmax
                theory_params.max_l = lmax

            self.ells = np.arange(self.lmax + 1)

        elif self.theory_provider == "ccl":
            if theory_params == None:
                theory_params = params_default_ccl

            tracers = []
            tracers_name = []
            if self.include_nl_bias:
                tracers_nl = []
                self.ptc = ccl.nl_pt.EulerianPTCalculator(
                    with_NC=True,
                    with_IA=True,
                    log10k_min=-5,
                    log10k_max=3,
                    nk_per_decade=40,
                )
                self.ptc.update_ingredients(theory_params)

            if include_lensing:
                tracers.append(
                    ccl.tracers.CMBLensingTracer(
                        cosmo=theory_params,
                        z_source=1089,
                        n_samples=100,
                    )
                )
                tracers_name.append("P")
                if self.include_nl_bias:
                    tracers_nl.append(ccl.nl_pt.PTMatterTracer())

            for i, n in enumerate(self.name_bins):
                zeta = self.dndz[n][0][self.izmin : self.izmax[n] + 1]
                dndz = self.dndz[n][1][self.izmin : self.izmax[n] + 1]
                
                if self.include_nl_bias:
                    if self.higher_order_bias:
                        # print('all biases')
                        bs = (zeta, (-4 / 7) * (self.bias[n] - 1))
                        b3nl = (zeta, self.bias[n] - 1)
                    elif self.higher_order_bias_bs:
                        # print('bias bs')
                        bs = (zeta, (-4 / 7) * (self.bias[n] - 1))
                        b3nl = None

                    else:
                        bs = None
                        b3nl = None
                    b1 = np.ones_like(self.bias[n])
                    tracers_nl.append(
                        ccl.nl_pt.PTNumberCountsTracer(
                            b1=(zeta, self.bias[n]),
                            b2=(zeta, self.bias2[n]),
                            bs=bs,
                            b3nl=b3nl,
                        )
                    )
                else:
                    b1 = self.bias[n]

                if self.include_mag_bias:
                    mag_bias = (zeta, (self.mag_bias[n] + 2) / 5)
                else:
                    mag_bias = None
                if self.include_ia_bias:
                    # F_IA = lambda z, L_mean, L_star : -A_IA * C_IA * (Omegab+Omegac)*(1+z)**eta_IA * (L_mean / L_star) ** beta_IA
                    ia_bias = (
                        zeta,
                        -0.004 * np.ones_like(zeta),
                    )  # TODO: Add IA bias parameters
                else:
                    ia_bias = None

                tracers.append(
                    ccl.tracers.NumberCountsTracer(
                        cosmo=theory_params,
                        dndz=(zeta, dndz),
                        bias=(zeta, b1),
                        mag_bias=mag_bias,
                        has_rsd=self.include_rsd,
                        n_samples=256,
                    )
                )
                tracers_name.append("G" + str(i))

            if self.include_weaklensing:

                for i, n in enumerate(self.name_bins):
                    zeta = self.dndz[n][0][self.izmin : self.izmax[n] + 1]
                    dndz = self.dndz[n][1][self.izmin : self.izmax[n] + 1]

                    tracers.append(
                        ccl.tracers.WeakLensingTracer(
                            cosmo=theory_params,
                            dndz=(zeta, dndz),
                            has_shear=True,
                            n_samples=256,
                            ia_bias=ia_bias,
                        )
                    )
                    tracers_name.append("L" + str(i))

            self.tracers = tracers
            self.tracers_name = tracers_name

            if self.include_nl_bias:
                self.tracers_nl = tracers_nl

            if lmax == None:
                self.lmax = 1500
            else:
                self.lmax = lmax

            self.ells = np.arange(2, self.lmax + 1)

        else:
            raise ValueError(
                "Theory {} not available, either 'camb' or 'ccl'".format(
                    self.theory_provider
                )
            )

        self.params = theory_params

    def update_parameters(self, **params_values):

        if self.theory_provider == "camb":
            h = params_values.get("h", self.params.H0 / 100)
            self.params.H0 = h * 100
            self.params.omch2 = (
                params_values.get("Omega_c", self.params.omch2 / h / h) * h * h
            )
            self.params.ombh2 = (
                params_values.get("Omega_b", self.params.ombh2 / h / h) * h * h
            )
            self.params.omk = params_values.get("Omega_k", self.params.omk)
            self.params.InitPower.As = params_values.get(
                "A_s", self.params.InitPower.As
            )
            self.params.InitPower.n_s = params_values.get(
                "n_s", self.params.InitPower.ns
            )

        elif self.theory_provider == "ccl":

            # We will update it only if cosmology parameters are updated
            # self.params is a ccl.Cosmology object store in cache
            cached_params = self.params.to_dict()

            update_cosmo = False
            d = copy.deepcopy(cached_params)
            for k in ["h", "Omega_c", "Omega_b", "A_s", "n_s", "Omega_k", "m_nu"]:
                d[k] = params_values.get(k, params_default_ccl[k])
                if d[k] == None:
                    d[k] = params_default_ccl[k]

                if d[k] != cached_params[k]:
                    update_cosmo = True

            if update_cosmo:
                if isinstance(self.extra_params, dict):
                    d["extra_parameters"] = self.extra_params
                    if "matter_power_spectrum" in d:
                        d.pop("matter_power_spectrum")
                        self.params = ccl.Cosmology(**d, matter_power_spectrum="camb")
                else:
                    self.params = ccl.Cosmology(**d)

                if self.include_nl_bias:
                    ccl.growth_factor(self.params, 1.0)  # Compute growth factor at z=0

            ## bias parameters
            ## add if on the type of the bias

            for i in range(self.nbin):
                if self.bias_type == "global":
                    d[f"bias_{i}"] = params_values.get("bias_global", 1.0)
                    if self.include_nl_bias:
                        d[f"bias_nl_{i}"] = params_values.get("bias_nl_global", 1.0)

                elif self.bias_type == "binwise":
                    d[f"bias_{i}"] = params_values.get(f"bias_{i}", 1.0)
                    if self.include_nl_bias:
                        d[f"bias_nl_{i}"] = params_values.get(f"bias_nl_{i}", 1.0)
                else:
                    assert False, "Bias type not recognized"

            # We redefine the tracers, can we do more optimal by just updating the biases ?
            if self.include_nl_bias:
                self.ptc.update_ingredients(self.params)
                tracers_nl = []

            tracers = []

            if self.include_lensing:
                tracers.append(
                    ccl.tracers.CMBLensingTracer(
                        cosmo=self.params,
                        z_source=1089,
                        n_samples=100,
                    )
                )
                if self.include_nl_bias:
                    tracers_nl.append(ccl.nl_pt.PTMatterTracer())

            for i, n in enumerate(self.name_bins):
                zeta = self.dndz[n][0][self.izmin : self.izmax[n] + 1]
                z_array = self.dndz[n][0]
                if self.dndz_marginalization:                                ##https://arxiv.org/pdf/2301.11895
                    dndz_i = self.dndz[n][1]
                    dndz_int = interp1d(z_array, dndz_i, fill_value=0, bounds_error=False)
                    z_c = z_array[np.argmax(dndz_i)]
                    if self.dndz_marg_type == "global":
                        d[f"wz_{i}"] = params_values.get("wz_global", 1.0)
                        d[f"Dz_{i}"] = params_values.get("Dz_global", 0.0)
                    elif self.dndz_marg_type == "binwise":
                        d[f"wz_{i}"] = params_values.get(f"wz_{i}", 1.0)
                        d[f"Dz_{i}"] = params_values.get(f"Dz_{i}", 0.0)
                    # print(d)
                    dndz = dndz_int(z_c + d[f"wz_{i}"]*(zeta-z_c)+d[f"Dz_{i}"])
                    self.dndz_marg[n] = dndz
                    
                else:
                    dndz = self.dndz[n][1][self.izmin : self.izmax[n] + 1]


                if self.include_mag_bias:
                    mag_bias = (zeta, (self.mag_bias[n] + 2) / 5)
                else:
                    mag_bias = None

                if self.include_ia_bias:
                    # F_IA = lambda z, L_mean, L_star : -A_IA * C_IA * (Omegab+Omegac)*(1+z)**eta_IA * (L_mean / L_star) ** beta_IA
                    ia_bias = (
                        zeta,
                        -0.004 * np.ones_like(zeta),
                    )  # TODO: Add IA bias parameters
                else:
                    ia_bias = None

                ## updates dei tracers
                if self.include_nl_bias:
                    if self.higher_order_bias:
                        # print('all biases')
                        bs = (zeta, (-4 / 7) * (d[f"bias_{i}"] * self.bias[n] - 1))
                        b3nl = (zeta, d[f"bias_{i}"] * self.bias[n] - 1)
                    elif self.higher_order_bias_bs:
                        # print('bias bs')
                        bs = (zeta, (-4 / 7) * (d[f"bias_{i}"] * self.bias[n] - 1))
                        b3nl = None
                    # elif self.higher_order_bias_b3nl:
                    #     print('bias b3nl')
                    #     bs = None
                    #     b3nl = (zeta, self.bias[n] - 1)
                    else:
                        bs = None
                        b3nl = None
                    bias_ = np.ones_like(
                        self.bias[n]
                    )  # In case of nl bias, bias for tracer is set to 1
                    tracers_nl.append(
                        ccl.nl_pt.PTNumberCountsTracer(
                            b1=(zeta, d[f"bias_{i}"] * self.bias[n]),
                            b2=(zeta, d[f"bias_nl_{i}"] * self.bias2[n]),
                            bs=bs,
                            b3nl=b3nl,
                        )
                    )
                else:
                    bias_ = d[f"bias_{i}"] * self.bias[n]

                tracers.append(
                    ccl.tracers.NumberCountsTracer(
                        cosmo=self.params,
                        dndz=(zeta, dndz),
                        bias=(zeta, bias_),
                        mag_bias=mag_bias,
                        has_rsd=self.include_rsd,
                        n_samples=256,
                    )
                )
            if self.include_weaklensing:
                for i, n in enumerate(self.name_bins):
                    zeta = self.dndz[n][0][self.izmin : self.izmax[n] + 1]
                    dndz = self.dndz[n][1][self.izmin : self.izmax[n] + 1]
                    tracers.append(
                        ccl.tracers.WeakLensingTracer(
                            cosmo=self.params,
                            dndz=(zeta, dndz),
                            has_shear=True,
                            n_samples=256,
                            ia_bias=ia_bias,
                        )
                    )
            self.tracers = tracers
            if self.include_nl_bias:
                self.tracers_nl = tracers_nl

        else:
            raise ValueError(
                "Theory {} not available, either 'camb' or 'ccl'".format(
                    self.theory_provider
                )
            )

    ## compute Cls with theory+nuissance parameters
    def get_theory_cls(self, **kwargs):
        if self.theory_provider == "camb":
            self.update_parameters(**kwargs)
            results = camb.get_results(self.params)
            cls_theory = results.get_source_cls_dict(raw_cl=True)
            for k in cls_theory.keys():
                # Transform CMB lensing Phi into Kappa
                ka, kb = k.split("x")
                if ka == "P":
                    cls_theory[k] = self.ells * (self.ells + 1) / 2 * cls_theory[k]
                if kb == "P":
                    cls_theory[k] = self.ells * (self.ells + 1) / 2 * cls_theory[k]

            # Transforms CAMB keys to match CCL keys
            cls_theory_ = {}
            label_map = {"P": "P"}
            for i in range(self.nbin):
                label_map[f"W{i+1}"] = f"G{i}"
                label_map[f"W{i+1+self.nbin}"] = f"L{i}"

            for k in cls_theory.keys():
                ka, kb = k.split("x")
                cls_theory_[label_map[ka], label_map[kb]] = cls_theory[k]
            cls_theory = cls_theory_

        elif self.theory_provider == "ccl":

            # if isinstance(self.extra_params, dict):         ## if extra_parameter then update the dictionary
            self.update_parameters(
                **kwargs
            )  ## update parameters both nuisance+cosmology
            cls_theory = {}
            if self.include_nl_bias:
                for i, tri in enumerate(self.tracers):
                    for j, trj in enumerate(self.tracers):
                        if j < i:
                            continue
                        pk_nl = self.ptc.get_biased_pk2d(
                            tracer1=self.tracers_nl[i],
                            tracer2=self.tracers_nl[j],
                        )
                        name_cls = (self.tracers_name[i], self.tracers_name[j])
                        cls_theory[name_cls] = ccl.angular_cl(
                            self.params,
                            tri,
                            trj,
                            self.ells,
                            p_of_k_a=pk_nl,
                            limber_integration_method="spline",
                        )
                        cls_theory[name_cls] = np.insert(
                            cls_theory[name_cls], 0, [0, 0]
                        )

            else:
                for i, tri in enumerate(self.tracers):
                    for j, trj in enumerate(self.tracers):
                        if j < i:
                            continue
                        name_cls = (self.tracers_name[i], self.tracers_name[j])
                        cls_theory[name_cls] = ccl.angular_cl(
                            self.params,
                            tri,
                            trj,
                            self.ells,
                            limber_integration_method="spline",
                        )
                        cls_theory[name_cls] = np.insert(
                            cls_theory[name_cls], 0, [0, 0]
                        )

        else:
            raise ValueError(
                "Theory {} not available, either 'camb' or 'ccl'".format(
                    self.theory_provider
                )
            )

        return cls_theory

