import numpy as np

from camb import model, initialpower
import camb

import pyccl as ccl

# ----------------------------
# -Common defaults parameters-
# ----------------------------
# https://wiki.cosmos.esa.int/planck-legacy-archive/images/4/43/Baseline_params_table_2018_68pc_v2.pdf
# base plikHM TTTEEE lowl lowE lensing post BAO
# or table 2 last column of https://www.aanda.org/10.1051/0004-6361/201833910
defaults_params = {
    "As": 2.105e-9,
    "ns": 0.9665,
    "tau": 0.0561,
    "Omb": 0.04897468162,
    "Omc": 0.2606667599,
    "Omk": 0,
    "H0": 67.66,
    "mnu": 0.06,
    "lmax": 2000,
    "lmin_limber": 100,
}

# DEMNUNI DEFAULTS
# defaults_params = {
#     "As": 2.1265e-9,
#     "ns": 0.96,
#     "tau": 0.0561,
#     "Omb": 0.05,
#     "Omc": 0.27,
#     "Omk": 0,
#     "H0": 67.,
#     "mnu": 0.,
#     "lmax": 2000,
#     "lmin_limber": 300,
# }
_h = defaults_params["H0"] / 100
defaults_params["h"] = _h
defaults_params["ombh2"] = defaults_params["Omb"] * _h * _h
defaults_params["omch2"] = defaults_params["Omc"] * _h * _h

# --------------------------
# -CAMB defaults parameters-
# --------------------------
params_default_camb = camb.CAMBparams()
params_default_camb.InitPower.set_params(
    As=defaults_params["As"], ns=defaults_params["ns"]
)

params_default_camb.set_cosmology(
    H0=defaults_params["H0"],
    ombh2=defaults_params["ombh2"],
    omch2=defaults_params["omch2"],
    mnu=defaults_params["mnu"],
    tau=defaults_params["tau"],
)
params_default_camb.set_for_lmax(defaults_params["lmax"], lens_potential_accuracy=1)

# params_default_camb.Want_CMB = False
# params_default_camb.Want_CMB_lensing = False
# params_default_camb.Want_transfer = False

# params_default_camb.NonLinear = model.NonLinear_both

# params_default_camb.SourceTerms.limber_windows = True
# params_default_camb.SourceTerms.limber_phi_lmin = defaults_params["lmin_limber"]
# params_default_camb.SourceTerms.counts_lensing = True
# params_default_camb.SourceTerms.counts_density = True

# params_default_camb.Accuracy.LensingBoost = 1.0
# params_default_camb.Accuracy.NonlinSourceBoost = 1.0
# params_default_camb.Accuracy.BesselBoost = 1.0
# params_default_camb.Accuracy.LimberBoost = 1.0
# params_default_camb.Accuracy.SourceLimberBoost = 2.0
# params_default_camb.Accuracy.SourcekAccuracyBoost = 2.0

params_default_camb.SourceTerms.limber_windows = True
params_default_camb.SourceTerms.limber_phi_lmin = defaults_params["lmin_limber"]
params_default_camb.SourceTerms.counts_density = True
params_default_camb.SourceTerms.counts_redshift = True
params_default_camb.SourceTerms.counts_lensing = False
params_default_camb.SourceTerms.counts_velocity = True
params_default_camb.SourceTerms.counts_radial = False
params_default_camb.SourceTerms.counts_timedelay = False
params_default_camb.SourceTerms.counts_ISW = True
params_default_camb.SourceTerms.counts_potential = True

params_default_camb.NonLinearModel.Min_kh_nonlinear = 0.005
params_default_camb.NonLinearModel.halofit_version = "mead2020"
params_default_camb.NonLinearModel.HMCode_A_baryon = 3.13
params_default_camb.NonLinearModel.HMCode_eta_baryon = 0.603
params_default_camb.NonLinearModel.HMCode_logT_AGN = 7.8

# CCL defaults

cosmo_ccl = {
    "Omega_c": defaults_params["Omc"],
    "Omega_b": defaults_params["Omb"],
    "Omega_k": defaults_params["Omk"],
    "h": defaults_params["h"],
    "n_s": defaults_params["ns"],
    "A_s": defaults_params["As"],
    "m_nu": defaults_params["mnu"],
}
params_default_ccl = ccl.Cosmology(**cosmo_ccl)
