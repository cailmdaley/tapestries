import numpy as np
from cobaya.likelihood import Likelihood
from scipy import stats
from dr1_cmbx import eDR1like
import os
from scipy.integrate import simpson

# module_dir = os.path.dirname(os.path.dirname(eDR1like.__file__))
data_dir = '/leonardo_work/EUHPC_E05_083/llegrand/q1_cmbx/' # Temporary, need to decide where to store the data for DR1_CMBX

class LikeCobayaPyccl(Likelihood):
    def initialize(self):
        self.abspath = os.path.join(data_dir, self.path)
        if self.input_file == None:
            self.input_file = os.path.join(
                self.abspath,
                f"cls_{self.field}_{self.nbins}_{self.cmb_exp}_{self.cut}_{self.redshift}_{self.mask}.pkl",
            )
        print(f"[LikeCobaya] Loading data from {self.input_file}")
        data = np.load(self.input_file, allow_pickle=True)
        self.data = data
        self.nspectra = len(data["cls"])
        if self.data_type == "q1":
            if self.order_data == None:
                if self.nspectra == 1:
                    self.order_data = ["g0_g0"]
                elif self.nspectra == 2:
                    self.order_data = ["kappa_g0", "g0_g0"]
                elif self.nspectra == 3:
                    self.order_data = ["g0_g0", "g0_g1", "g1_g1"]
                elif self.nspectra == 5:
                    self.order_data = [
                        "kappa_g0",
                        "kappa_g1",
                        "g0_g0",
                        "g0_g1",
                        "g1_g1",
                    ]
                elif self.nspectra == 6:
                    self.order_data = [
                        "g0_g0",
                        "g0_g1",
                        "g0_g2",
                        "g1_g1",
                        "g1_g2",
                        "g2_g2",
                    ]
                elif self.nspectra == 9:
                    self.order_data = [
                        "kappa_g0",
                        "kappa_g1",
                        "kappa_g2",
                        "g0_g0",
                        "g0_g1",
                        "g0_g2",
                        "g1_g1",
                        "g1_g2",
                        "g2_g2",
                    ]
                elif self.nspectra == 10:
                    self.order_data = [
                        "g0_g0",
                        "g0_g1",
                        "g0_g2",
                        "g0_g3",
                        "g1_g1",
                        "g1_g2",
                        "g1_g3",
                        "g2_g2",
                        "g2_g3",
                        "g3_g3",
                    ]
                elif self.nspectra == 14:
                    self.order_data = [
                        "kappa_g0",
                        "kappa_g1",
                        "kappa_g2",
                        "kappa_g3",
                        "g0_g0",
                        "g0_g1",
                        "g0_g2",
                        "g0_g3",
                        "g1_g1",
                        "g1_g2",
                        "g1_g3",
                        "g2_g2",
                        "g2_g3",
                        "g3_g3",
                    ]
                elif self.nspectra == 27:
                    self.order_data = [
                        "kappa_g0",
                        "kappa_g1",
                        "kappa_g2",
                        "kappa_g3",
                        "kappa_g4",
                        "kappa_g5",
                        "g0_g0",
                        "g0_g1",
                        "g0_g2",
                        "g0_g3",
                        "g0_g4",
                        "g0_g5",
                        "g1_g1",
                        "g1_g2",
                        "g1_g3",
                        "g1_g4",
                        "g1_g5",
                        "g2_g2",
                        "g2_g3",
                        "g2_g4",
                        "g2_g5",
                        "g3_g3",
                        "g3_g4",
                        "g3_g5",
                        "g4_g4",
                        "g4_g5",
                        "g5_g5",
                    ]

        elif self.data_type == "mock_dr1":
            if self.order_data == None:
                assert self.nspectra == 91
                # self.order_data = ['kappa_kappa', 'kappa_gbin1', 'kappa_gbin2', 'kappa_gbin3', 'kappa_gbin4', 'kappa_gbin5', 'kappa_gbin6', 'kappa_wlbin1', 'kappa_wlbin2', 'kappa_wlbin3', 'kappa_wlbin4', 'kappa_wlbin5', 'kappa_wlbin6', 'gbin1_gbin1', 'gbin1_gbin2', 'gbin1_gbin3', 'gbin1_gbin4', 'gbin1_gbin5', 'gbin1_gbin6', 'gbin1_wlbin1', 'gbin1_wlbin2', 'gbin1_wlbin3', 'gbin1_wlbin4', 'gbin1_wlbin5', 'gbin1_wlbin6', 'gbin2_gbin2', 'gbin2_gbin3', 'gbin2_gbin4', 'gbin2_gbin5', 'gbin2_gbin6', 'gbin2_wlbin1', 'gbin2_wlbin2', 'gbin2_wlbin3', 'gbin2_wlbin4', 'gbin2_wlbin5', 'gbin2_wlbin6', 'gbin3_gbin3', 'gbin3_gbin4', 'gbin3_gbin5', 'gbin3_gbin6', 'gbin3_wlbin1', 'gbin3_wlbin2', 'gbin3_wlbin3', 'gbin3_wlbin4', 'gbin3_wlbin5', 'gbin3_wlbin6', 'gbin4_gbin4', 'gbin4_gbin5', 'gbin4_gbin6', 'gbin4_wlbin1', 'gbin4_wlbin2', 'gbin4_wlbin3', 'gbin4_wlbin4', 'gbin4_wlbin5', 'gbin4_wlbin6', 'gbin5_gbin5', 'gbin5_gbin6', 'gbin5_wlbin1', 'gbin5_wlbin2', 'gbin5_wlbin3', 'gbin5_wlbin4', 'gbin5_wlbin5', 'gbin5_wlbin6', 'gbin6_gbin6', 'gbin6_wlbin1', 'gbin6_wlbin2', 'gbin6_wlbin3', 'gbin6_wlbin4', 'gbin6_wlbin5', 'gbin6_wlbin6', 'wlbin1_wlbin1', 'wlbin1_wlbin2', 'wlbin1_wlbin3', 'wlbin1_wlbin4', 'wlbin1_wlbin5', 'wlbin1_wlbin6', 'wlbin2_wlbin2', 'wlbin2_wlbin3', 'wlbin2_wlbin4', 'wlbin2_wlbin5', 'wlbin2_wlbin6', 'wlbin3_wlbin3', 'wlbin3_wlbin4', 'wlbin3_wlbin5', 'wlbin3_wlbin6', 'wlbin4_wlbin4', 'wlbin4_wlbin5', 'wlbin4_wlbin6', 'wlbin5_wlbin5', 'wlbin5_wlbin6', 'wlbin6_wlbin6']
                self.order_data = list(data["cls"].keys())

        elif self.data_type == "mock_dr1_demnuni":
            print(f"[LikeCobaya] Using mock_dr1_demnuni data")
            if self.order_data == None:
                self.order_data = list(data["cls"].keys())
        elif self.data_type == "fs2_q1_7bins" or self.data_type == "fs2_q1_7bins_truez":
            print(f"[LikeCobaya] Using FS2 mock Q1 data")
            if self.order_data == None:
                self.order_data = list(data["cls"].keys())
        else:
            assert False, "Data type not recognized"


        # We select a subset of spectra based on the subset data list of cls or tags
        self.order_data_orig = self.order_data.copy()
        if self.subset_data is not None:
            if isinstance(self.subset_data, str):
                self.subset_data = [self.subset_data]
            self.order_data = [o for o in self.order_data if any(sub in o for sub in self.subset_data)]
            if len(self.order_data) == 0:
                raise ValueError(f"No spectra found in the data with the subset {self.subset_data}")
        
        if self.gg_only and self.cross_lensing_only:
            raise ValueError("Cannot use both gg_only and cross_lensing_only at the same time. Please choose one.")
        if self.cross_lensing_only:
            self.order_data = [o for o in self.order_data if "kappa" in o and "kappa_kappa" not in o]
            if len(self.order_data) == 0:
                raise ValueError("No kappa spectrum found in the data")
        elif self.gg_only:
            self.order_data = [o for o in self.order_data if "kappa" not in o]
       

        # We redefine the number of spectra based on the subset of spectra we are using
        self.nspectra = len(self.order_data)
        subset_indices = [self.order_data_orig.index(k) for k in self.order_data]
        
        assert len(subset_indices) == self.nspectra, f"Not all spectra found in data: {self.order_data} not in {data['cls'].keys()}"
        if self.verbose:
            print(f"[LikeCobaya] Using {self.nspectra} spectra: {self.order_data}, with indices {subset_indices} from {data['cls'].keys()}")

        self.clkeys_data2cobaya()

        self.data_spectra = np.array([np.squeeze(data["cls"][o]["cl"]) for o in self.order_data])
        nbps_full = self.data_spectra.shape[1]

        self.Bbl = [data["cls"][o]["Bbl"] for o in self.order_data]
        
        try:
            self.leff = [data["cls"][o]["leff"] for o in self.order_data]
        except KeyError:
            print(f"[LikeCobaya] leff not found in the data, using default values for lmax=3000 and 100 bins")
            _leff = np.array([  51.5,  151.5,  251.5,  351.5,  451.5,  551.5,  651.5,  751.5,
                        851.5,  951.5, 1051.5, 1151.5, 1251.5, 1351.5, 1451.5, 1551.5,
                    1651.5, 1751.5, 1851.5, 1951.5, 2051.5, 2151.5, 2251.5, 2351.5,
                    2451.5, 2551.5, 2651.5, 2751.5, 2851.5])
            
            self.leff = [_leff for o in self.order_data]
        try:
            data['dndz']
        except KeyError:
            if self.data_type == "fs2_q1_7bins":
                print(f"[LikeCobaya] dndz not found in the data, using default values for FS2 dndz: outputs/flagship2/19347.parquet")
                import pandas 
                dndz = pandas.read_parquet(os.path.join(data_dir, 'outputs/flagship2/19347.parquet'))['dndz']
                dndz = dndz.to_numpy()
                z= np.arange(6.01, step=0.01)
                dndz = dndz[1:]
                data['dndz'] = {}
                for i in range(len(dndz)):
                    data['dndz'][f'bin{i}'] = np.array([z, dndz[i]])

            elif self.data_type == "fs2_q1_7bins_truez":
                print(f"[LikeCobaya] dndz not found in the data, using default values for FS2 dndz: outputs/flagship2/dndz_fs2_truez_sgb.pkl")
                _input_file = os.path.join(data_dir, 'outputs/flagship2/dndz_fs2_truez_sgb.pkl')
                dndz_true = np.load(_input_file, allow_pickle=True)
                data['dndz'] = dndz_true['7bin']

        # print(f"[LikeCobaya] dndz is {data['dndz']}")
        if isinstance(self.lmin, int):
            self.lmin = {k: self.lmin for k in self.order_data}
        else:
            assert isinstance(self.lmin, dict), "lmin must be a single value or a dictionary"

        if isinstance(self.lmax, int):
            self.lmax = {k: self.lmax for k in self.order_data}
        else:
            assert isinstance(self.lmax, dict), "lmax must be a single value or a dictionary"

        self.bpmin = np.zeros(self.nspectra, dtype="int")
        self.bpmax = np.zeros(self.nspectra, dtype="int")

        for ispec in range(self.nspectra):
            self.bpmin[ispec] = np.where(self.leff[0] > self.lmin[self.order_data[ispec]])[0][0]
            self.bpmax[ispec] = np.where(self.leff[0] < self.lmax[self.order_data[ispec]])[0][-1]

        self.lmax_theory = max(self.lmax.values())#self.Bbl[0].shape[1]-1

        self.nbps_vec = np.sum(
            self.bpmax - self.bpmin + np.ones(self.nspectra), dtype=int
        )

        if self.do_kappa_tf and self.ccl_include_lensing:
            if self.kappa_tf_path == None:
                tf_default_dict = {
                    "planckPR4_p": "planck_pr4",
                    "actbaseline": "act_baseline",
                }
                self.transfer_function = os.path.join(
                    os.path.dirname(self.abspath),
                    "tf",
                    f"tf_{self.field.lower()}_{tf_default_dict[self.cmb_exp]}.pkl",
                )
            else:
                self.transfer_function = self.kappa_tf_path
        else:
            self.transfer_function = None

        # CMB lensing Monte-Carlo normalisation correction
        if self.transfer_function != None:
            assert os.path.exists(
                self.transfer_function
            ), f"Transfer function file {self.transfer_function} not found, please run the notebooks/tf_calculation.py script to generate it"
            if self.verbose:
                print(
                    f"[LikeCobaya] Loading CMB kappa transfer function from {self.transfer_function}"
                )
            self.tf_lensing = np.load(self.transfer_function, allow_pickle=True)["tf"]
        else:
            if self.verbose:
                print(
                    f"[LikeCobaya] No CMB kappa transfer function provided, using unity"
                )
            self.tf_lensing = np.ones(nbps_full)

        if self.use_cov_theory:
            try:
                covfull = data["cov_theory"]
                if self.verbose:
                    print(
                        f"[LikeCobaya] Using the theory covariance matrix"
                    )
            except KeyError:
                raise KeyError(
                    f"[LikeCobaya] Theory covariance matrix not found in {self.input_file}, but `use_cov_theory` is True. Aborting."
                )
        else:
            covfull = data["cov"]
            if self.verbose:
                print(
                    f"[LikeCobaya] Using the data covariance matrix"
                )
        
        subset_covfull = [[covfull[i][j] for j in subset_indices] for i in subset_indices]

        cov = np.empty((self.nbps_vec, self.nbps_vec))
        posi = 0
        deltai = 0
        for isp in range(self.nspectra):
            posi += deltai
            deltai = self.bpmax[isp] - self.bpmin[isp] + 1
            if "kappa" in self.order_data[isp]:
                currtfi = self.tf_lensing[self.bpmin[isp] : self.bpmax[isp] + 1]
            else:
                currtfi = np.ones(deltai)

            posj = 0
            deltaj = 0
            for jsp in range(self.nspectra):
                posj += deltaj
                deltaj = self.bpmax[jsp] - self.bpmin[jsp] + 1

                if "kappa" in self.order_data[jsp]:
                    currtfj = self.tf_lensing[self.bpmin[jsp] : self.bpmax[jsp] + 1]
                else:
                    currtfj = np.ones(deltaj)

                cov[posi : posi + deltai, posj : posj + deltaj] = subset_covfull[isp][jsp][
                    self.bpmin[isp] : self.bpmax[isp] + 1,
                    self.bpmin[jsp] : self.bpmax[jsp] + 1,
                ] * np.outer(currtfi, currtfj)

        self.icov = np.linalg.inv(cov)

        self.errors = np.empty_like(self.data_spectra)
        for isp in range(self.nspectra):
            self.errors[isp] = np.sqrt(np.diag(subset_covfull[isp][isp]))

        if self.transfer_function != None:
            nmin = min(len(self.tf_lensing), nbps_full)
            for isp in range(self.nspectra):
                if "kappa" in self.order_data[isp]:
                    self.data_spectra[isp, 0:nmin] *= self.tf_lensing[0:nmin]

        self.cov_complete = data["cov_all"]

        self.bps_data_vec = np.empty(self.nbps_vec)
        posi = 0
        deltai = 0
        for isp in range(self.nspectra):
            posi += deltai
            deltai = self.bpmax[isp] - self.bpmin[isp] + 1
            self.bps_data_vec[posi : posi + deltai] = self.data_spectra[
                isp, self.bpmin[isp] : self.bpmax[isp] + 1
            ]

        # Theory initialization

        self.theory = eDR1like.Theory(
            dndz=data["dndz"],
            dndz_marginalization = self.dndz_marginalization,
            dndz_marg_type = self.dndz_marg_type,
            include_lensing=self.ccl_include_lensing,
            include_weaklensing=self.ccl_include_weaklensing,
            theory_provider="ccl",
            theory_params=self.ccl_theory_params,
            extra_params=self.ccl_extra_params,
            include_nl_bias=self.ccl_include_nl_bias,
            higher_order_bias=self.ccl_higher_order_bias,
            higher_order_bias_bs=self.ccl_higher_order_bias_bs,
            include_mag_bias=self.ccl_include_mag_bias,
            include_rsd=self.ccl_include_rsd,
            include_ia_bias=self.ccl_include_ia_bias,
            mag_bias_z=self.ccl_mag_bias_z,
            bias_type=self.bias_type,
            lmax=self.lmax_theory,
            zmin=self.zmin,
            zmax=self.zmax,
            set_bias_cst=self.set_bias_cst,
            dndz_threshold=self.dndz_threshold,
            bias_model= self.bias_model,
        )

        self.dndz = self.theory.dndz
        self.nbin = len(self.dndz)

        self.ndof = 0 # Will be set in initialize_with_provider or in the logp function

        if self.lmax_knl:
            if self.verbose:
                print(
                    "[LikeCobaya] Using kNL scale cuts for the lmax of the bandpowers"
                )
            lmax_knl = np.zeros(self.nbin + 1, dtype=int)
            for i, ibin in enumerate(self.dndz.keys()):
                z_arr = self.dndz[ibin][0][self.theory.izmin : self.theory.izmax[ibin] + 1]
                dndz_arr = self.dndz[ibin][1][self.theory.izmin : self.theory.izmax[ibin] + 1]
                _meanz = self.meanz(z_arr, dndz_arr)
                a = 1/(1 + _meanz)  # scale factor at mean redshift                
                lmax_knl[i] =  round(self.theory.params.kNL(a)*self.theory.params.comoving_radial_distance(a))
            for key in self.order_data:
                b1, b2 = key.split("_")
                if b1[0] == "g" or b1[:3] == "bin":
                    lknl1 = lmax_knl[int(b1[-1:])]
                else:
                    lknl1 = np.inf
                if b2[0] == "g" or b2[:3] == "bin":
                    lknl2 = lmax_knl[int(b2[-1:])]
                else:
                    lknl2 = np.inf
                self.lmax[key] = min(min(self.lmax[key], lknl1), lknl2)

        if self.verbose:
            print("[LikeCobaya] I will use these Cls and scale cuts:")
            for i in range(self.nspectra):
                print(
                    f"  - {self.order_data[i]}: lmin={self.lmin[self.order_data[i]]}, lmax={self.lmax[self.order_data[i]]}"
                )
            print(f"[LikeCobaya] I will use a total of {self.nbps_vec} bandpowers")

            if self.bias_type == "global":
                if self.ccl_include_nl_bias == True:
                    print(
                        "[LikeCobaya] Non linear bias: sampling with two global bias amplitude parameters"
                    )
                else:
                    print(
                        "[LikeCobaya] Linear bias: sampling with one global bias amplitude parameter"
                    )

            elif self.bias_type == "binwise":
                if self.ccl_include_nl_bias == True:
                    print(
                        f"[LikeCobaya] Non linear bias: sampling with binwise bias amplitude parameters, on {self.nbin} redshift bins"
                    )
                else:
                    print(
                        f"[LikeCobaya] Linear bias: sampling with binwise bias amplitude parameters, on {self.nbin} redshift bins"
                    )
            else:
                raise ValueError(
                    f"Bias type {self.bias_type} not recognized, must be 'global' or 'binwise'"
                )
        
            if self.dndz_threshold > 0:
                print(
                    f"[LikeCobaya] dndz threshold: {self.dndz_threshold} (dndz values below this threshold will be set to zero)"
                )
            if self.zmax:
                print(
                    f"[LikeCobaya] zmax: {self.zmax} (dndz values above this redshift will be set to zero)"
                )

            print(f"[LikeCobaya] Using {self.bias_model} bias model for the linear galaxy bias")

    def initialize_with_provider(self, provider):
        self.ndof = self.nbps_vec - provider.model.prior.d()
        if self.verbose:
            print(f"[LikeCobaya] ndof = {self.ndof}")
            print(f"[LikeCobaya] number of sampled params = {provider.model.prior.d()}")
            print(f"[LikeCobaya] sampled params = {provider.model.prior.params}")

    def get_requirements(self):
        return {}

    def get_bps(
        self,
        cl_theory,
        lmax_theory=None,
    ):

        if lmax_theory == None:
            lmax_theory = self.lmax_theory

        theory_spectra = np.empty_like(self.data_spectra)

        for isp, o in enumerate(self.order_theory):
            theory_spectra[isp] = self.Bbl[isp][:, : lmax_theory + 1].dot(
                cl_theory[o][: lmax_theory + 1]
            )

        return theory_spectra

    def _get_bps_vec(
        self,
        cls,
        order,
        lmax,
    ):

        bps_vec = np.empty(self.nbps_vec)
        posi = 0
        deltai = 0
        for isp, o in enumerate(order):
            posi += deltai
            deltai = self.bpmax[isp] - self.bpmin[isp] + 1
            bps_vec[posi : posi + deltai] = self.Bbl[isp][
                self.bpmin[isp] : self.bpmax[isp] + 1, : lmax + 1
            ].dot(cls[o][: lmax + 1])

        return bps_vec

    def clkeys_data2cobaya(self):
        """Translate data Cls keys into cobaya Cls keys

        The order_theory dictionnary contains the Cls keys in the same order as the data dicttionnary

        """

        self.order_theory = []
        for cl_dat_key in self.order_data:
            kp = "G"
            if self.data_type == "q1":
                a, b = cl_dat_key.split("_")
                if a == "kappa":
                    a = "P"
                elif a[0] == "g":
                    a = kp + str(int(a[1]))
                if b == "kappa":
                    b = "P"
                elif b[0] == "g":
                    b = kp + str(int(b[1]))
                self.order_theory.append((a, b))

            elif self.data_type == "mock_dr1_demnuni":
                a, b = cl_dat_key.split("_")
                if a == "kappa":
                    a = "P"
                elif a[0] == "g":
                    a = kp + str(int(a[1]) - 1)
                if b == "kappa":
                    b = "P"
                elif b[0] == "g":
                    b = kp + str(int(b[1]) - 1)
                self.order_theory.append((a, b))

            elif self.data_type == "mock_dr1":
                key_mapping = {
                    "kappa": "P",
                    "gbin1": "W0",
                    "gbin2": "W1",
                    "gbin3": "W2",
                    "gbin4": "W3",
                    "gbin5": "W4",
                    "gbin6": "W5",
                    "wlbin1": "W6",
                    "wlbin2": "W7",
                    "wlbin3": "W8",
                    "wlbin4": "W9",
                    "wlbin5": "W10",
                    "wlbin6": "W11",
                    # 'gbin1': 'W1', 'gbin2': 'W2', 'gbin3': 'W3', 'gbin4': 'W4', 'gbin5': 'W5', 'gbin6': 'W6',
                    # 'wlbin1': 'W7', 'wlbin2': 'W8', 'wlbin3': 'W9', 'wlbin4': 'W10', 'wlbin5': 'W11', 'wlbin6': 'W12'
                }

                self.order_theory = []
                for key in self.order_data:
                    parts = key.split("_")
                    transformed_key = (
                        key_mapping[parts[0]] + "x" + key_mapping[parts[1]]
                    )
                    self.order_theory.append(transformed_key)
            elif self.data_type == "fs2_q1_7bins" or self.data_type == "fs2_q1_7bins_truez":
                a, b = cl_dat_key.split("_")
                a = kp + str(int(a[-1]))
                b = kp + str(int(b[-1]))
                self.order_theory.append((a, b))

            else:
                raise ValueError("Data type not recognized")
        print(f"[LikeCobaya] Using the following order for the theory Cls: {self.order_theory}")
        return

    def get_cls_theory(self, param_values):
        """Compute the Cls for the given parameters."""
        cl_theory = self.theory.get_theory_cls(**param_values)

        return cl_theory

    def logp(self, _derived=None, **params_values):
        """Compute the log-likelihood for the given parameters."""
        cl_theory = self.get_cls_theory(params_values)
        bps_theory_vec = self._get_bps_vec(
            cl_theory, self.order_theory, self.lmax_theory ## TBC
        )

        delta = bps_theory_vec - self.bps_data_vec
        chi2 = delta @ self.icov @ delta

        if _derived is not None:
            _derived["sigma8"] = self.theory.params.sigma8()
            _derived["S8"] = (
                self.theory.params.sigma8()
                * (self.theory.params["Omega_m"] / 0.3) ** 0.5
            )
            
            if self.ndof == 0:
                self.ndof = self.nbps_vec

            _derived["chi2_reduced"] = chi2 / self.ndof
            _derived["PTE"] = 1.0 - stats.chi2.cdf(chi2, self.ndof)

        return -chi2 / 2, _derived
    
    def SNratio(
        self,
        cross_lensing_only=False,
    ):

        if cross_lensing_only:
            assert self.has_lensing
            return np.sqrt(
                self.bps_data_vec[0 : self.nbps_vec_lensing]
                @ self.icov_lensing
                @ self.bps_data_vec[0 : self.nbps_vec_lensing]
            )
        else:
            return np.sqrt(self.bps_data_vec @ self.icov @ self.bps_data_vec)
        
    def SNratio_lowregime(self, cl_theory):
        cl_theory = self.get_cls_theory()
        bps_theory_vec = self._get_bps_vec(
            cl_theory, self.order_theory, self.lmax_theory)

        delta = bps_theory_vec - self.bps_data_vec
        bg = np.dot(bps_theory_vec.T, np.dot(self.icov, self.bps_data_vec)) / np.dot(bps_theory_vec.T, np.dot(self.icov, bps_theory_vec))
        sigma = 1 / np.sqrt(np.dot(bps_theory_vec.T, np.dot(self.icov, bps_theory_vec)))
        snr1 = bg / sigma
        return snr1

    @staticmethod
    def meanz(z, dndz):
        norm = simpson(dndz, z)
        meanz = simpson(z*dndz, z)
        return meanz/norm   
