![](_page_0_Picture_0.jpeg)

#### LE3 VMPZ-ID: current DR1 work and results on RR2

Jérôme Odier, Gaël Alguero, Fabio Brambilla, Loux Mirouze, Martín Rodriguez-Monroy, Elena Zucca, Juan Macías-Pérez

LPSC Grenoble

![](_page_0_Picture_4.jpeg)

## Plan

![](_page_1_Picture_2.jpeg)

- [1. Overview](#page-2-0)
- [2. Photometric masks and CatRed catalogues](#page-5-0)
- [2. Decontamination Analysis](#page-11-0)
- [3. RR2 decontamination analysis](#page-16-0)
- [5. DR1 status](#page-27-0)

#### <span id="page-2-0"></span>[1. Overview](#page-2-0)

# LE3 VMPZ-ID (in a nutshell) 1/2

![](_page_3_Picture_2.jpeg)

- LE3 VMPZ-ID aims to characterize spatial variations of survey properties, instrumental performance and sky properties (≡ systematics) and their impact on galaxy detection.
- Main products:
  - HEALPix maps (aka masks) of systematics,
  - CatRed catalogs,
  - SOM visibility mask
  - Random catalogs from effective coverage and visibility mask
  - ISD weights ?
- Main GitLab repo:

[https://gitlab.euclid-sgs.uk/PF-LE3-ID/LE3\\_VMPZ\\_ID/](https://gitlab.euclid-sgs.uk/PF-LE3-ID/LE3_VMPZ_ID/)

• Software User Manual (SUM):

[https://gitlab.euclid-sgs.uk/PF-LE3-ID/LE3\\_ID\\_IAL\\_Pipelines/-/](https://gitlab.euclid-sgs.uk/PF-LE3-ID/LE3_ID_IAL_Pipelines/-/blob/develop/LE3_ID_VMPZ_Pipeline/doc/) [blob/develop/LE3\\_ID\\_VMPZ\\_Pipeline/doc/](https://gitlab.euclid-sgs.uk/PF-LE3-ID/LE3_ID_IAL_Pipelines/-/blob/develop/LE3_ID_VMPZ_Pipeline/doc/)

#### LE3 VMPZ-ID DR1 pipeline

![](_page_4_Picture_2.jpeg)

Five main pipelines:

Mosaic and stars, CL and WL CatRed galaxy density, Concatenation of products

![](_page_4_Figure_5.jpeg)

4/24

#### <span id="page-5-0"></span>[2. Photometric masks and](#page-5-0) [CatRed catalogues](#page-5-0)

#### VMPZ-ID HEALPix masks

![](_page_6_Picture_2.jpeg)

- Produced by the Mosaic and star pipeline at the tile level.
- Main masks
  - Survey Masks
    - Solar Aspect Angle (SAA), Alpha Angle (AA), Beta Angle (BA)
    - Exposure
  - Per band mosaic masks
    - FootPrint, Depth, BitMask, Coverage
    - Noise, PSF,
    - Effective Coverage combining all frequency bands and improving star masking
  - Sky masks
    - Zodiacal Light, Galactic Extinction
    - Star Brightness, Std, Mean,
- Concatenated by regions (need to be defined) with CONCAT pipeline
- Data available on DPS and CC-IN2P3 VMPZ-ID area (contact us by slack if you need help)

#### Satellite position, exposure and coverage masks

![](_page_7_Figure_1.jpeg)

#### Depth, Noise, PSF masks

![](_page_8_Picture_1.jpeg)

![](_page_8_Figure_2.jpeg)

![](_page_8_Figure_3.jpeg)

![](_page_8_Figure_4.jpeg)

![](_page_8_Figure_5.jpeg)

![](_page_8_Figure_6.jpeg)

![](_page_8_Figure_7.jpeg)

#### Sky masks

![](_page_9_Picture_2.jpeg)

![](_page_9_Figure_3.jpeg)

![](_page_9_Figure_4.jpeg)

![](_page_9_Figure_5.jpeg)

![](_page_9_Figure_6.jpeg)

#### CatRed catalogs

![](_page_10_Picture_2.jpeg)

- The Catalog Reduction PF is responsible for generating the galaxy catalogs by crossing MER, PHZ, SHE and SPE data.
- Main LE3 consumers:
  - Weak Lensing: DpdLE3clFullInputCat,
  - Clusters of Galaxies: DpdWLPosCatalog, DpdWLShearCatalog, DpdWLProxyShearCatalog, DpdLE3clCombInputCat
- Produced by the so-called "galaxy density pipelines"
- For RR2 only DpdLE3clFullInputCat (cluster catalog) and DpdWLPosCatalog (2-pcf position catalogue) have been produced
- Notice that DpdWLPosCatalog is concatenated but very reduced in information
- CatRed has been bypassed by RR2 CosmoHub catalogs
- WL CatRed pipeline arrives very late in processing after PHZ and SHR calibration

#### <span id="page-11-0"></span>[2. Decontamination Analysis](#page-11-0)

#### Systematics on galaxy density

![](_page_12_Picture_2.jpeg)

• Spatial variations of survey and sky properties as well as of instrumental performance will imprint on the measured galaxy density

![](_page_12_Picture_4.jpeg)

⇒ VMPZ-ID is expected to construct a visibility mask and associated randoms from survey, instrument and sky properties

# Decontamination methods considered

- Linear multiplicative methods in DES
  - Iterative Sytematics Decontamination (ISD): Elvin-Poole et al. [\[arXiv:1708.01536\]](https://arxiv.org/abs/1708.01536), Rodriguez-Monroy et al. [\[arXiv:2105.13540\]](https://arxiv.org/abs/2105.13540)
  - Elastic Net Regularisation (ENet): Weaverdyck, Huterer [\[arXiv:2007.14499\]](https://arxiv.org/abs/2007.14499)
- Linear multiplicative and additive models
  - Theoretical framework: Weaverdyck, Huterer [\[arXiv:2007.14499\]](https://arxiv.org/abs/2007.14499)
  - Joint methods: Hernandez-Monteagudo [\[arXiv:2412.14827](https://ui.adsabs.harvard.edu/abs/2025OJAp....8E..93H/abstract) ]
- Non-linear method in KIDS
  - Organised randoms from Self-Organizing Maps (SOM): Johnston et al. [\[arXiv:2012.08467\]](https://arxiv.org/abs/2012.08467)

Parallel implementation of the ISD (optimize 1D correlations) and SOM (batch SOM in CPU and GPU) algorithms for Euclid.

#### Current decontamination strategy

![](_page_14_Picture_2.jpeg)

- Run ISD on all available templates (raw systematic maps or combinations)
- Select most significant templates from the ISD run
- Produce contaminated and un-contaminated simulations from ISD weights
- Optimize template selection for the SOM on simulations
- Apply SOM to the data using optimized template selection
- Contaminate simulations from SOM results
- Optimize ISD algorithm on those simulations

#### Validation on DES-Y3 simulations

![](_page_15_Picture_1.jpeg)

- Test our implementation of ISD and SOM algorithms
- Validate template selection strategy (Mirouze+2025 in preparation)

![](_page_15_Figure_4.jpeg)

![](_page_15_Figure_5.jpeg)

#### <span id="page-16-0"></span>[3. RR2 decontamination](#page-16-0) [analysis](#page-16-0)

#### Masks and catalog selection

- Consider only south RR2 region (RR2S)
- Use 32 systematic templates (VMPZ-ID masks)
- Work with HEALPix NSIDE = 4096
- POS\_TOM\_BIN\_IDs from suggested selection in CosmoHub
- A cut on effective coverage of 0.8 is always applied

![](_page_17_Figure_8.jpeg)

#### Iterative template selection

![](_page_18_Picture_2.jpeg)

- Use ISD algorithm to check which templates are significant
- Consider only raw templates (PCA components to be investigated)
- Need to set limits on template extreme values
- Use Gaussian/lognormal galaxy number density simulations for uncertainties on 1D correlation

![](_page_18_Figure_7.jpeg)

![](_page_18_Figure_8.jpeg)

#### ISD weight maps

![](_page_19_Picture_2.jpeg)

- We can compute weight maps to be used in the 2-pcf
- Should we make official products?

![](_page_19_Figure_5.jpeg)

![](_page_19_Figure_6.jpeg)

#### SOM visibility mask for bin 1

![](_page_20_Picture_2.jpeg)

- We use 10 ISD more significant templates
- Hierarchical clustering for computing visibility mask

![](_page_20_Figure_5.jpeg)

![](_page_20_Figure_6.jpeg)

## SOM 1D correlations in bin 1

![](_page_21_Picture_2.jpeg)

• This can be used to contaminate simulations. Work in progress

![](_page_21_Figure_4.jpeg)

![](_page_21_Figure_5.jpeg)

![](_page_21_Figure_6.jpeg)

![](_page_21_Figure_7.jpeg)

![](_page_21_Figure_8.jpeg)

#### Randoms for bin 1

![](_page_22_Picture_2.jpeg)

• Compute randoms from data distritrution and SOM

![](_page_22_Figure_4.jpeg)

![](_page_22_Figure_5.jpeg)

#### 2-pcf as a function of selection

![](_page_23_Picture_2.jpeg)

![](_page_23_Figure_3.jpeg)

#### 2-pcf results - VMPZ-ID TreeCorr

![](_page_24_Picture_2.jpeg)

![](_page_24_Figure_3.jpeg)

#### 2-pcf results - L3 pipeline

![](_page_25_Picture_2.jpeg)

![](_page_25_Figure_3.jpeg)

#### Conclusion on RR2S

![](_page_26_Picture_2.jpeg)

- VMPZ-ID pipelines have been run in production mode by system team on RR2: Mosaic, CatRed, galaxy number density
- WL CatRed pipeline was launched on test mode
- Decontamination pipeline has been launched manually on RR2S, but validation procedure is not yet optimal
- RR2S VMPZ-ID products, algorithms and results are available on DPS and/or CosmoHub. Feedback welcome.
- Further work expected for investigating galaxy selection in collaboration with GC-8

#### <span id="page-27-0"></span>5. DR1 status

#### DR1 pipeline current status

![](_page_28_Picture_2.jpeg)

- DR1 VMPZ-ID processing has been started: Mosaic and star masks are already available for WIDE and DEEPs
- Masking is going to be improved for cluster processing: bright stars not included in the survey, diffuse objects, stars-galaxies separation - We can do similarly for WL
- WL CatRed pipeline can not be launched until PHZ and SHEAR calibration are performed - this might induce delays on SGS WL products
- Work needed on optimization and validation on TR1
  - galaxy selection criteria to be tested: interaction with KP-GC8
  - Position simulations are needed: interaction with KP-GC8
  - Blind (or nearly) template selection criteria to be investigated
  - Null-like tests to be defined to avoid over-correction: dr1-kp-wl-le3-1
- Acknowledgements: Hervé Aussel, Nicolas Tessore, Martin Kuemmel, Dominique Bagot, Pierre Casenove, Marine Ruffenach, Sam Farrens, C. Benoists

# Thanks for your attention!

![](_page_29_Picture_1.jpeg)