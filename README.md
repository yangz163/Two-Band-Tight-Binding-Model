Two-Band Tight-Binding Model + Bolztmann Transport Equation Solver

This code script provides a Python framework for the analysis of axis-dependent conduction polarity (ADCP)
and axis-resolved thermoelectric properties using a simplified two-band tight-binding model. The thermoelectric
properties were calculated under Boltzmann transport pictures with constant relaxation time approximation.

**1. Overview**

(1) **TB_model.py**: Establish a tunnable two-band model to study the impact of band gap and band anisotorpy to ADCP.

(2) **BTE.py**: Provide functions to read band structure from TB_model.py and calculate thermoelectric properties under
Boltzmann Transport formalism.

(3) **get_results.py**: The main execution script, containing control panels and file generation functions.

**2. Simulation Pipline**

(1) Build the band structure based on user defined parameters.\
(2) Sampling the Brillouin zone to extract eigenvalues.\
(3) Percompute transport distribution function and density of states (DOS) to improve efficiency during Fermi integrals.\
(4) Solving Boltzmann transport equations to obtain thermoelectric properties at each chemical potential.\
(5) Determine intrinsic chemical potential and the carrier concentations corresponding to each chemical potentials based on
the charge neutrality.\
(6) Generate results in .csv files.

**3. Usage**

Setting up all parameters in the control panel block of get_results.py and run:\
python get_result.py

**4.Key Parameters**

**a, b, c**:          Cell parameters in Angst\
**k_c, k_v**:         Band anisotropic factors for conduction band and valence band\
**Eg_eV**:            Band gap in eV\
**nkpt**:             _k_-mesh density per axis\
**T_K**:              Absolute temperature\
**nE**:               Number of energy bins for DOS calculation\
**n_mu**:             Number of chemical potential bins for transport properties calculation\
**tau0_s**:           Constant carrier relaxation time, set to 10 fs\
**nv_vb, nv_cb**:     Band degeneracy for valence band and conduction band\

**5.Output Files**

**tansport.csv**:                    Axis-resolved themoelectric properties\
**DOS.csv**:                         density of states\
**band_2D.csv**:                     Band structuer along X -> GAMMA -> Y\
**bands.png**:                       3D band stucture within XY-plane\
**weighting_function.csv**:          Axis-resolved Seebeck distribution function (SDF)\
**get_result.py**:                   Log of the calculations
