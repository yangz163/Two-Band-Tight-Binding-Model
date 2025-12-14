#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ---------------------------------------------------------------------
# USER-DEFINED PARAMETERS
a = 5.0            # In Angst
b = 6.0            # In Angst
c = 7.0            # In Angst
k_c = 3            # CB anisotropy factor
k_v = 3            # VB anisotropy factor
Eg_eV = 0.1        # band gap in eV
nkpt = 150         # k-mesh density per axis
T_K = 300.0        # temperature (K)
Nv_vb = 1          # VB degeneracy
Nv_cb = 1          # CB degeneracy
nE   = 10001       # number of energy bins for DOS
n_mu = 30001       # number of mu steps for BTE calculations
tau0_s     = 1e-14 # For constant relaxation approximation
outdir = "./"
# ---------------------------------------------------------------------

# Conversion helper
q_e     = 1.602176634e-19  # J/eV
J_TO_EV = 1.0 / q_e
Angst   = 1e-10

# --- path setup ---
HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import TB_model as TB
import BTE


# ----------------------------- Logging --------------------------------
def setup_logging(log_path):
    logger = logging.getLogger("get_result")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s",
                            datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Working dir: %s", os.getcwd())
    return logger
# ----------------------------------------------------------------------


def main():
    os.makedirs(outdir, exist_ok=True)
    log_file = os.path.join(outdir, "get_result.log")
    logger = setup_logging(log_file)
    t0 = time.perf_counter()

    # Print calculation parameters
    logger.info("=========================== User Defined Parameters ===========================")
    logger.info(" ")
    logger.info(f"Cell parameters: a, b, c = {a}, {b}, {c} Angst")
    logger.info(f"Eg: {Eg_eV} eV")
    logger.info(f"Band anisotropy factors: {k_v} for VB and {k_c} for CB")
    logger.info(f"Band degeneracy: {Nv_vb} for VB and {Nv_cb} for CB")
    logger.info(f"Temperature: {T_K} K")
    logger.info(f"tau: {tau0_s} s")
    logger.info(f"B.Z. sampled on {nkpt}x{nkpt}x{nkpt} grid")
    logger.info(" ")
    logger.info("===============================================================================")
    logger.info(" ")
    logger.info(" ")
    logger.info("============================== Calculation Logs ===============================") 
    logger.info(" ")  
    logger.info("Building tight-binding band structure...")

    # Generating band structure
    band_structure = TB.build_tb(
        a, b, c, 
        k_c, k_v, Eg_eV
    )
    logger.info("Band structure ready.")
    logger.info("Extracting eigenvalues and calculating transport distribution functions ...")

    # Extracting eigenvalues and calculate group velocities
    eigs_grid = BTE.get_eigenvalue_grid(band_structure, nkpt, a, b, c)
    logger.info("Done !")
    logger.info("Solving Boltzmann Transport Equations ...")

    # Doing transport properties calculation
    tp = BTE.transport_properties(
        band_structure, nkpt, a, b, c, 
        T_K, tau0_s=tau0_s, nE=nE, n_mu=n_mu,
        Nv_vb=Nv_vb, Nv_cb=Nv_cb
    )
    eigs_grid["Nv_vb"] = float(Nv_vb)
    eigs_grid["Nv_cb"] = float(Nv_cb)
    eigs_grid["_E_grid_J"]      = np.asarray(tp["E_grid"], dtype=float)
    eigs_grid["_DOS_cell"]      = np.asarray(tp["DOS_cell"], dtype=float)
    eigs_grid["_N_nuclea_auto"] = float(tp["N_nuclea_auto"])

    # Save DOS as array for latter use
    E_grid_J   = np.asarray(tp["E_grid"], dtype=float)
    DOS_cell_J = np.asarray(tp["DOS_cell"], dtype=float)
    dE_J = (E_grid_J[-1] - E_grid_J[0]) / (E_grid_J.size - 1)
    logger.info("Energy grid: %d points, span = [%.4e, %.4e] eV, dE = %.4e meV",
                E_grid_J.size, E_grid_J[0] / q_e, E_grid_J[-1] / q_e, 1e3 * dE_J / q_e)

    dos_dict = {
        "E_grid": E_grid_J,        # Joules
        "D_cell": DOS_cell_J,      # states / J / cell
        "dE": dE_J                  # handy spacing
    }

    # Determine Fermi level via charge neutrality
    N_nuclea_auto = float(tp["N_nuclea_auto"])   # Fully fill VB at 0K
    logger.info(f"Nuclear charge (full VB at T=0): {N_nuclea_auto} per cell")
    logger.info("Solving for intrinsic chemical potential (charge neutrality) using DOS...")
    mu0_J = BTE.get_mu(N_nuclea_auto, eigs_grid, a, b, c, T_K, nE)
    logger.info("Intrinsic mu: %.6f eV", mu0_J / q_e)

    # Printing transport properties
    logger.info("Assembling transport.csv rows (n_mu = %d)...", n_mu)
    mu_grid_J = np.asarray(tp["mu_grid_J"], dtype=float)

    Sx   = np.asarray(tp["S_xx"], dtype=float)
    Sy   = np.asarray(tp["S_yy"], dtype=float)
    Sz   = np.asarray(tp["S_zz"], dtype=float)

    sigx = np.asarray(tp["sigma_xx"], dtype=float)
    sigy = np.asarray(tp["sigma_yy"], dtype=float)
    sigz = np.asarray(tp["sigma_zz"], dtype=float)

    PFx  = np.asarray(tp["PF_xx"], dtype=float)
    PFy  = np.asarray(tp["PF_yy"], dtype=float)
    PFz  = np.asarray(tp["PF_zz"], dtype=float)

    kex  = np.asarray(tp["kappa_e_xx"], dtype=float)
    key  = np.asarray(tp["kappa_e_yy"], dtype=float)
    kez  = np.asarray(tp["kappa_e_zz"], dtype=float)

    rows = []

    for i, mu in enumerate(mu_grid_J):
        # carrier from DOS at this mu
        n_m3 = BTE.get_carrier_concentration(N_nuclea_auto, dos_dict, mu, a, b, c, T_K)

        E_rel_eV = (mu - mu0_J) * J_TO_EV

        Zx = (PFx[i] * T_K / (kex[i]+3)) if kex[i] > 0 else 0.0
        Zy = (PFy[i] * T_K / (key[i]+3)) if key[i] > 0 else 0.0
        Zz = (PFz[i] * T_K / (kez[i]+3)) if kez[i] > 0 else 0.0

        rows.append({
            "E_minus_mu_eV": E_rel_eV,
            "carrier_concentration_cm-3": n_m3 * 1e-6,
            "S_xx_uV_per_K": Sx[i] * 1e6,
            "S_yy_uV_per_K": Sy[i] * 1e6,
            "S_zz_uV_per_K": Sz[i] * 1e6,
            "sigma_xx_S_per_m": sigx[i],
            "sigma_yy_S_per_m": sigy[i],
            "sigma_zz_S_per_m": sigz[i],
            "PF_xx_uWcm-1K-2": PFx[i] * 1e-4,
            "PF_yy_uWcm-1K-2": PFy[i] * 1e-4,
            "PF_zz_uWcm-1K-2": PFz[i] * 1e-4,
            "ke_xx_Wm-1K-1": kex[i],
            "ke_yy_Wm-1K-1": key[i],
            "ke_zz_Wm-1K-1": kez[i],
            "ZT_xx": Zx,
            "ZT_yy": Zy,
            "ZT_zz": Zz
        })

    transport_csv = os.path.join(outdir, "transport.csv")
    pd.DataFrame(rows).to_csv(transport_csv, index=False)

    logger.info("Writing DOS.csv referenced to intrinsic mu...")
    E_rel_eV = (E_grid_J - mu0_J) * J_TO_EV
    DOS_per_eV_cell = DOS_cell_J * q_e  # states/eV/cell

    dos_csv = os.path.join(outdir, "DOS.csv")
    pd.DataFrame({
        "E_minus_mu_eV": E_rel_eV,
        "DOS_per_eV_per_cell": DOS_per_eV_cell
    }).to_csv(dos_csv, index=False)
    logger.info("Wrote DOS: %s  (points=%d)", dos_csv, E_rel_eV.size)

    # Get 3D band structure
    logger.info("Rendering bands.png @ kz = 0 ...")
    nk = nkpt
    kx = np.linspace(-np.pi / (a * Angst), np.pi / (a * Angst), nk, endpoint=False)
    ky = np.linspace(-np.pi / (b * Angst), np.pi / (b * Angst), nk, endpoint=False)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    kz0 = 0.0
    Ev = band_structure(KX, KY, kz0, 0) * J_TO_EV - (mu0_J * J_TO_EV)
    Ec = band_structure(KX, KY, kz0, 1) * J_TO_EV - (mu0_J * J_TO_EV)

    fig = plt.figure(figsize=(8, 6), dpi=600)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(KX, KY, Ev, alpha=0.85, linewidth=0, antialiased=True)
    ax.plot_surface(KX, KY, Ec, alpha=0.85, linewidth=0, antialiased=True)
    ax.set_xlabel(r"$k_x$ (1/Angst)")
    ax.set_ylabel(r"$k_y$ (1/Angst)")
    ax.set_zlabel(r"$E - \mu$ (eV)")
    ax.set_title("TB bands (kz=0)")
    plt.tight_layout()
    band_png = os.path.join(outdir, "bands.png")
    fig.savefig(band_png, bbox_inches="tight", dpi=600)
    plt.close(fig)
    logger.info("Wrote bands: %s", band_png)

    # Plotting 2D band structure X to GAMMA to Y
    npath = 301  # total number of k-points along the k-paths
    kX = np.array([ np.pi/(a*Angst), 0.0, 0.0 ])
    kG = np.array([ 0.0,            0.0, 0.0 ])
    kY = np.array([ 0.0,  np.pi/(b*Angst), 0.0 ])

    n1 = npath // 2
    n2 = npath - n1
    seg1 = np.linspace(kX, kG, n1, endpoint=False) # X to GAMMA 
    seg2 = np.linspace(kG, kY, n2, endpoint=True) # GAMMA to Y
    kpts = np.vstack([seg1, seg2]) # Stack two path together                  
    dk = np.diff(kpts, axis=0)
    step = np.linalg.norm(dk, axis=1)
    s = np.concatenate([[0.0], np.cumsum(step)])
    # Unit conversion
    Ev_J = band_structure(kpts[:,0], kpts[:,1], kpts[:,2], 0)
    Ec_J = band_structure(kpts[:,0], kpts[:,1], kpts[:,2], 1)
    Ev_eV = Ev_J * J_TO_EV - (mu0_J * J_TO_EV)
    Ec_eV = Ec_J * J_TO_EV - (mu0_J * J_TO_EV)
    label_idx = {"X": 0, "Gamma": n1, "Y": n1 + n2 - 1}

    band2d_csv = os.path.join(outdir, "bands_2D.csv")
    pd.DataFrame({
        "s_1_per_A": s,
        "kx_1_per_A": kpts[:,0],
        "ky_1_per_A": kpts[:,1],
        "kz_1_per_A": kpts[:,2],
        "E_v_eV": Ev_eV,
        "E_c_eV": Ec_eV,
        "X_index": [label_idx["X"]]*len(s),
        "Gamma_index": [label_idx["Gamma"]]*len(s),
        "Y_index": [label_idx["Y"]]*len(s),
    }).to_csv(band2d_csv, index=False)


    # Conduction channel term, calculated under 300K at Fermi level
    channel_xE = np.asarray(tp["channel_xE"], dtype=float)
    channel_yE = np.asarray(tp["channel_yE"], dtype=float)
    channel_zE = np.asarray(tp["channel_zE"], dtype=float)
    dfdE = BTE.minus_df_dE(E_grid_J, mu0_J, T_K)

    # Seebeck distribution function (SDF), E - Ef as x-axis
    L_xE = q_e * channel_xE * (E_grid_J - mu0_J) * dfdE
    L_yE = q_e * channel_yE * (E_grid_J - mu0_J) * dfdE
    L_zE = q_e * channel_zE * (E_grid_J - mu0_J) * dfdE

    wf_csv = os.path.join(outdir, "SDF.csv")
    pd.DataFrame({
        "E_abs (EV)": E_grid_J / q_e, # Absolute value
        "E_minus_mu_eV": E_rel_eV, # E - Ef
        "W_x_Js_per_m":  L_xE,             
        "W_y_Js_per_m":  L_yE,
        "W_z_Js_per_m":  L_zE,
    }).to_csv(wf_csv, index=False)
    logger.info(" ")
    logger.info("===============================================================================")
    # 10) Final summary
    logger.info(" ")
    elapsed = time.perf_counter() - t0
    logger.info("Done. Total runtime: %.2f s", elapsed)

if __name__ == "__main__":
    main()


