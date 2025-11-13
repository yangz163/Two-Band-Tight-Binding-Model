#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#========================== Physical Constants in SI Units ===============================#

hbar_SI = 1.054571817e-34 # Reduced Plank's Constant J*s
kB_SI   = 1.380649e-23 # Boltzmann constant J/K
q_e     = 1.602176634e-19 # Elementary charge C
m_e     = 9.1093837015e-31 # Free electron mass kg
Angst   = 1e-10 # Angstrom in m
PI      = np.pi
tau0_s = 1e-14 # 10 fs
spin = 2 # Spin degeneracy

# Conversion helpers
EV_TO_J   = q_e                 # 1 eV = 1.602176634e-19 J
J_TO_EV   = 1.0 / q_e
CM3_TO_M3 = 1e6                 # 1 cm^-3 = 1e6 m^-3

#======================== Tools for Transport Calculations ================================#
def fermi(E, mu, T_K):
    x = np.clip((E - mu) / (kB_SI * T_K), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(x))

def minus_df_dE(E, mu, T_K):
    x = (E - mu) / (kB_SI * T_K)
    exp = np.exp(np.clip(x, -700, 700))
    return exp / (kB_SI * T_K * (1.0 + exp) ** 2)

def get_eigenvalue_grid(band_structure, nkpt, a, b, c):

    # Endpoint=False to keep periodicity)
    kx = np.linspace(-PI / (a * Angst), PI / (a * Angst), nkpt, endpoint=False)
    ky = np.linspace(-PI / (b * Angst), PI / (b * Angst), nkpt, endpoint=False)
    kz = np.linspace(-PI / (c * Angst), PI / (c * Angst), nkpt, endpoint=False)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")

    # Separately extract eigenvalues from VB and CB to facilitate semimetal case
    E_vb = band_structure(KX, KY, KZ, 0).astype(float)
    E_cb = band_structure(KX, KY, KZ, 1).astype(float)
    return {"kx": kx, "ky": ky, "kz": kz, "E_vb": E_vb, "E_cb": E_cb}

def get_velocity_grid(eigs_grid):

    kx, ky, kz = eigs_grid["kx"], eigs_grid["ky"], eigs_grid["kz"]
    E_vb = eigs_grid["E_vb"]
    E_cb = eigs_grid["E_cb"]

    nkpt = kx.size
    if E_vb.shape != (nkpt, nkpt, nkpt) or E_cb.shape != (nkpt, nkpt, nkpt):
        raise ValueError("Energy cubes must be (nkpt, nkpt, nkpt).")
    if nkpt < 3:
        raise ValueError("nkpt must be >= 3 for central differences.")

    dkx = float(kx[1] - kx[0])
    dky = float(ky[1] - ky[0])
    dkz = float(kz[1] - kz[0])

    def grad_periodic(E3, dx, dy, dz):
        dEx = (np.roll(E3, -1, axis=0) - np.roll(E3,  1, axis=0)) / (2.0 * dx)
        dEy = (np.roll(E3, -1, axis=1) - np.roll(E3,  1, axis=1)) / (2.0 * dy)
        dEz = (np.roll(E3, -1, axis=2) - np.roll(E3,  1, axis=2)) / (2.0 * dz)
        return dEx, dEy, dEz

    dEx_vb, dEy_vb, dEz_vb = grad_periodic(E_vb, dkx, dky, dkz)
    dEx_cb, dEy_cb, dEz_cb = grad_periodic(E_cb, dkx, dky, dkz)

    # Separately calculating group velocity from CB and VB to allow the code do calculation when there's energy overlap
    vx_vb = dEx_vb / hbar_SI
    vy_vb = dEy_vb / hbar_SI
    vz_vb = dEz_vb / hbar_SI

    vx_cb = dEx_cb / hbar_SI
    vy_cb = dEy_cb / hbar_SI
    vz_cb = dEz_cb / hbar_SI

    return {
        "vx_vb": vx_vb, "vy_vb": vy_vb, "vz_vb": vz_vb,
        "vx_cb": vx_cb, "vy_cb": vy_cb, "vz_cb": vz_cb,
    }

#============================== Boltzmann Equations ==================================#

# Precalculate the energy dependent part
def precompute_channels_and_DOS(eigs_grid, vel_grid, E_bin, a, b, c,
                                tau0_s, Nv_vb, Nv_cb, mu_J):

    # Establishing the k-grid indexes
    kx = np.asarray(eigs_grid["kx"], dtype=float)
    ky = np.asarray(eigs_grid["ky"], dtype=float)
    kz = np.asarray(eigs_grid["kz"], dtype=float)
    dk_vol = (kx[1] - kx[0]) * (ky[1] - ky[0]) * (kz[1] - kz[0])
    pref = dk_vol / (8.0 * np.pi**3)
    spin = 2.0

    # Cell volume in m^3
    Vcell = a * b * c * (Angst ** 3)

    # Reformat the vectors
    Ev = np.asarray(eigs_grid["E_vb"], dtype=float).ravel(order="C")
    Ec = np.asarray(eigs_grid["E_cb"], dtype=float).ravel(order="C")

    vx_vb = np.asarray(vel_grid["vx_vb"], dtype=float).ravel(order="C")
    vy_vb = np.asarray(vel_grid["vy_vb"], dtype=float).ravel(order="C")
    vz_vb = np.asarray(vel_grid["vz_vb"], dtype=float).ravel(order="C")
    vx_cb = np.asarray(vel_grid["vx_cb"], dtype=float).ravel(order="C")
    vy_cb = np.asarray(vel_grid["vy_cb"], dtype=float).ravel(order="C")
    vz_cb = np.asarray(vel_grid["vz_cb"], dtype=float).ravel(order="C")

    # Define energy grid
    Emin = float(min(Ev.min(), Ec.min()))
    Emax = float(max(Ev.max(), Ec.max()))
    E_bin = int(E_bin)
    edges = np.linspace(Emin, Emax, E_bin + 1, endpoint=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dE = edges[1] - edges[0]

    idx_vb = np.floor((Ev - Emin) / dE).astype(int)
    idx_cb = np.floor((Ec - Emin) / dE).astype(int)
    idx_vb = np.clip(idx_vb, 0, E_bin - 1)
    idx_cb = np.clip(idx_cb, 0, E_bin - 1)

    # DOS per cell
    counts_vb = np.bincount(idx_vb, minlength=E_bin).astype(float)
    counts_cb = np.bincount(idx_cb, minlength=E_bin).astype(float)
    counts_total = Nv_vb * counts_vb + Nv_cb * counts_cb
    DOS_Jm3 = (spin * pref * counts_total) / dE
    DOS_cell = DOS_Jm3 * Vcell  # states / J / cell

    # sigma(E) known as transport distribution function
    sum_vx2_vb = np.bincount(idx_vb, weights=vx_vb * vx_vb, minlength=E_bin)
    sum_vy2_vb = np.bincount(idx_vb, weights=vy_vb * vy_vb, minlength=E_bin)
    sum_vz2_vb = np.bincount(idx_vb, weights=vz_vb * vz_vb, minlength=E_bin)

    sum_vx2_cb = np.bincount(idx_cb, weights=vx_cb * vx_cb, minlength=E_bin)
    sum_vy2_cb = np.bincount(idx_cb, weights=vy_cb * vy_cb, minlength=E_bin)
    sum_vz2_cb = np.bincount(idx_cb, weights=vz_cb * vz_cb, minlength=E_bin)

    sum_vx2 = Nv_vb * sum_vx2_vb + Nv_cb * sum_vx2_cb
    sum_vy2 = Nv_vb * sum_vy2_vb + Nv_cb * sum_vy2_cb
    sum_vz2 = Nv_vb * sum_vz2_vb + Nv_cb * sum_vz2_cb

    scale = (tau0_s * pref) / dE
    channel_xE = scale * sum_vx2
    channel_yE = scale * sum_vy2
    channel_zE = scale * sum_vz2

    # Seebeck distibution function (SDF)
    if mu_J is None:
        W_xE = np.zeros_like(channel_xE)
        W_yE = np.zeros_like(channel_yE)
        W_zE = np.zeros_like(channel_zE)
    else:
        Erel = centers - float(mu_J)
        W_xE = channel_xE * Erel
        W_yE = channel_yE * Erel
        W_zE = channel_zE * Erel

    N_nuclea_auto = spin * float(Nv_vb)

    return {
        "E_centers_J": centers,
        "channel_xE": channel_xE,
        "channel_yE": channel_yE,
        "channel_zE": channel_zE,
        "DOS_cell": DOS_cell,
        "N_nuclea_auto": N_nuclea_auto,
        "W_xE": W_xE,
        "W_yE": W_yE,
        "W_zE": W_zE,
    }

def transport_coefficient(E_grid, channel_xE, channel_yE, channel_zE, mu_J, T_K):

    if E_grid.size < 2:
        raise ValueError("E_grid must have at least 2 points.")

    dE = (E_grid[-1] - E_grid[0]) / (E_grid.size - 1)

    # Thermal weight
    w_fd = minus_df_dE(E_grid, mu_J, T_K) 
    Erel = (E_grid - mu_J)

    def integrate(comp):

        L0 = np.sum(comp * (Erel**0) * w_fd) * dE
        L1 = np.sum(comp * (Erel**1) * w_fd) * dE
        L2 = np.sum(comp * (Erel**2) * w_fd) * dE
        return L0, L1, L2

    L0x, L1x, L2x = integrate(channel_xE)
    L0y, L1y, L2y = integrate(channel_yE)
    L0z, L1z, L2z = integrate(channel_zE)

    return {
        "L0": {"xx": L0x, "yy": L0y, "zz": L0z},
        "L1": {"xx": L1x, "yy": L1y, "zz": L1z},
        "L2": {"xx": L2x, "yy": L2y, "zz": L2z},
    }

def get_mu(N_nuclea, eigs_grid, a, b, c, T_K, E_bin):
    kB_SI = 1.380649e-23  # J/K
    Angst = 1e-10
    Vcell = a * b * c * (Angst**3)


    kx = np.asarray(eigs_grid["kx"], dtype=float)
    ky = np.asarray(eigs_grid["ky"], dtype=float)
    kz = np.asarray(eigs_grid["kz"], dtype=float)
    dk_vol = (kx[1]-kx[0]) * (ky[1]-ky[0]) * (kz[1]-kz[0])
    pref_k = dk_vol / (8.0 * np.pi**3)
    spin = 2.0

    Ev = np.asarray(eigs_grid["E_vb"], dtype=float).ravel(order="C")
    Ec = np.asarray(eigs_grid["E_cb"], dtype=float).ravel(order="C")
    Emin = float(min(Ev.min(), Ec.min()))
    Emax = float(max(Ev.max(), Ec.max()))

    nE = E_bin
    edges = np.linspace(Emin, Emax, nE+1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dE = edges[1] - edges[0]

    Nv_vb = float(eigs_grid.get("Nv_vb", 1.0))
    Nv_cb = float(eigs_grid.get("Nv_cb", 1.0))

    counts_vb, _ = np.histogram(Ev, bins=edges)
    counts_cb, _ = np.histogram(Ec, bins=edges)

    DOS_v_cell = (spin * pref_k * Nv_vb * counts_vb / dE) * Vcell  # states/J/cell
    DOS_c_cell = (spin * pref_k * Nv_cb * counts_cb / dE) * Vcell
    DOS_cell   = DOS_v_cell + DOS_c_cell

    def fermi(E, mu):
        x = (E - mu) / (kB_SI * T_K)
        x = np.clip(x, -700.0, 700.0)
        return 1.0 / (1.0 + np.exp(x))

    def N_of_mu(mu):
        return float(np.sum(DOS_cell * fermi(centers, mu)) * dE)

    threshold = 1e-5  # Numerical tolerance to find intinsic chemical potential
    mu_grid = np.linspace(Emin, Emax, 10001)
    residuals = np.abs([N_nuclea - N_of_mu(mu) for mu in mu_grid])

    if np.min(residuals) < threshold:
        mu_best = mu_grid[np.argmin(residuals)]
        return mu_best

    def R(mu):
        return N_nuclea - N_of_mu(mu)

    lo = Emin + 0.05 * (Emax - Emin)
    hi = Emax - 0.05 * (Emax - Emin)
    Rlo, Rhi = R(lo), R(hi)
    if Rlo * Rhi > 0.0:
        pad = 0.02 * (Emax - Emin)
        for _ in range(40):
            lo -= pad; hi += pad
            Rlo, Rhi = R(lo), R(hi)
            if Rlo * Rhi <= 0.0:
                break

    for _ in range(256):
        mid = 0.5 * (lo + hi)
        Rm = R(mid)
        if abs(Rm) < threshold or (hi - lo) < 1e-15:
            return mid
        if Rlo * Rm <= 0.0:
            hi, Rhi = mid, Rm
        else:
            lo, Rlo = mid, Rm

    return 0.5 * (lo + hi)

def get_carrier_concentration(N_nuclea, dos, mu_J, a, b, c, T_K):

    Vcell = a * b * c * (Angst ** 3)
    E_J = np.asarray(dos["E_grid"], dtype=float)

    if "D_cell" in dos:
        D_cell = np.asarray(dos["D_cell"], dtype=float)
    elif "D_E" in dos:
        D_cell = np.asarray(dos["D_E"], dtype=float) * Vcell
    else:
        raise KeyError("DOS dict must contain 'D_cell' or 'D_E'.")

    if E_J.size < 2:
        raise ValueError("DOS energy grid must have at least 2 points.")
    dE = (E_J[-1] - E_J[0]) / (E_J.size - 1)

    # Number of electrons per cell (this means positive carrier concentration is n-type)
    Ne_per_cell = float(np.sum(D_cell * fermi(E_J, mu_J, T_K)) * dE)

    # carrier concentration (m^-3)
    n_carrier = (Ne_per_cell - N_nuclea) / Vcell
    return n_carrier

def transport_properties(band_structure, nkpt, a, b, c, T_K,
                         tau0_s, nE, n_mu,
                         Nv_vb=1, Nv_cb=1):

    eigs_grid = get_eigenvalue_grid(band_structure, nkpt, a, b, c)
    vel_grid  = get_velocity_grid(eigs_grid)

    pre = precompute_channels_and_DOS(
        eigs_grid=eigs_grid,
        vel_grid=vel_grid,
        E_bin=nE,
        a=a, b=b, c=c,
        tau0_s=tau0_s,
        Nv_vb=Nv_vb,
        Nv_cb=Nv_cb,
        mu_J = None
    )

    E_grid     = np.asarray(pre["E_centers_J"], dtype=float)
    channel_xE = np.asarray(pre["channel_xE"], dtype=float)
    channel_yE = np.asarray(pre["channel_yE"], dtype=float)
    channel_zE = np.asarray(pre["channel_zE"], dtype=float)
    N_nuclea_auto = float(pre["N_nuclea_auto"])

    # mu grid
    E_min, E_max = float(E_grid[0]), float(E_grid[-1])
    if not np.isfinite(E_min) or not np.isfinite(E_max) or E_min >= E_max:
        raise ValueError("Invalid energy span in transport_properties.")
    if nE < 2 or n_mu < 2:
        raise ValueError("nE and n_mu must be >= 2.")
    trim = 0.02 # Move away from the upper and lower bounds to avoid artifical overshoot
    mu_grid_J = np.linspace(E_min + trim*(E_max - E_min),
                            E_max - trim*(E_max - E_min),
                            n_mu, endpoint=True) 

    sigma_x = np.zeros(n_mu); sigma_y = np.zeros(n_mu); sigma_z = np.zeros(n_mu)
    S_x     = np.zeros(n_mu); S_y     = np.zeros(n_mu); S_z     = np.zeros(n_mu)
    PF_x    = np.zeros(n_mu); PF_y    = np.zeros(n_mu); PF_z    = np.zeros(n_mu)
    ke_x    = np.zeros(n_mu); ke_y    = np.zeros(n_mu); ke_z    = np.zeros(n_mu)

    for i, mu in enumerate(mu_grid_J):
        Ls = transport_coefficient(E_grid, channel_xE, channel_yE, channel_zE, mu, T_K)
        L0, L1, L2 = Ls["L0"], Ls["L1"], Ls["L2"]

        # conductivity (S/m)
        sigma_x[i] = (q_e**2) * L0["xx"]
        sigma_y[i] = (q_e**2) * L0["yy"]
        sigma_z[i] = (q_e**2) * L0["zz"]

        # Seebeck (V/K)
        S_x[i] = (L1["xx"] / L0["xx"]) / (-q_e * T_K) if L0["xx"] > 0 else 0.0 # Negative charge because it is electron
        S_y[i] = (L1["yy"] / L0["yy"]) / (-q_e * T_K) if L0["yy"] > 0 else 0.0 # Negative charge because it is electron
        S_z[i] = (L1["zz"] / L0["zz"]) / (-q_e * T_K) if L0["zz"] > 0 else 0.0 # Negative charge because it is electron

        # power factor (W/m/K^2)
        PF_x[i] = (S_x[i]**2) * sigma_x[i]
        PF_y[i] = (S_y[i]**2) * sigma_y[i]
        PF_z[i] = (S_z[i]**2) * sigma_z[i]

        # electronic thermal conductivity (W/m/K)
        ke_x[i] = (L2["xx"] - (L1["xx"]**2)/L0["xx"]) / T_K if L0["xx"] > 0 else 0.0
        ke_y[i] = (L2["yy"] - (L1["yy"]**2)/L0["yy"]) / T_K if L0["yy"] > 0 else 0.0
        ke_z[i] = (L2["zz"] - (L1["zz"]**2)/L0["zz"]) / T_K if L0["zz"] > 0 else 0.0

    return {
        "mu_grid_J": mu_grid_J,
        "sigma_xx": sigma_x, "sigma_yy": sigma_y, "sigma_zz": sigma_z,
        "S_xx": S_x,         "S_yy": S_y,         "S_zz": S_z,
        "PF_xx": PF_x,       "PF_yy": PF_y,       "PF_zz": PF_z,
        "kappa_e_xx": ke_x,  "kappa_e_yy": ke_y,  "kappa_e_zz": ke_z,
        "E_grid": E_grid,
        "channel_xE": channel_xE, "channel_yE": channel_yE, "channel_zE": channel_zE,
        "DOS_cell": pre["DOS_cell"],
        "N_nuclea_auto": N_nuclea_auto,

    }
