#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# constants
PI      = np.pi
Angst   = 1e-10
hbar_SI = 1.054_571_817e-34   # J * s
m_e     = 9.109_383_7015e-31  # kg
q_e     = 1.602_176_634e-19   # J/eV  or C

# Effective mass matrices
def eff_mass(band_index: int, axis_index: int, k_c: float, k_v: float) -> float:
    """Effective mass (kg) for (band, axis)."""
    base = 0.2 * m_e
    if band_index == 0:        # VB
        if axis_index == 0: return base       # x
        if axis_index == 1: return k_v* base  # y
        if axis_index == 2: return base       # z
    elif band_index == 1:      # CB
        if axis_index == 0: return k_c*base  # x
        if axis_index == 1: return base          # y
        if axis_index == 2: return base          # z
    raise ValueError("Invalid (band_index, axis_index).")

# Cell parameters for the hypothetical primitive cell
def cell_para(axis_index: int, a: float, b: float, c: float) -> float:

    if axis_index == 0: return a
    if axis_index == 1: return b
    if axis_index == 2: return c
    raise ValueError("axis_index must be 0, 1, or 2")

def hopping(band_index: int, axis_index: int,
            a: float, b: float, c: float, k_c: float, k_v: float) -> float:

    m_eff = eff_mass(band_index, axis_index, k_c, k_v)
    parameters = cell_para(axis_index, a, b, c) * Angst
    return (hbar_SI**2) / (2.0 * m_eff * parameters **2)

# Constructing the TB model based on user provided parameters and returns band structure
def build_tb(a, b, c, k_c, k_v, Eg_eV):
    """
    Unified dispersion:
        E(kx, ky, kz, band_index)
    band_index = 0 for VB, 1 for CB.

    Enforces:
      E_VB(Gamma) = 0.0
      E_CB(Gamma) = Eg_eV
    """

    # lattice parameters for trig arguments
    A = cell_para(0, a, b, c) * Angst
    B = cell_para(1, a, b, c) * Angst
    C = cell_para(2, a, b, c) * Angst

    # hoppings (eV)
    tvx = hopping(0, 0, a, b, c, k_c, k_v)
    tvy = hopping(0, 1, a, b, c, k_c, k_v)
    tvz = hopping(0, 2, a, b, c, k_c, k_v)
    tcx = hopping(1, 0, a, b, c, k_c, k_v)
    tcy = hopping(1, 1, a, b, c, k_c, k_v)
    tcz = hopping(1, 2, a, b, c, k_c, k_v)

    # raw dispersions (both equal 0 at Gamma analytically)
    def Ev0(kx, ky, kz):
        return -(2.0*tvx*(1.0 - np.cos(kx*A))
                 + 2.0*tvy*(1.0 - np.cos(ky*B))
                 + 2.0*tvz*(1.0 - np.cos(kz*C)))

    def Ec0(kx, ky, kz):
        return +(2.0*tcx*(1.0 - np.cos(kx*A))
                 + 2.0*tcy*(1.0 - np.cos(ky*B))
                 + 2.0*tcz*(1.0 - np.cos(kz*C)))

    # numeric zero-referencing at Gamma (robust to any future edits)
    Ev0_G = Ev0(0.0, 0.0, 0.0)  # should be 0.0
    Ec0_G = Ec0(0.0, 0.0, 0.0)  # should be 0.0

    def E_disp(kx, ky, kz, band_index):
        if band_index == 0:  # VB: set VBM at 0 eV
            return Ev0(kx, ky, kz) - Ev0_G
        elif band_index == 1:  # CB: set CBM at Eg_eV
            return Ec0(kx, ky, kz) - Ec0_G + Eg_eV * q_e
        else:
            raise ValueError("band_index must be 0 (VB) or 1 (CB)")


    return np.vectorize(E_disp)
