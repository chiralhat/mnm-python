# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 17:19:07 2016

@author: ccollett

This provides the functions necessary to calculate relaxation rates in
the single molecule magnet Mn12, as described in
https://doi.org/10.1103/PhysRevLett.110.087205
"""

import qutip as qt
import numpy as np
import scipy.constants as sc

"""This section defines the Hamiltonian parameters for Mn12"""

S = 10
s = S
dd, b40, b44, b60, b64 = np.array(
    [-0.459, -2.34e-5, 2e-5, -1e-8, -1e-7]) * 1.44
gpara = 2
gperp = 1.93
d = dd - (30 * s * (s + 1) - 25) * b40
ddd = d + (105 * s**2 * (s + 1)**2 - 525 * s * (s + 1) + 294) * b60

b = 35 * b40
c = -b44 / 2
theta = 90 / 180 * np.pi
phi = 0 / 180 * np.pi
g = 2
Mb = 5.788e-9
Kb = 8.617e-5

"""Here we define the necessary spin operators"""

sx, sy, sz = qt.jmat(S)
Sx, Sy, Sz = sx, sy, sz
sp = qt.jmat(S, '+')
sm = qt.jmat(S, '-')
spz = sp * sz
szp = sz * sp
smz = sm * sz
szm = sz * sm
sp2 = sp**2
sm2 = sm**2


p = 1.356e3  # Mass density
Cs = 1122  # Transverse speed of sound
g02 = 1.21  # Spin-phonon coupling strength; called kappa^(2) in the paper

"""Phonon transition rate prefactors"""
ph1 = (ddd**2 * sc.e * Kb**5) / (24 * np.pi * p * Cs**5 * (sc.hbar / sc.e)**4)
ph2 = (g02 * ddd**2 * sc.e * Kb**5) / \
    (32 * np.pi * p * Cs**5 * (sc.hbar / sc.e)**4)

"""Creating the necessary Stevens operators for the Hamiltonian"""
c_matrix = sp**4 + sm**4
e_matrix = sx**2 - sy**2
b60_matrix = 231 * Sz**6 - (315 * s * (s + 1) - 735) * Sz**4 + \
    (105 * s**2 * (s + 1)**2 - 525 * s * (s + 1) + 294) * Sz**2
b64_matrix = 1 / 4 * ((11 * Sz**2 - (s * (s + 1) + 38)) *
                      c_matrix + c_matrix * (11 * Sz**2 - (s * (s + 1) + 38)))

def h_broaden(H, Hwid, nloop=50):
    """Produces a Gaussian field distribution around H with width Hwid and size
    nloop."""

    Hbroad = np.linspace(H - 2 * Hwid, H + 2 * Hwid, nloop)
    Hweights = np.exp(-np.linspace(-2 * Hwid, 2 * Hwid, nloop) ** 2 /
                      (2 * Hwid**2)) / np.sqrt(Hwid * 2 * np.pi)
    return [Hbroad, Hweights]


def ham_0(H, phi=0, Hz=-400):
    """Assembles the Hamiltonian for a given field, where H is the transverse
    field, phi is the angle in the hard plane, and Hz is the longitudinal field
    """
    H_diag = d * Sz**2 + b * Sz**4 + b60 * b60_matrix
    H_off = b64 * b64_matrix - c * c_matrix
    H_perp = gperp * (Mb / Kb) * H * (Sx * np.cos(phi) + Sy * np.sin(phi))
    H_field = gpara * (Mb / Kb) * Hz * Sz + H_perp
    return H_diag + H_off - H_field


def estate(H, phi=0, Hz=-400):
    """Finds the eigenenergies and eigenstates of the Hamiltonian at a given
    field, where H is the transverse field, phi is the angle in the hard plane,
    and Hz is the longitudinal field.
    """

    return ham_0(H, phi, Hz).eigenstates()


def s1(states):
    """Calculates the matrix elements between states S for {Sx,Sz}, and returns
    an array over all states."""

    out = np.abs(np.array([(m**2)[0] for m in (spz + szp).transform(states)]) +
                 np.array([(m**2)[0] for m in (smz + szm).transform(states)]))
    return out


def s2(states):
    """Calculates the matrix elements between states S for Sx**2-Sy**2,
    and return an array over all states."""

    return np.abs(np.array([(m**2)[0] for m in sm2.transform(states)]) +
                  np.array([(m**2)[0] for m in sp2.transform(states)]))


def boltz(E, T):
    """Calculates the boltzmann factor for a set of energies E and temp T."""

    return np.array([[np.exp((E[k] - E[i]) / T)
                      for k in np.arange(21)] for i in np.arange(21)])


def rate(H, T, ph1, ph2, phi=0, Hz=-400):
    """Calculates the rate matrix for Mn12 given a transverse field H, a temp
    T, prefactors ph1 and ph2, hard plane angle phi, and longitudinal field Hz.
    """

    energies, states = estate(H, phi, Hz)
    b_t = boltz(energies, T)
    s1element = s1(states)
    s2element = s2(states)

    def in_el(i, k):
        e1 = ph1 * s1element[i, k] * ((energies[k] - energies[i])**3)
        e2 = ph2 * s2element[i, k] * ((energies[k] - energies[i])**3)
        b = (1 / (b_t[i, k] - 1))
        return ((e1 * b) + (e2 * b))
    out = [[np.sum([
            0 if k == i else -in_el(i, k) for k in np.arange(21)]) if i == j
            else in_el(i, j) for j in np.arange(21)] for i in np.arange(21)]
    return qt.Qobj(np.array(out).transpose())


def rate_map(H, phi=0, Hz=-400, T=3.21):
    """Finds the slowest nonzero eigenvalue of the rate matrix for a given
    transverse field H, hard plane angle phi, and longitudinal field Hz."""

    return np.sort(np.abs(rate(H, T, ph1, ph2, phi, Hz).eigenenergies()))[1]


def rate_broad(H, Hwid, mod=1, phi=0, Hz=-400, nloop=50):
    """Does the same calculation as rate_map, but including the application of
    field broadening."""

    if isinstance(H, int) or isinstance(H, np.int32):
        Hbroad, Hweights = h_broaden(H, Hwid, nloop)
        return mod * \
            np.sum(Hweights * np.array([rate_map(H, phi, Hz) for H in Hbroad]))
    else:
        si = len(H)
        rates = np.zeros(si)
        for i in range(si):
            Hbroad, Hweights = h_broaden(H[i], Hwid, nloop)
            rates[i] = mod * \
                np.sum(Hweights * np.array([rate_map(H, phi) for H in Hbroad]))
        return rates
