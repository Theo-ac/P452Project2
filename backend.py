import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
import matplotlib.pyplot as plt

import quspin
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general

# ── QuSpin helpers ─────────────────────────────────────────────────────────────

def _translation_perms(Lx, Ly):
    """Return (T_x, T_y) permutation arrays for Lx×Ly lattice (site = x + y*Lx)."""
    T_x = np.array([((x + 1) % Lx) + y * Lx for y in range(Ly) for x in range(Lx)])
    T_y = np.array([x + ((y + 1) % Ly) * Lx for y in range(Ly) for x in range(Lx)])
    return T_x, T_y


def build_H_quspin(Lx, Ly, Nup, J=1.0, kblock_x=None, kblock_y=None):
    """
    Build the J-coupling Heisenberg Hamiltonian  H_J = J Σ S_i·S_j
    in the Nup (= N/2 + Sz) sector using QuSpin.

    The Zeeman term  -h·Sz  is a constant within each sector and is
    added externally as  E_total = E_J - h*(Nup - N/2).

    kblock_x / kblock_y : momentum quantum numbers (integers 0..Lx-1 / 0..Ly-1)
                          for translational symmetry.  None = no translation used.
    """
    N = Lx * Ly
    bonds = get_square_bonds_pbc(Lx, Ly)

    # QuSpin spin_basis_general uses Pauli-scaled operators:
    #   "z"  eigenvalues ±1  (= 2 S^z),  "±" amplitude 2  (= 2 S^±)
    # Therefore  S_i·S_j = (1/4)"zz" + (1/8)["+-" + "-+"]
    Jzz = [[J / 4, i, j] for i, j in bonds]
    Jpm = [[J / 8, i, j] for i, j in bonds]
    Jmp = [[J / 8, i, j] for i, j in bonds]
    static = [["zz", Jzz], ["+-", Jpm], ["-+", Jmp]]

    kw = {"Nup": Nup}
    if kblock_x is not None or kblock_y is not None:
        T_x, T_y = _translation_perms(Lx, Ly)
        if kblock_x is not None:
            kw["Tx"] = (T_x, kblock_x)
        if kblock_y is not None:
            kw["Ty"] = (T_y, kblock_y)

    basis = spin_basis_general(N, **kw)
    H = hamiltonian(static, [], basis=basis, dtype=np.float64,
                    check_herm=False, check_pcon=False, check_symm=False)
    return H, basis


def precompute_spectrum_quspin(Lx, Ly, J=1.0, n_eigs=8,
                                kblock_x=None, kblock_y=None, verbose=True):
    """
    Diagonalise H_J in every Nup sector and return a dict {Nup: eigenvalue_array}.
    This only needs to run ONCE; the h-dependence is then a simple shift.
    """
    N = Lx * Ly
    spectrum = {}
    total_dim = 0
    for Nup in range(N + 1):
        H, basis = build_H_quspin(Lx, Ly, Nup, J=J,
                                   kblock_x=kblock_x, kblock_y=kblock_y)
        D = basis.Ns
        if D == 0:
            continue
        total_dim += D
        sz_val = Nup - N / 2
        if verbose:
            print(f"  Nup={Nup:3d}  Sz={sz_val:+5.1f}  dim={D:>10,}")
        k = min(n_eigs, D)
        if D <= 2 * k:
            ev = np.sort(H.eigvalsh())[:k]
        else:
            ev = np.sort(H.eigsh(k=k, which='SA', return_eigenvectors=False))
        spectrum[Nup] = ev
    if verbose:
        print(f"  Total Hilbert-space dim explored: {total_dim:,}")
    return spectrum


def h_sweep_from_spectrum(spectrum, N, h_vals, n_lowest=6):
    """
    Given pre-computed {Nup: E_J eigenvalues} and Zeeman shift,
    return:
      gs_sz_arr  - ground-state Sz at each h
      lowest_arr - (len(h_vals), n_lowest) lowest total eigenvalues
    """
    gs_sz_arr  = np.empty(len(h_vals))
    lowest_arr = np.empty((len(h_vals), n_lowest))

    for hi, h in enumerate(h_vals):
        all_ev = []
        best_e, best_sz = np.inf, None
        for Nup, evs_J in spectrum.items():
            sz = Nup - N / 2
            evs_tot = evs_J - h * sz
            for e in evs_tot:
                all_ev.append(e)
            if evs_tot[0] < best_e:
                best_e, best_sz = evs_tot[0], sz
        all_ev.sort()
        gs_sz_arr[hi]  = best_sz
        lowest_arr[hi] = all_ev[:n_lowest]

    return gs_sz_arr, lowest_arr

def bose_fermi_profiles(NB, NF, mB, mF, omegaB, omegaF, gB, gBF,
                        r_max=None, Nr=600, tol=1e-5, max_iter=200, mix=0.4):
    """
    Self-consistent Thomas-Fermi / LDA density profiles for a spherical
    Bose-Fermi mixture in a harmonic trap.

    Convention: hbar = 1.
    TF (bosons):   nB(r) = max(0, (muB - VB(r) - gBF*nF(r)) / gB)
    LDA (fermions): nF(r) = (1/6pi^2) * (2*mF*max(0, muF - VF(r) - gBF*nB(r)))^(3/2)
    """
    if r_max is None:
        # BEC Thomas-Fermi radius: R_TF = (15 gB NB / (4π mB ω²))^(1/5)
        RTF_B = (15 * gB * NB / (4 * np.pi * mB * omegaB**2))**0.2 * 2.0
        # Fermi radius from 3D harmonic trap: E_F = (6 N_F)^(1/3) ħω,
        # R_F = sqrt(2 E_F / (m_F ω²))
        EF_est = (6 * NF)**(1/3)
        RTF_F = np.sqrt(2 * EF_est / (mF * omegaF**2)) * 1.5
        r_max = max(RTF_B, RTF_F) * 1.5

    r  = np.linspace(0, r_max, Nr)
    VB = 0.5 * mB * omegaB**2 * r**2
    VF = 0.5 * mF * omegaF**2 * r**2

    def norm(n):
        return 4 * np.pi * np.trapezoid(n * r**2, r)

    def nB_of(muB, nF):
        return np.maximum((muB - VB - gBF * nF) / gB, 0.0)

    def nF_of(muF, nB):
        arg = 2 * mF * np.maximum(muF - VF - gBF * nB, 0.0)
        return np.sqrt(arg)**3 / (6 * np.pi**2)

    def find_mu(fn, partner, N_target, lo, hi):
        def resid(mu):
            return norm(fn(mu, partner)) - N_target
        # Auto-expand bracket if needed
        for _ in range(30):
            if resid(hi) > 0:
                break
            hi *= 2
        for _ in range(30):
            if resid(lo) < 0:
                break
            lo -= abs(lo) + 0.1
        return brentq(resid, lo, hi, xtol=1e-10)

    # Initial guess: non-interacting
    muB = gB * NB / (4*np.pi/3 * r_max**3 / 8)   # rough centre density
    muF = (6*np.pi**2 * NF / (4*np.pi/3 * (r_max/2)**3))**(2/3) / (2*mF)
    nB  = nB_of(muB, np.zeros(Nr))
    nF  = nF_of(muF, np.zeros(Nr))

    for _ in range(max_iter):
        nB_old, nF_old = nB.copy(), nF.copy()

        muB = find_mu(nB_of, nF, NB, lo=-100.0, hi=VB.max() + gB*NB + abs(gBF)*NF + 1)
        nB_new = nB_of(muB, nF)

        muF = find_mu(nF_of, nB_new, NF, lo=-100.0,
                      hi=VF.max() + (6*np.pi**2*NF)**(2/3)/(2*mF) + abs(gBF)*nB_new.max() + 1)
        nF_new = nF_of(muF, nB_new)

        nB = (1 - mix)*nB + mix*nB_new
        nF = (1 - mix)*nF + mix*nF_new

        delta = max(np.max(np.abs(nB - nB_old)), np.max(np.abs(nF - nF_old)))
        if delta < tol:
            break

    return r, nB, nF, muB, muF
