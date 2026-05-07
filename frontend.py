import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from backend import bose_fermi_profiles, tf_radius
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
st.title("Theo's Density Profile Calculator")
# Physical parameters (hbar=1, mb=1, wb=1)
# Inspired by Rb-87 bosons + K-40 fermions
gb = float(st.text_input("Enter a value for g_B", value = 0.5))
gbf = float(st.text_input("Enter a value for g_BF", value = 0.0))
wb = float(st.text_input("Enter a value for w_B", value = 1.0))
wf = float(st.text_input("Enter a value for w_F", value = 1.0))
mb = float(st.text_input("Enter a value for m_B", value = 1.0))
mf = float(st.text_input("Enter a value for m_F", value = 40/87))

NB, NF = 1000.0, 500.0

r_ref, nB_ref, nF_ref, _, _ = bose_fermi_profiles(NB, NF, mb, mf, wb, wf, gb, gBF=0.0)
cum_B = np.cumsum(nB_ref * r_ref**2) / (np.sum(nB_ref * r_ref**2) + 1e-30)
cum_F = np.cumsum(nF_ref * r_ref**2) / (np.sum(nF_ref * r_ref**2) + 1e-30)
xlim_max = max(r_ref[np.searchsorted(cum_B, 0.999)],
               r_ref[np.searchsorted(cum_F, 0.999)]) * 1.4

nfcenter = nF_ref[1]
label = ""
dmuF  = (6*np.pi**2 * nfcenter)**(2/3) / (3 * mf * nfcenter)
if gbf == 0.0:
            label = "Non-interacting"
elif gbf**2 >= gb *dmuF :
            label = "Strong Repulsion" if gbf > 0 else "Collapse (strong attraction)"
elif gbf > 0.0:
            label = "Weak Repulsion"
else:
            label = "Attraction"

fig, ax = plt.subplots()
r, nB, nF, muB, muF = bose_fermi_profiles(NB, NF, mb, mf, wb, wf, gb, gbf)
RTF_B=tf_radius(r, nB),
RTF_F=tf_radius(r, nF),
param_str = (f'Assumptions (dimensionless HO units), \n'
             f'  hbar = mB = omegaB = 1  →  a_ho = 1, \n'
             f'  mF = 40/87 = {mf:.3f}  (K-40 / Rb-87), \n'
             f'  omegaF = {wf},  gB = {gb}, \n'
             f'  NB = {NB},  NF = {NF}')

ax.plot(r, nB, label='nB(r)  bosons', color='tab:blue', lw=2)
ax.plot(r, nF, label='nF(r)  fermions', color='tab:orange', lw=2, ls='--')
ax.set_title(label, fontsize=10)
ax.set_xlabel(r'$r\;/\;a_{ho}$')
ax.set_ylabel(r'density $n(r)\;[a_{ho}^{-3}]$')
ax.axvline(RTF_B, color='tab:blue',   lw=1, ls=':', alpha=0.7,
               label=fr"$R_{{TF,B}}="+str({RTF_B[0]:.2f})+"$")
ax.axvline(RTF_F, color='tab:orange', lw=1, ls=':', alpha=0.7,
               label=fr"$R_{{TF,F}}="+str({RTF_F[0]:.2f})+"$")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_xlim(0, xlim_max)
info = (f"$n_B(0) = {nB[1]:.3f}$\n"
            f"$n_F(0) = {nF[1]:.3f}$")
ax.annotate(info, xy=(0.97, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

fig.suptitle('Bose-Fermi Mixture Density Profiles (TF/LDA, spherical harmonic trap)',
             fontsize=12)
st.pyplot(fig)
st.write(param_str)
