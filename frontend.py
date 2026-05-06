import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
#from backend import
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
st.title("Theo's Density Profile Calculator")
gb = st.text_input("Enter a value for g_B", value = 0.0)
gbf = st.text_input("Enter a value for g_BF", value = 0.0)
wb = st.text_input("Enter a value for w_B", value = 0.0)
wf = st.text_input("Enter a value for w_F", value = 0.0)
mb = st.text_input("Enter a value for m_B", value = 0.0)
mf = st.text_input("Enter a value for m_F", value = 0.0)

nb, nf = 1000, 500

r_ref, nB_ref, nF_ref, _, _ = bose_fermi_profiles(nb, nf, mb, mf, wb, wf, gb, gbf)
cum_B = np.cumsum(nB_ref * r_ref**2) / (np.sum(nB_ref * r_ref**2) + 1e-30)
cum_F = np.cumsum(nF_ref * r_ref**2) / (np.sum(nF_ref * r_ref**2) + 1e-30)
xlim_max = max(r_ref[np.searchsorted(cum_B, 0.999)],
               r_ref[np.searchsorted(cum_F, 0.999)]) * 1.4

