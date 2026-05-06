import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from backend import bose_fermi_profiles
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
st.title("Theo's Density Profile Calculator")
gb = float(st.text_input("Enter a value for g_B", value = 1.0))
gbf = float(st.text_input("Enter a value for g_BF", value = 0.0))
wb = float(st.text_input("Enter a value for w_B", value = 1.0))
wf = float(st.text_input("Enter a value for w_F", value = 1.0))
mb = float(st.text_input("Enter a value for m_B", value = 1.0))
mf = float(st.text_input("Enter a value for m_F", value = 1.0))

nb, nf = 1000.0, 500.0

r_ref, nB_ref, nF_ref, _, _ = bose_fermi_profiles(nb, nf, mb, mf, wb, wf, gb, gbf)
cum_B = np.cumsum(nB_ref * r_ref**2) / (np.sum(nB_ref * r_ref**2) + 1e-30)
cum_F = np.cumsum(nF_ref * r_ref**2) / (np.sum(nF_ref * r_ref**2) + 1e-30)
xlim_max = max(r_ref[np.searchsorted(cum_B, 0.999)],
               r_ref[np.searchsorted(cum_F, 0.999)]) * 1.4

