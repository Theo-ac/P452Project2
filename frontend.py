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
