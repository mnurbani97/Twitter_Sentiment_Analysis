import streamlit as st
import eda
import prediction

page = st.sidebar.selectbox('Pilih Halaman : ', ('Dashboard', 'Prediction'))

if page == 'Dashboard' : 
    eda.run()
else:
    prediction.run()