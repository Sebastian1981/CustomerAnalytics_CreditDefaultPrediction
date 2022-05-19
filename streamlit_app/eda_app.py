

import streamlit as st
import os
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt


############################################################################################
# set paths
############################################################################################
rootdir = os.getcwd()
DATAPATH = Path(rootdir) / 'data'
Path(DATAPATH).mkdir(parents=True, exist_ok=True)


def run_eda_app():
    submenu = ["descriptive stats", "data types", 'target distribution']
    choice = st.sidebar.selectbox("SubMenu", submenu)
    
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        joblib.dump(df, DATAPATH / 'data.pkl')
        st.dataframe(df.head())

        if choice == "descriptive stats":
            st.subheader('Descriptive Statistics')
            st.write(df.describe())
        elif choice == "data types":
            st.subheader('Data Types')
            st.write(df.dtypes.astype(str))
        elif choice == "target distribution":
            fig, ax = plt.subplots(1,1)
            ax.hist(df['Status']) 
            ax.set_title('credit default distribution')
            st.pyplot(fig)

    else:
        st.warning("You need to upload a csv file.")