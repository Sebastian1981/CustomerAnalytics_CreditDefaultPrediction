

import streamlit as st
import os
from pathlib import Path
import pandas as pd
import joblib

############################################################################################
# set paths
############################################################################################
rootdir = os.getcwd()
DATAPATH = Path(rootdir) / 'data'
Path(DATAPATH).mkdir(parents=True, exist_ok=True)


def run_eda_app():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        joblib.dump(df, DATAPATH / 'data.pkl')
    else:
        st.warning("You need to upload a csv file.")