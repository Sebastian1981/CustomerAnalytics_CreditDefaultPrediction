import streamlit as st
import os
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from utils import plot_num_feature_distribution, plot_cat_feature_distribution, plot_target_distribution

rootdir = os.getcwd()
DATAPATH = Path(rootdir) / 'data'
Path(DATAPATH).mkdir(parents=True, exist_ok=True)

def run_eda_app():
    submenu = ["upload data file", "descriptive stats", "data types", "target distribution", "feature distribution"]
    choice = st.sidebar.selectbox("SubMenu", submenu)
    
    if choice == "upload data file":
        st.subheader('Upload Your Dataset.')
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            joblib.dump(df, DATAPATH / 'data.pkl')
            st.dataframe(df.head())

    elif choice == "descriptive stats":
        st.subheader('Descriptive Statistics')
        with open(DATAPATH / 'data.pkl','rb') as f:
            df = joblib.load(f)
        st.write(df.describe())
    elif choice == "data types":
        st.subheader('Data Types')
        with open(DATAPATH / 'data.pkl','rb') as f:
            df = joblib.load(f)
        st.write(df.dtypes.astype(str))
    elif choice == "target distribution":
        st.subheader('Target Distribution')
        with open(DATAPATH / 'data.pkl','rb') as f:
            df = joblib.load(f)
        fig = plot_target_distribution(df, 'Status')
        st.pyplot(fig)
    elif choice == "feature distribution":
        st.subheader('Feature Distribution')
        with open(DATAPATH / 'data.pkl','rb') as f:
            df = joblib.load(f)
        # get numeric and categorical features
        num_features = list(df.dtypes[(df.dtypes == 'int64') | (df.dtypes == 'float64')].index[2:-1])
        cat_features = list(df.dtypes[df.dtypes == 'object'].index)
        # plot numeric feature distributions
        num_feature = st.selectbox("Select a numeric feature", num_features)
        fig = plot_num_feature_distribution(df, num_feature)
        st.pyplot(fig)
        # plot categorical feature distributions
        cat_feature = st.selectbox("Select a categorical feature", cat_features)
        fig = plot_cat_feature_distribution(df, cat_feature)
        st.pyplot(fig)

    else:
        st.warning("You need to upload a csv file.")