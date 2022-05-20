import streamlit as st
import os
from pathlib import Path
import joblib
from utils import load_pipeline, convert_df

rootdir = os.getcwd()
DATAPATH = Path(rootdir) / 'data'
MODELPATH = Path(rootdir) / 'model'
Path(DATAPATH).mkdir(parents=True, exist_ok=True)
Path(MODELPATH).mkdir(parents=True, exist_ok=True)

def run_ml_app():

    if st.button('Caclulate Credit Default Risks'):

        # load data
        with open(DATAPATH / 'data.pkl','rb') as f:
            df = joblib.load(f)
        # seperate into numerical and categorical features
        num_features = list(df.dtypes[(df.dtypes == 'int64') | (df.dtypes == 'float64')].index[2:-1])
        cat_features = list(df.dtypes[df.dtypes == 'object'].index)

        # get feature array
        X = df[num_features + cat_features]

        # load pipeline
        pipeline = load_pipeline(MODELPATH, 'pipeline.pkl')

        # make predictions
        y_scores = pipeline.predict_proba(X)
        print('predicted credit default scores: ', y_scores[:,1])
        
        # append predictions to df
        df['predicted credit default scores'] = y_scores[:,1]
        print(df.head())

        # convert data to csv
        csv = convert_df(df) 
        st.download_button(label="Download data as CSV", data=csv, file_name='data_scored.csv', mime='text/csv')    