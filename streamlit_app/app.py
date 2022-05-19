from importlib.resources import path
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve 


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import os
import joblib
from pathlib import Path

import shap
from utils import convert_df
from eda_app import run_eda_app
#from config_app import run_config_app
from ml_app import load_pipeline
#from eval_app import run_eval_app
#from explain_app import run_explain_app

import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)


############################################################################################
# set paths
############################################################################################
rootdir = os.getcwd()
DATAPATH = Path(rootdir) / 'data'
MODELPATH = Path(rootdir) / 'model'
Path(DATAPATH).mkdir(parents=True, exist_ok=True)
Path(MODELPATH).mkdir(parents=True, exist_ok=True)


def main():
    st.title("Let´s predict credit defaults!")

    menu = ["About", "EDA", "ML", "Eval", "Explain"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "About":
        st.subheader("About")

    elif choice == "EDA":
        st.subheader('Exploratory Data Analysis')
        run_eda_app()
        logging.info("running eda app.")
    
    elif choice == "ML":
        # load data
        with open(DATAPATH / 'data.pkl','rb') as f:
            df = joblib.load(f)
        # seperate into numerical and categorical features
        num_features = list(df.dtypes[(df.dtypes == 'int64') | (df.dtypes == 'float64')].index[2:-1])
        cat_features = list(df.dtypes[df.dtypes == 'object'].index)
        print('-----------------------------------')
        print('detected those numeric features: \n', num_features)
        print('-----------------------------------')
        print('detected those categorical features: \n', cat_features)

        # get feature array
        X = df[num_features + cat_features]

        # load pipeline
        pipeline = load_pipeline(MODELPATH, 'pipeline.pkl')
        logging.info("pipeline successfully loaded.")

        # make predictions
        y_scores = pipeline.predict_proba(X)
        print('predicted credit default scores: ', y_scores[:,1])
        
        # append predictions to df
        df['predicted credit default scores'] = y_scores[:,1]
        print(df.head())

        # convert data to csv
        csv = convert_df(df) 
        st.download_button(label="Download data as CSV", data=csv, file_name='data_scored.csv', mime='text/csv')

    
    
    
    

    
    


    
    


if __name__ == "__main__":
    main()



