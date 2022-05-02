from operator import index
import os
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


# set paths
rootdir = os.getcwd()
MODELPATH = Path(rootdir).parents[0] / 'models'
# load pipeline
with open(MODELPATH / 'pipeline.pkl','rb') as f:
    pipeline = joblib.load(f)
print('Imported the pipeline: \n', pipeline)


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')


uploaded_file = st.file_uploader("Choose a file")


if uploaded_file is not None:
    #read csv
    df = pd.read_csv(uploaded_file)
    print('The data was successfully loaded.')
    print('The data contains {} rows and {} columns.'.format(df.shape[0], df.shape[1]))

    # seperate into numerical and categorical features
    num_features = list(df.dtypes[(df.dtypes == 'int64') | (df.dtypes == 'float64')].index[2:-1])
    cat_features = list(df.dtypes[df.dtypes == 'object'].index)
    print('-----------------------------------')
    print('detected those numeric features: \n', num_features)
    print('-----------------------------------')
    print('detected those categorical features: \n', cat_features)

    # get feature array
    X = df[num_features + cat_features]

    # make predictions
    y_scores = pipeline.predict_proba(X)
    print('predicted credit default scores: ', y_scores[:,1])
    
    # append predictions to df
    df['predicted credit default scores'] = y_scores[:,1]
    print(df.head())

    # convert data to csv
    csv = convert_df(df) 
    st.download_button(label="Download data as CSV", data=csv, file_name='data_scored.csv', mime='text/csv')




else:
    st.warning("You need to upload a csv file.")





