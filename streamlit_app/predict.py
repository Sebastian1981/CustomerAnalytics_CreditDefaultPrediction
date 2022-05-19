# import modules
import os
from pathlib import Path
import joblib
import pandas as pd

# set paths
rootdir = os.getcwd()
DATAPROCESSEDPATH = Path(rootdir) / 'data'
MODELPATH = Path(rootdir) / 'model'


# load training dataset
df = pd.read_csv(DATAPROCESSEDPATH / 'data_preprocessed.csv')

# load pipeline
with open(MODELPATH / 'pipeline.pkl','rb') as f:
    pipe_loaded = joblib.load(f)

print(pipe_loaded)

## select first row
#df = df[0:1]

# seperate into numerical and categorical features
num_features = list(df.dtypes[(df.dtypes == 'int64') | (df.dtypes == 'float64')].index[2:-1])
cat_features = list(df.dtypes[df.dtypes == 'object'].index)
label = ['Status']
print('-----------------------------------')
print('numeric features: \n', num_features)
print('-----------------------------------')
print('categorical features: \n', cat_features)

# get feature array
X = df[num_features + cat_features]

# make prediction
y_scores = pipe_loaded.predict_proba(X)
print('predicted credit default scores: ', y_scores[:,1])

# append predictions to df
df['predicted credit default scores'] = y_scores[:,1]
print(df.head())