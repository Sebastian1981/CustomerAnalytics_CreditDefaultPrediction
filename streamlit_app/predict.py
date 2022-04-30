# import modules
import os
from pathlib import Path
import joblib

# set paths
rootdir = os.getcwd()
DATAPROCESSEDPATH = Path(rootdir).parents[0] / 'data_preprocessed'
MODELPATH = Path(rootdir).parents[0] / 'models'


# load training dataset
with open(DATAPROCESSEDPATH / 'data_preprocessed.pkl','rb') as f:
    df_new = joblib.load(f)

# load pipeline
with open(MODELPATH / 'pipeline.pkl','rb') as f:
    pipe_loaded = joblib.load(f)

print(pipe_loaded)

# select first row
df_new = df_new[0:1]

# seperate into numerical and categorical features
num_features = list(df_new.dtypes[(df_new.dtypes == 'int64') | (df_new.dtypes == 'float64')].index[2:-1])
cat_features = list(df_new.dtypes[df_new.dtypes == 'object'].index)
label = ['Status']
print('-----------------------------------')
print('numeric features: \n', num_features)
print('-----------------------------------')
print('categorical features: \n', cat_features)

# get feature array
X_new = df_new[num_features + cat_features]

# make prediction
y_scores = pipe_loaded.predict_proba(X_new)[0]
print('credit default probability: {:.1f} percent.'.format(y_scores[1]*100))