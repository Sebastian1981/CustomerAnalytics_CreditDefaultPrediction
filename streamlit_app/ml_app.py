import streamlit as st
import os
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from utils import convert_df
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


rootdir = os.getcwd()
DATAPATH = Path(rootdir) / 'data'
MODELPATH = Path(rootdir) / 'model'
Path(DATAPATH).mkdir(parents=True, exist_ok=True)
Path(MODELPATH).mkdir(parents=True, exist_ok=True)

def run_ml_app():
    submenu = ["train/test split", "train model"]
    choice = st.sidebar.selectbox("SubMenu", submenu)

    if choice == "train/test split":
        st.subheader('Train/Test Split Your Dataset.')
        st.write('Please select seed for reproducibility.')
        seed = st.slider('seed',0,1000)
        st.write('Please select the size of the test set in  percent.')
        test_size = st.slider('test set %',25,75)
        if st.button('Split Data into Train/Test'):
            # Separate into train and test sets
            with open(DATAPATH / 'data.pkl','rb') as f:
                df = joblib.load(f)
            # seperate into numerical and categorical features
            num_features = list(df.dtypes[(df.dtypes == 'int64') | (df.dtypes == 'float64')].index[2:-1])
            cat_features = list(df.dtypes[df.dtypes == 'object'].index)
            # seperate features and target
            X = df[num_features + cat_features]
            y = df['Status'].values.reshape(-1)
            X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(X, y, df, test_size=test_size/100, stratify=y, random_state=seed)
            # show data splitting results
            st.write('training dataset containing {} samples.'.format(df_train.shape[0]))
            st.dataframe(df_train[num_features + cat_features + ['Status']])
            st.write('test dataset containing {} samples.'.format(df_test.shape[0]))
            st.dataframe(df_test[num_features + cat_features + ['Status']])
            # save training and testing datasets
            joblib.dump(df_train, DATAPATH / 'df_train.pkl')
            joblib.dump(df_test, DATAPATH / 'df_test.pkl')
            joblib.dump(X_train, DATAPATH / 'X_train.pkl')
            joblib.dump(X_test, DATAPATH / 'X_test.pkl')
            joblib.dump(y_train, DATAPATH / 'y_train.pkl')
            joblib.dump(y_test, DATAPATH / 'y_test.pkl')
        

    elif choice == 'train model':
        st.subheader('Train Your Model.')
        st.write('Please select seed for reproducibility.')
        seed = st.slider('seed',0,1000)

        classifiers = ["LGBM", "LOGREG", 'XGBOOST', 'RANDOMFOREST']
        choice = st.selectbox("Classifiers", classifiers)
        
        if choice == 'LGBM':
            classifier = LGBMClassifier(random_state=seed, lass_weight='balanced')
        elif choice == 'XGBOOST':
            classifier = XGBClassifier(random_state=seed, scale_pos_weight=10)
        elif choice == 'LOGREG':
            classifier = LogisticRegression(random_state=seed, max_iter=100000, class_weight='balanced')
        elif choice == 'RANDOMFOREST':
            classifier = RandomForestClassifier(random_state=seed, class_weight='balanced')

        # Define preprocessing for numeric columns (normalize them so they're on the same scale)
        numeric_features = list(np.arange(0,3)) # note that numbers are required here, not names!
        # Define preprocessing for categorical features (e.g. encode the Age column)
        categorical_features = list(np.arange(3,17))

        # setup the pipeline elements
        numeric_transformer = Pipeline(steps=[
                                            ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
                                            ('scaler', MinMaxScaler())
                                            ]
                                    )
                                    
        categorical_transformer = Pipeline(steps=[
                                            ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
                                            ('onehotencode', OneHotEncoder(handle_unknown='error', drop='first', sparse=True))
                                                ]
                                            )

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(transformers=[
                                                    ('num', numeric_transformer, numeric_features), 
                                                    ('cat', categorical_transformer, categorical_features)
                                                    ], remainder='drop'
                                        )

        # Create preprocessing and training pipeline
        pipeline = Pipeline(steps=[
                                ('preprocessor', preprocessor),
                                ('gbc', classifier)
                                ]
                            )

        # fit the pipeline
        if st.button('fit model'):
            # load data
            with open(DATAPATH / 'X_train.pkl','rb') as f:
                X_train = joblib.load(f)
            with open(DATAPATH / 'y_train.pkl','rb') as f:
                y_train = joblib.load(f)
            
            # fit model
            with st.spinner('Wait for model to be trained...'):
                pipeline.fit(X_train, y_train)
            st.success('Done!')
            
            # save model
            joblib.dump(pipeline, MODELPATH / 'pipeline.pkl')



        if st.button('predict default risk for test dataset'):
            # load data
            with open(DATAPATH / 'df_test.pkl','rb') as f:
                df = joblib.load(f)
            
            # seperate into numerical and categorical features
            num_features = list(df.dtypes[(df.dtypes == 'int64') | (df.dtypes == 'float64')].index[2:-1])
            cat_features = list(df.dtypes[df.dtypes == 'object'].index)

            # get feature array
            X = df[num_features + cat_features]

            # load pipeline
            with open(MODELPATH / 'pipeline.pkl','rb') as f:
                pipeline = joblib.load(f)

            # make predictions
            y_scores = pipeline.predict_proba(X)
            print('predicted credit default scores: ', y_scores[:,1])
            
            # append predictions to df
            df['predicted credit default scores'] = y_scores[:,1]
            print(df.head())

            # convert data to csv
            csv = convert_df(df) 
            st.download_button(label="Download data as CSV", data=csv, file_name='data_scored.csv', mime='text/csv')    