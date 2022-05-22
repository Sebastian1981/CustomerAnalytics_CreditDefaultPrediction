import streamlit as st
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from utils import get_categorical_feature_names_encoded
import matplotlib.pyplot as plt
import shap


rootdir = os.getcwd()
DATAPATH = Path(rootdir) / "data"
MODELPATH = Path(rootdir) / "model"

def run_explain_app():
    # load a sample of the testing dataset
    with open(DATAPATH / 'df_test.pkl','rb') as f:
        df_test = joblib.load(f).sample(100)
    # load features
    with open(MODELPATH / 'num_features.pkl','rb') as f:
        num_features = joblib.load(f)
    with open(MODELPATH / 'cat_features.pkl','rb') as f:
        cat_features = joblib.load(f)
        
    # load trained model
    try:
        with open(MODELPATH / 'pipeline.pkl','rb') as f:
            pipeline = joblib.load(f)
            # get encoded categorical feature names
            cat_features_enc = get_categorical_feature_names_encoded(pipeline, 'onehotencode', cat_features)
    except FileNotFoundError as e:
        st.error("""Please train the model first.""")

    st.write('Explaining the following test dataset')
    st.dataframe(df_test)

    # calc shap values
    if st.button('calculate shapley values'):
        with st.spinner('calculating shapley values...'):        

            # init shap explainer
            explainer = shap.explainers.Permutation(model = pipeline['classifier'].predict,
                                                    masker = pipeline['preprocessor'].transform(df_test), 
                                                    feature_names = num_features + cat_features_enc,
                                                    max_evals=1000)
            # calculate shapley values
            shap_values = explainer(pipeline['preprocessor'].transform(df_test)) 

            # save shapley values
            with open(MODELPATH / 'shap_values.pkl','wb') as f:
                joblib.dump(shap_values, f)

    # show global feature importance
    if st.button('show shap bar plot'):
        # load shapley values
        try:
            with open(MODELPATH / 'shap_values.pkl','rb') as f:
                shap_values = joblib.load(f)
        except FileNotFoundError as e:
            st.error("""Please calculate the shapley values first.""")
        
        # bar plot 
        shap.plots.bar(shap_values, 
                       show=False)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_title('Feature importance')
        st.pyplot(fig)


    if st.button('show shap beeswarm plot'): 
        # load shapley values
        try:
            with open(MODELPATH / 'shap_values.pkl','rb') as f:
                shap_values = joblib.load(f)
        except FileNotFoundError as e:
            st.error("""Please calculate the shapley values first.""")
        
        # bar plot 
        shap.plots.beeswarm(shap_values, 
                            show=False)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_title('Feature importance')
        st.pyplot(fig)



    # get relevant features from fitted pipeline  
    feat = st.selectbox("Select feature for partial dependence plot:", num_features+cat_features_enc)
    
    if st.button('show partial dependence plot'):
        # load shapley values
        try:
            with open(MODELPATH / 'shap_values.pkl','rb') as f:
                shap_values = joblib.load(f)
        except FileNotFoundError as e:
            st.error("""Please calculate the shapley values first.""")
        
        fig = plt.figure(figsize=(8,6))
        ax = fig.gca()
        shap.dependence_plot(feat, 
                            shap_values = shap_values.values, 
                            features = pd.DataFrame(data = pipeline['preprocessor'].transform(df_test), 
                                                    columns = num_features + cat_features_enc),
                            x_jitter = 0.5,
                            xmin="percentile(5.0)",
                            xmax="percentile(95.0)", 
                            interaction_index=None,
                            title = 'SHAP Dependence Plot: SHAP Value vs {}'.format(feat),
                            ax=ax,
                            show=False)
        ax.grid('on')
        st.pyplot(fig)

