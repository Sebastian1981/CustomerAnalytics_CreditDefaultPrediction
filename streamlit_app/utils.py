import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
import shap

def plot_num_feature_distribution(df:pd.DataFrame, num_feature:str):
            """show distribution and boxplots for a given numeric feature."""
            fig, ax = plt.subplots(1,2, figsize=(16,6))    
            sns.histplot(data=df, 
                        x=num_feature, 
                        hue='Status', 
                        stat='percent', 
                        kde=True,
                        element='step',
                        ax=ax[0])
            ax[0].set_title(num_feature)
            ax[0].legend(['credit default', 'no credit default'])

            sns.boxplot(data=df, 
                        y=num_feature, 
                        x='Status',
                        ax=ax[1])
            ax[1].set_title(num_feature)
            ax[1].set_xticklabels(['credit default', 'no credit default'])
            return fig

def plot_cat_feature_distribution(df, cat_feature):
    """plot histogram for a given categorical feauture stratified by the target columns e.g. here Status"""
    fig, ax = plt.subplots(figsize=(8,6))
    ax = sns.countplot(x=cat_feature, hue="Status", data=df, ax=ax)
    ax.set_title('Barplot Stratified by ' + cat_feature, fontdict={'fontsize': 12, 'fontweight': 'medium'})
    ax.legend(['not credit default', 'credit default'], fontsize=12)
    return fig

def plot_target_distribution(df, target_label:str='Status'):
    """plot histogram for the target variable e.g. here Status"""
    fig, ax = plt.subplots(figsize=(8,6))
    ax = sns.countplot(x=target_label, data=df, ax=ax)
    ax.set_title('credit default distribution', fontdict={'fontsize': 12, 'fontweight': 'medium'})
    return fig

def convert_df(df):
    """Convert dataframe to csv"""
    return df.to_csv(index=False).encode('utf-8')

def get_categorical_feature_names_encoded(pipeline:Pipeline, hotencoding_label:str, categorical_feature_names:list):
    """Get the names of categorical features after being hotencoded in a sklearn pipeline. 
        Pass in the pipeline object, the label of the one-hotencoding step in the pipeline. 
        Also pass in the names of the categorical feautures before the one-hot-encoding.
        Watchout: the code is little instable since the hard-coded transformers list: transformers_[1][1]. U might need to adapt this!"""
    return list(pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps[hotencoding_label].get_feature_names_out(categorical_feature_names))    

def select_row_from_dataframe(dataframe:pd.DataFrame, row:int)->pd.DataFrame:
    """Select single row from dataframe for model explainability given a row number."""
    return dataframe.iloc[np.arange(row,row+1),:]

def select_id_from_dataframe(dataframe:pd.DataFrame, id:int)->pd.DataFrame:
    """Select single row from dataframe for model explainability given a particular ID"""
    return dataframe.loc[dataframe['ID']==id,:]

def get_sorted_shap_values(shap_values, order='descending')->np.array:
    """get sorted shap_values from shap values input. Sorted mean in descending order using the absolute value of shap values."""
    shap_values_sorted = [shap_values[0].values[ind] for ind in np.argsort(np.abs(shap_values[0].values), kind='stable')]
    if order == 'ascending':
        return shap_values_sorted
    elif order == 'descending':
        return shap_values_sorted[::-1]
    else:
        print('check order specification')

def get_sorted_features_from_shap_values(shap_values, order='descending')->list:
    """get sorted feature names from shap values input. Sorted mean in descending order using the absolute shap values."""
    feature_names_sorted = [shap_values.feature_names[feature_ind] for feature_ind in np.argsort(np.abs(shap_values[0].values), kind='stable')]
    if order == 'ascending':
        return feature_names_sorted
    elif order == 'descending':
        return feature_names_sorted[::-1]
    else:
        print('check order specification')

def get_shap_values_list(pipeline, feature_names, dataframe:pd.DataFrame, row_selected:pd.DataFrame, number_random_samples:int=20)->list:
    """Get a list of shap values by randomly drawing samples from the input dataframe and calculating shap values for a single data row.
        It is assumed that the trained pipeline is inputted with the model being accessable using pipeline['rfe] and the data transformation
        being accessable using pipeline['preprocessor'].transform(data). Also provide a list of the feature names outputted by the pipeline."""
    shap_values_list = [] # collect n shap values for calculating the mean and standard deviation
    for n in range(number_random_samples):
        # get data sample
        data = dataframe.sample(100, replace=True) 
        # init shap explainer
        explainer = shap.explainers.Permutation(model=pipeline['rfe'].predict,
                                                masker=pipeline['preprocessor'].transform(data), 
                                                feature_names=feature_names,
                                                max_evals=1000)
        # calculate shapley values
        shap_values = explainer(pipeline['preprocessor'].transform(row_selected))
        shap_values_list.append(shap_values)
    return shap_values_list        

def get_mean_from_shap_value_list(shap_values_list):
    return np.array([shap_values_list[n][0].values for n in range(len(shap_values_list))]).mean(axis=0)

def get_sd_from_shap_value_list(shap_values_list):
    return np.array([shap_values_list[n][0].values for n in range(len(shap_values_list))]).std(axis=0)        

def get_sorted_mean_shap_values(shap_values_list:list, order='descending')->np.array:
    """Get sorted shap_values inputing the shap_value_list from the method "get_shap_values_list". 
        Sorted mean in descending order using the absolute value of shap values.
        The sorted mean and the sorted standard deviation are returned """
    # calculate mean values and standard deviation
    shap_values_mean = get_mean_from_shap_value_list(shap_values_list)
    shap_values_sd = get_sd_from_shap_value_list(shap_values_list)
    # sorte shap values 
    shap_values_mean_sorted = [shap_values_mean[ind] for ind in np.argsort(np.abs(shap_values_mean), kind='stable')]
    shap_values_sd_sorted = [shap_values_sd[ind] for ind in np.argsort(np.abs(shap_values_mean), kind='stable')]
    if order == 'ascending':
        return shap_values_mean_sorted, shap_values_sd_sorted
    elif order == 'descending':
        return shap_values_mean_sorted[::-1], shap_values_sd_sorted[::-1]
    else:
        print('check order specification')

def get_sorted_features_from_mean_shap_values(shap_values_list:list, order='descending')->np.array:
    """get sorted feature names inputing the shap_value_list from the method "get_shap_values_list"."""
    # calculate mean values and standard deviation
    shap_values_mean = get_mean_from_shap_value_list(shap_values_list)
    # sort the feature names
    feature_names_sorted = [shap_values_list[0].feature_names[feature_ind] for feature_ind in np.argsort(np.abs(shap_values_mean), kind='stable')]
    if order == 'ascending':
        return feature_names_sorted
    elif order == 'descending':
        return feature_names_sorted[::-1]
    else:
        print('check order specification')