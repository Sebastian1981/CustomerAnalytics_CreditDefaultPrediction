import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline

def convert_df(df):
    """Convert dataframe to csv"""
    return df.to_csv(index=False).encode('utf-8')

def get_categorical_feature_names_encoded(pipeline:Pipeline, hotencoding_label:str, categorical_feature_names:list):
    """Get the names of categorical features after being hotencoded in a sklearn pipeline. 
        Pass in the pipeline object, the label of the one-hotencoding step in the pipeline. 
        Also pass in the names of the categorical feautures before the one-hot-encoding.
        Watchout: the code is little instable since the hard-coded transformers list: transformers_[1][1]. U might need to adapt this!"""
    return list(pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps[hotencoding_label].get_feature_names_out(categorical_feature_names))    

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

   
