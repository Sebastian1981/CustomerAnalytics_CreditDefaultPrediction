import streamlit as st
import os
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve 

############################################################################################
# set paths
############################################################################################
rootdir = os.getcwd()
DATAPATH = Path(rootdir) / "data"
MODELPATH = Path(rootdir) / "model"

def run_eval_app():

    # load training dataset
    with open(DATAPATH / 'df_train.pkl','rb') as f:
        df_train = joblib.load(f)
    with open(DATAPATH / 'X_train.pkl','rb') as f:
        X_train = joblib.load(f)
    with open(DATAPATH / 'y_train.pkl','rb') as f:
        y_train = joblib.load(f)    

    # load testing dataset
    with open(DATAPATH / 'df_test.pkl','rb') as f:
        df_test = joblib.load(f)
    with open(DATAPATH / 'X_test.pkl','rb') as f:
        X_test = joblib.load(f)
    with open(DATAPATH / 'y_test.pkl','rb') as f:
        y_test = joblib.load(f)

    # load trained model
    try:
        with open(MODELPATH / 'pipeline.pkl','rb') as f:
            pipeline = joblib.load(f)
    except FileNotFoundError as e:
        st.error("""Please train the model first.""")


    # show model statistics
    if st.button('show model performance stats'): 
    
        # make barplot of model performance stats
        fig, ax = plt.subplots(2, 2, figsize=(8,8))

        x = ['train set', 'test set']
        height = [accuracy_score(y_train, pipeline.predict(df_train)), accuracy_score(y_test, pipeline.predict(df_test))]
        ax[0,0].bar(x, height)
        ax[0,0].set_ylim([0,1])
        ax[0,0].set_title('accuracy score')

        x = ['train set', 'test set']
        height = [roc_auc_score(y_train, pipeline.predict(df_train)), roc_auc_score(y_test, pipeline.predict(df_test))]
        ax[0,1].bar(x, height)
        ax[0,1].set_ylim([0,1])
        ax[0,1].set_title('roc-auc score')

        x = ['train set', 'test set']
        height = [recall_score(y_train, pipeline.predict(df_train)), recall_score(y_test, pipeline.predict(df_test))]
        ax[1,0].bar(x, height)
        ax[1,0].set_ylim([0,1])
        ax[1,0].set_title('recall score')

        x = ['train set', 'test set']
        height = [precision_score(y_train, pipeline.predict(df_train)), precision_score(y_test, pipeline.predict(df_test))]
        ax[1,1].bar(x, height)
        ax[1,1].set_ylim([0,1])
        ax[1,1].set_title('precision score')

        st.pyplot(fig)

    # show confusion matrix
    if st.button('show models confusion matrix'):

        # plot confusion matrix for training and test set
        fig, ax = plt.subplots(1,2,figsize=(12,6))

        cm = confusion_matrix(y_train, pipeline.predict(df_train), labels=pipeline.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
        disp.plot(ax=ax[0])
        ax[0].set_title("Confusion matrix: training set")

        cm = confusion_matrix(y_test, pipeline.predict(df_test), labels=pipeline.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
        disp.plot(ax=ax[1])
        ax[1].set_title("Confusion matrix: test set")

        st.pyplot(fig)


    # plot roc curve
    if st.button('show models roc curve'):

        # plot roc curves for training and test set
        fig, ax = plt.subplots(1,2,figsize=(12,6))

        fpr, tpr, _ = roc_curve(y_train, pipeline.predict_proba(df_train)[:,1])
        disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=True)
        disp.plot(ax=ax[0])
        ax[0].set_title("ROC curve: training set")
        ax[0].plot([0, 1], [0, 1], 'k--')
        ax[0].grid('on')

        fpr, tpr, _ = roc_curve(y_test, pipeline.predict_proba(df_test)[:,1])
        disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=True)
        disp.plot(ax=ax[1])
        ax[1].set_title("ROC curve: test set")
        ax[1].plot([0, 1], [0, 1], 'k--')
        ax[1].grid('on')

        st.pyplot(fig)


    # plot roc curve
    if st.button('show models recall-precision curve'):

        # plot precision recall curves for training and test set
        fig, ax = plt.subplots(1,2,figsize=(12,6))

        precision, recall, _ = precision_recall_curve(y_train, pipeline.predict_proba(df_train)[:,1])
        disp = PrecisionRecallDisplay(precision, recall)
        disp.plot(ax=ax[0])
        ax[0].set_title("Precision-recall curve: train set")
        ax[0].plot([1, 0], [0, 1], 'k--')
        ax[0].grid('on')

        precision, recall, _ = precision_recall_curve(y_test, pipeline.predict_proba(df_test)[:,1])
        disp = PrecisionRecallDisplay(precision, recall)
        disp.plot(ax=ax[1])
        ax[1].set_title("Precision-recall curve: test set")
        ax[1].plot([1, 0], [0, 1], 'k--')
        ax[1].grid('on')

        st.pyplot(fig)
