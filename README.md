# Credit Default Prediction with Machine Learning

## Project Overview
The purpose of this AI project is to predict the risk that individuals will default on their loan repayments. The basis for this is freely accessible data from the world wide web. First, the feasibility of implementing the question using machine learning algorithms was examined. The implementation for this is in the various Jupyter notebooks. A web application was then implemented, which allows the user to visualize the underlying data with simple menu navigation, and then to train and evaluate machine learning models. In addition, an approach from game theory was implemented here to make model decisions transparent by visualizing the so-called Shapley values.

## Exploratory Data Analysis
The distribution of the target variable shows a class imbalance having around 20% loan repayment defaulters. This class imbalance was considered for modeling in order to achieve optimal model performances.

![target distribution](/images/target_distribution.png)

The web application let´s you easily select particular features to be displayed in terms of their distriutions, as shown below. For example, from the boxplots for the income variable can be seen that the risk for credit default is higher for lower incomes, as can be expected.

![numeric feature distribution](/images/numeric_feature_distribution.PNG)

The figure below shows the credit default distribution stratified by gender.

![categorical feature distribution](/images/categorical_feature_distribution.PNG)

