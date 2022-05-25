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

## Model Evaluation
Once the model has been trained, the web application let´s the user easily explore the overall performance of the model in classifying the credit repayment defaulters vs the none-defaulters. For example, we can see that in this particular case, the performace on the training set is similar to the performance on the test dataset indicating there is barely any overfitting of the model. To be more specific, the overall model performance e.g. the accuracy score in the range of 80% and the area under the roc (roc-auc) indicate classification robustness. However, a closer look at the recall score and the precision score shows that only 60% of the true credit defaulters can be correctly classified by the model as such. Also the precision in the range of around 70% reveals many customers who belong to the none-defaulters incorrectly being classified as defaulters. In this particular case, the training data was sufficient. Hence, a promising attempt to increase the model performance with special focus on recall and precision would be to put more effort into both collecting more features and also engineering more feature.

![eval metrics](images/model_eval_01.PNG)

![confusion metrics](images/model_eval_02.PNG)

The receiver-operating characteristics is interesting in terms that the false-positive rate remains at zero % until the true-positive rate (recall) reaches around 50%. Higher recall can be achieved at the cost of increasing false-positive rates. 

![roc](images/model_eval_03.PNG)

The precision-recall curve confirms that achieving a recall of 50% comes in for free in terms that the precision starts decreasing only when the recall is above 50%. An efficient set point would be to chose a recall of around 80% and thus allowing the precision to drop to around 80%. However, optimizing the tradeoff between recall and precision requires profound business knowlede in terms of the costs of false-positive and false-negative predictions.  

![precision recall](images/model_eval_04.PNG)


## Model Explainability
The web application allows the user to easily visualize the overall feature importance as shown below. We can see that the overall most important feature is the income explaining 13% on average of the credit defaulting risk. 

![shap bar plot](images/shap_feature_importance.PNG)

Having a closer look at e.g. the dependence of the credit default risk in dependence of the income reveals that lower incomes correlates with higher credit defaulting risks, which is in consistence with the finding above (see EDA section).

![shap bar plot](images/shap_feature_importance_2.PNG)

Using a game theoretical approach allows us to also explore the machine leaning model decision making process for a single customer, as shown below. In this particular case, the model predicted 100% credit default risk. It shows that the most important factors were the type of credit, the income and the loan amount. All three factors significantly pushed the predicted credit default risk to higher values. 

![shap bar plot](images/shap_single_decision.PNG)