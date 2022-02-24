# Customer-Churn-Banking-

## Project Overview
In this project I built a classification model capable of predicting if a certain customer is likely to maintain his/her account with a bank or terminate said account based on certain attributes about the customer in question.

Tree based algorithms were used for model building starting with a Decision Tree Classifier which was optimised by pruning, before culminating in a Random Forest Classifier so as to utilize the benefits of an ensemble of the pruned Decision Trees. The Random Forest Classifier was then optimised by tuning its n_estimators hyperparameter.

Two models were built. The first, a baseline model which allowed me to determine baseline metrics without handling class imbalance and the second, a model built with class imbalance handled using class weights. Upon analysing results, the baseline model had an accuracy score of approximately 87% and a recall of just under 50% implying that it is not well suited to correctly identifying the positive cases (churn), making it unsuitable for the project objective. On the other hand, utilising class weights in the weighted model, I was able to increase recall to 83%, although accuracy dipped to 72%, the weighted model was therefore deemed a better fit for the project objective.

Prototyping is done in the **Churn_Modelling_(Banking).ipynb** notebook file complete with a step-by-step walk through of model building logic, visualisations and model explainability. Deployment is done via the python script **churn_app.py** culminating in a live [web application](https://share.streamlit.io/oreolorun/customer-churn-banking-/main/churn_app.py).

## _TABLE OF CONTENTS_
* __1.0 OVERVIEW__
* 1.1 Dataset Description
* __2.0 DATA PREPARATION__
* 2.1 Duplicate Check
* 2.2 Missing Value Check
* 2.3 Checking for Missing Value Placeholder
* __3.0 DESCRIPTION OF CATEGORICAL DATA__
* 3.1 Gender
* 3.2 Geography
* 3.3 HasCrCard
* 3.4 IsActiveMember
* 3.5 Exited
* __4.0 CREATING FEATURE MATRIX X AND PREDICTION TARGET Y__
* __5.0 STATISTICAL DESCRIPTION OF NUMERICAL ATTRIBUTES__
* 5.1 CreditScore
* 5.2 Age
* 5.3 Tenure
* 5.4 Balance
* 5.5 NumOfProducts
* 5.6 EstimatedSalary
* 5.7 Numerical Attributes Distribution Plots
* __6.0 FEATURE SELECTION__
* 6.1 Creating a Test Set
* 6.2 Mutual Information Scoring
* 6.3 Correlation Matrix
* 6.4 Categorical Plots
* 6.5 Creating Feature Matrix
* __7.0 MODEL BUILDING__
* 7.1 Numerical Preprocessing
* 7.2 Categorical Preprocessing
* 7.3 Combined Preprocessor
* 7.4 Model Pipeline
* 7.5 Cross Validation
* 7.6 Pruning
* _7.6.1 Optimal Effective Alpha_
* _7.6.2 Visualising Training and Validation Accuracy_
* _7.6.3 Pruned Decision Tree_
* 7.7 Random Forest Classifier
* _7.7.1 Optimising Random Forest_
* _7.7.2 Class Weights_
* __8.0 FITTING FINAL MODELS__
* 8.1 Baseline Model
* 8.2 Weighted Model
* __9.0 TESTING MODELS__
* 9.1 Baseline Model
* _9.1.1 Results_
* _9.1.2 Metrics and Confusion Matrix Plots_
* 9.2 Weighted Model
* _9.2.1 Results_
* _9.2.2 Metrics and Consusion Matrix Plots_
* _9.3 Model Explanation_
* _9.3.1 SHAP Plots_
* __10.0 CONCLUSION__

![confusion_matrix](https://user-images.githubusercontent.com/92114396/142599544-fbcd8a25-1f05-40f4-89dd-4c374bb9c151.png)

