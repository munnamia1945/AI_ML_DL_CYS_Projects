# **Project 03: Customer Churn Prediction Using Machine Learning**

## **1\. Project Overview**

This project applies supervised machine learning to predict whether a telecommunications customer is likely to leave a service.

Using the IBM Telco Customer Churn dataset, five classification algorithms were trained and evaluated. The workflow includes data cleaning, feature engineering, preprocessing pipelines, class-imbalance handling, cross-validation, and permutation-based feature importance.

## **2\. Objectives**

* Predict customer churn using machine learning.  
* Compare five classification algorithms.  
* Address class imbalance using class weights.  
* Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.  
* Assess model performance using stratified cross-validation.  
* Identify influential features associated with model predictions.

## **3\. Dataset Description**

* **Dataset:** IBM Telco Customer Churn  
* **Source:** Public GitHub CSV dataset  
* **Total records:** 7,043  
* **Original columns:** 21  
* **Features used for modeling:** 19  
* **Target variable:** `Churn`

The `customerID` column was excluded from model training because it is an identifier rather than a meaningful predictive feature.

The target was encoded as:

* `Yes` → 1  
* `No` → 0

## **4\. Technologies Used**

* Python  
* Pandas  
* NumPy  
* Matplotlib  
* Scikit-learn  
* Google Colab

## **5\. Exploratory Data Analysis**

The dataset contains an imbalanced target distribution:

| Customer status | Records | Percentage |
| ----- | ----- | ----- |
| No churn | 5,174 | 73.46% |
| Churn | 1,869 | 26.54% |
| **Total** | **7,043** | **100%** |

This imbalance makes recall and F1-score important alongside accuracy when evaluating the models' ability to identify customers who churn.

## **6\. Data Preprocessing**

The preprocessing workflow included:

* Converting `TotalCharges` to numeric values.  
* Handling invalid numeric entries as missing values.  
* Removing records with a missing target.  
* Median imputation for numerical features.  
* Most-frequent imputation for categorical features.  
* One-hot encoding for categorical variables.  
* Standard scaling for numerical features.  
* Excluding `customerID` from the predictive features.

Scikit-learn's `ColumnTransformer` and `Pipeline` were used to organize preprocessing and model training.

Class weighting was used where configured to address the imbalanced target distribution.

## **7\. Models Implemented**

Five classification algorithms were evaluated:

1. Logistic Regression  
2. Decision Tree Classifier  
3. Random Forest Classifier  
4. Gradient Boosting Classifier  
5. Extra Trees Classifier

## **8\. Experimental Setup**

| Setting | Configuration |
| ----- | ----- |
| Training/test split | 80% / 20% |
| Training records | 5,634 |
| Test records | 1,409 |
| Stratification | Enabled |
| Random state for split | 42 |
| Cross-validation | 5-fold StratifiedKFold |
| Cross-validation random state | 47 |

Cross-validation was performed on the training set. The final comparison below reports the held-out test-set results.

## **9\. Evaluation Metrics**

* **Accuracy:** Overall proportion of correct predictions.  
* **Precision:** Proportion of predicted churners who actually churned.  
* **Recall:** Proportion of actual churners correctly identified.  
* **F1-score:** Harmonic mean of precision and recall.  
* **ROC-AUC:** Measures how well the model distinguishes between churn and non-churn across classification thresholds.

## **10\. Model Comparison**

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Gradient Boosting | 0.8062 | 0.6735 | 0.5241 | 0.5895 | 0.8434 |
| Logistic Regression | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8413 |
| Random Forest | 0.7828 | 0.5829 | 0.6390 | 0.6097 | 0.8336 |
| Extra Trees | 0.7537 | 0.5289 | 0.6604 | 0.5874 | 0.8198 |
| Decision Tree | 0.7303 | 0.4949 | 0.7727 | 0.6033 | 0.8004 |

### **Key Findings**

* **Gradient Boosting** achieved the highest test accuracy and ROC-AUC among the evaluated models.  
* **Logistic Regression** achieved the highest recall and F1-score.  
* Gradient Boosting's mean cross-validation ROC-AUC was **0.8478 ± 0.0122**.  
* The best model depends on the intended business objective: stronger overall discrimination or identifying a larger proportion of customers who may churn.

## **11\. Confusion Matrix**

The reported test-set confusion matrix for Gradient Boosting is:

| Actual / Predicted | No churn | Churn |
| ----- | ----- | ----- |
| Actual no churn | 940 | 95 |
| Actual churn | 178 | 196 |

*   
  **True negatives:** 940  
* **False positives:** 95  
* **False negatives:** 178  
* **True positives:** 196

The model correctly identified 196 churners and missed 178 actual churners in the test set.

## **12\. Feature Importance**

Permutation importance was used to estimate how much model performance changes when individual features are shuffled.

The reported leading features were:

1. `Contract`  
2. `tenure`  
3. `MonthlyCharges`

These results indicate which features were influential for the evaluated model. They should not be interpreted as proof that these variables cause customers to churn.

## **13\. How to Run in Google Colab**

1. Open `03-customer-churn-prediction.ipynb` in Google Colab.  
2. Ensure the notebook has internet access to retrieve the dataset.  
3. Run all cells sequentially.  
4. Review the model comparison, cross-validation, and feature-importance results.  
5. Inspect the generated output files.

## **14\. Generated Output Files**

The notebook exports the following files:

* `customer_churn_model_comparison.csv` — Test-set evaluation metrics for all models.  
* `customer_churn_cross_validation.csv` — Cross-validation results.  
* `customer_churn_feature_importance.csv` — Permutation feature-importance scores.  
* `customer_churn_results_summary.txt` — Summary of model results.

## **15\. Limitations**

* Models were evaluated using predefined configurations rather than an extensive hyperparameter search.  
* Class weighting was explored, but alternative imbalance-handling methods were not evaluated in the reported experiment.  
* The results come from a single train/test split.  
* Feature importance indicates predictive relevance, not causation.  
* Model performance may differ on data from other telecommunications providers or time periods.

## **16\. Future Improvements**

* Tune model hyperparameters using `GridSearchCV` or `RandomizedSearchCV`.  
* Evaluate additional imbalance-handling approaches, where appropriate.  
* Compare performance at different classification thresholds.  
* Analyze false negatives to understand which churners are missed.  
* Add probability calibration and explainability techniques.  
* Validate the model on additional datasets or later customer records.

## **17\. Learning Outcomes**

* Building preprocessing pipelines with `ColumnTransformer` and `Pipeline`.  
* Handling missing values and categorical variables.  
* Evaluating imbalanced classification problems.  
* Comparing precision, recall, F1-score, and ROC-AUC.  
* Applying stratified cross-validation.  
* Using permutation importance to interpret model predictions.

## **18\. Author**

**Md. Munna Mia**

CSE Undergraduate \| AI/ML and Cybersecurity

* GitHub: [https\://github.com/munnamia1945](https://github.com/munnamia1945)  
* Portfolio: [https\://www\.munnamia.com](https://www.munnamia.com/)

