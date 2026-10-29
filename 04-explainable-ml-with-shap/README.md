# **Project 04: Explainable Machine Learning Using SHAP**

## **1\. Overview**

This project demonstrates how to interpret machine learning predictions using **SHAP (SHapley Additive exPlanations)**. A Gradient Boosting classifier is trained to predict customer churn using the IBM Telco Customer Churn dataset. SHAP is then used to understand which features influence the model's predictions globally and for an individual customer.

The main objective is to make machine learning predictions more interpretable by examining feature importance, individual prediction explanations, and model limitations.

## **2\. Objectives**

* Train a machine learning model for customer churn prediction.  
* Evaluate the model using multiple classification metrics.  
* Identify influential features using global SHAP importance.  
* Visualize feature contributions using a SHAP summary plot.  
* Explain an individual customer's prediction using a SHAP waterfall plot.  
* Discuss the limitations of model explanations and predictive performance.

## **3\. Dataset**

**Dataset:** IBM Telco Customer Churn

**Source:** [IBM Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

The dataset contains customer information, subscribed services, billing details, contract types, and churn labels.

### **Target variable**

* `Churn = Yes`: The customer left the service.  
* `Churn = No`: The customer remained with the service.

### **Data preprocessing**

The following preprocessing steps were applied:

1. Loaded the dataset using Pandas.  
2. Converted `TotalCharges` to a numeric data type.  
3. Removed duplicate records.  
4. Encoded the target variable as binary values.  
5. Excluded `customerID` from the input features.  
6. Identified numerical and categorical features.  
7. Applied median imputation to numerical features.  
8. Applied most-frequent imputation and one-hot encoding to categorical features.  
9. Split the data into training and testing sets using stratified sampling.  
10. Fitted preprocessing transformations on the training data to avoid data leakage.

The train-test split used a test size of 20% and `random_state=42`.

## **4\. Model and Tools**

### **Machine learning model**

**Gradient Boosting Classifier**

The classifier was trained using the following configuration:

* `n_estimators = 100`  
* `learning_rate = 0.1`  
* `max_depth = 3`  
* `random_state = 42`

### **Technologies used**

* Python  
* Pandas  
* NumPy  
* Matplotlib  
* Scikit-learn  
* SHAP  
* Google Colab

## **5\. Model Performance**

The model was evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.

| Metric | Result |
| ----- | ----- |
| Accuracy | 80.62% |
| Precision | 67.35% |
| Recall | 52.41% |
| F1-score | 58.95% |
| ROC-AUC | 0.8433 |

*These values are from the recorded test-set results.*

### **Classification report**

| Class | Precision | Recall | F1-score |
| ----- | ----- | ----- | ----- |
| No Churn | 0.84 | 0.91 | 0.87 |
| Churn | 0.67 | 0.52 | 0.59 |

### **Performance interpretation**

The model achieved an ROC-AUC of 0.8433, indicating good ability to distinguish between customers who churn and those who remain.

However, churn recall was 52.41%. This means the model identified approximately 52% of the actual churn cases at the selected classification threshold. Some customers who eventually churned were therefore classified as non-churners.

Accuracy alone does not fully describe performance on this imbalanced classification problem. Recall, precision, F1-score, and ROC-AUC provide additional information about the model's strengths and weaknesses.

## **6\. Explainability with SHAP**

SHAP was used to explain how individual input features contributed to the predictions of the trained Gradient Boosting classifier.

Two types of explanations were examined:

* **Global explanation:** Identifies features with the largest average absolute SHAP values across the selected test samples.  
* **Local explanation:** Shows how individual feature values contributed to a specific customer's prediction.

The explanations help clarify the model's behavior but do not establish causal relationships.

## **7\. Global Feature Importance**

Global feature importance was calculated using the mean absolute SHAP value across the selected 200 test samples.

### **Top influential features**

| Rank | Feature | Mean Absolute SHAP Value |
| ----- | ----- | ----- |
| 1 | Contract: Month-to-month | 0.66047 |
| 2 | Tenure | 0.37365 |
| 3 | Online Security: No | 0.23256 |
| 4 | Monthly Charges | 0.21363 |

### **Interpretation**

**Contract: Month-to-month** had the highest mean absolute SHAP value among the listed features, followed by tenure, online security status, and monthly charges.

These values measure the average magnitude of each feature's contribution to the model output. A larger absolute SHAP value indicates greater influence on the model's predictions within the analyzed sample.

Global importance does not indicate whether a feature increases or decreases churn predictions. The direction of a contribution must be examined using a SHAP summary plot or individual explanations.

## **8\. SHAP Summary Plot**

The SHAP summary (beeswarm) plot visualizes the distribution of feature contributions across the selected samples.

It helps investigate:

* Which features have the largest effects on model output.  
* How SHAP values vary between customers.  
* Whether high or low feature values tend to correspond to positive or negative contributions.  
* Differences in the direction and magnitude of contributions across observations.

Positive SHAP values move the explained model output toward a higher churn score, while negative values move it toward a lower churn score, assuming the explanation is for the positive churn class.

The summary plot describes learned model behavior rather than proving that a feature causes customer churn.

## **9\. Individual Customer Explanation**

A SHAP waterfall plot was used to explain one customer's prediction.

### **Prediction result**

| Item | Result |
| ----- | ----- |
| Predicted churn probability | 2.48% |
| Predicted class | No Churn |
| Actual class | No Churn |
| Raw model output | Approximately −3.672 |
| Expected/base raw output | Approximately −1.563 |

The prediction matched the actual label for this example. However, one correct prediction does not establish that the model is reliable for every customer.

### **Important feature contributions**

| Feature | Feature Value | SHAP Value |
| ----- | ----- | ----- |
| Tenure | 72 | −1.01655 |
| Contract: Month-to-month | 0 | −0.67312 |
| Contract: Two year | 1 | −0.39786 |
| Total Charges | 8468.20 | \+0.32184 |
| Monthly Charges | 114.05 | \+0.27949 |
| Online Security: No | 0 | −0.22543 |
| Tech Support: No | 0 | −0.17354 |
| Contract: One year | 0 | −0.13959 |
| Online Backup: No | 0 | −0.13251 |
| Internet Service: Fiber optic | 1 | \+0.11151 |

### **Interpretation**

The largest negative contribution came from tenure, followed by the month-to-month and two-year contract indicators. These contributions moved the model output toward a lower churn score for this customer.

Total charges and monthly charges had positive SHAP contributions, moving the output toward a higher churn score. The final prediction reflects the combined contributions of all features rather than any single feature in isolation.

**Technical note:** The waterfall plot's raw output was approximately −3.672, while the predicted churn probability was 2.48%. These values are consistent with a log-odds explanation, where the raw output is transformed into a probability. SHAP values in this explanation are not direct percentage-point changes in churn probability.

One-hot encoded categorical features must also be interpreted carefully. For example, a value of `0` for `Contract: Month-to-month` means that this particular indicator is inactive; its negative SHAP value describes its contribution in this specific prediction, not the general effect of having no month-to-month contract.

## **10\. Key Findings**

1. The Gradient Boosting classifier achieved an ROC-AUC of 0.8433 on the test set.  
2. The model identified non-churn customers more effectively than churn customers at the evaluated threshold.  
3. Churn recall of 52.41% highlights room for improvement in detecting customers likely to leave.  
4. Contract type and tenure had the largest global SHAP importance among the reported features.  
5. Monthly charges and online security status also contributed meaningfully to model predictions.  
6. SHAP provided both a dataset-level view of feature importance and an individual-level explanation.  
7. Feature contributions depend on the customer and the model; they should not be interpreted as causal effects.

## **11\. Limitations**

* The model's churn recall indicates that a substantial proportion of actual churn cases may be missed.  
* Global SHAP importance measures contribution magnitude, not the direction of an effect.  
* SHAP explanations describe the fitted model and do not prove causal relationships.  
* Correlated features and one-hot encoded categories can complicate interpretation.  
* Local explanations for a single customer may not represent the behavior of the model across the entire dataset.  
* The analysis used a selected subset of 200 test samples for SHAP explanations; results may vary with the sample.  
* The selected classification threshold may not be optimal for business requirements.  
* The dataset may not represent every telecom provider or customer population.

## **12\. Future Improvements**

Potential extensions include:

* Tune the classification threshold to improve churn recall.  
* Compare precision-recall trade-offs using a precision-recall curve.  
* Evaluate threshold-independent and threshold-dependent metrics together.  
* Compare SHAP explanations across multiple models.  
* Investigate explanation stability across different test samples.  
* Examine interactions between contract type, tenure, and monthly charges.  
* Evaluate performance on additional datasets or an external test set.  
* Develop a customer churn analysis dashboard for interactive explanations.

These are proposed future steps and have not been claimed as completed experiments.

## **13\. Project Files**

Recommended repository structure:

04-explainable-ml-with-shap/  
├── 04-explainable-ml-with-shap.ipynb  
├── README.md  
├── shap\_feature\_importance.csv  
└── shap\_customer\_explanation.csv

The CSV files contain the exported global feature importance and individual customer explanation results. Include them in the repository if they are available and you intend to share them.

## **14\. Conclusion**

This project demonstrates how SHAP can improve the interpretability of a machine learning model for customer churn prediction. The Gradient Boosting classifier achieved an ROC-AUC of 0.8433, while SHAP identified contract type, tenure, online security status, and monthly charges as influential features in the analyzed sample.

The individual waterfall explanation showed how different feature contributions combined to produce a low predicted churn probability for one customer. At the same time, the model's churn recall indicates that further evaluation and threshold optimization may be needed before practical deployment.

Overall, the project combines predictive modeling with explainability and highlights why machine learning systems should be assessed not only by predictive performance but also by interpretability and limitations.

---

**Project:** 04 — Explainable Machine Learning Using SHAP  
**Domain:** Machine Learning, Explainable AI (XAI), Customer Analytics  
**Environment:** Google Colab  
**Status:** Completed initial model evaluation and SHAP analysis

## **15\. Author**

**Md. Munna Mia**

CSE Undergraduate \| AI/ML and Cybersecurity

* GitHub: [https\://github.com/munnamia1945](https://github.com/munnamia1945)  
* Portfolio: [https\://www\.munnamia.com](https://www.munnamia.com)

