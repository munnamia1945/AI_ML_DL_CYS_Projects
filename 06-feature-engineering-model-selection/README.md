# **Project 06: Feature Engineering & Model Selection**

A practical machine learning project demonstrating missing-value imputation, categorical encoding, feature scaling, feature engineering, cross-validation, model selection, and data leakage prevention using Scikit-learn.

## **Project Overview**

Feature engineering and model selection are essential steps in building reliable machine learning systems. This project implements a complete classification workflow using the Breast Cancer Wisconsin dataset.

The workflow compares multiple classification algorithms using stratified 5-fold cross-validation, selects the best-performing model based on mean F1-score, and evaluates the selected model on a held-out test set.

### **Objectives**

* Handle missing numerical and categorical values.  
* Create additional features from existing data.  
* Encode categorical variables using One-Hot Encoding.  
* Standardize numerical features using `StandardScaler`.  
* Prevent preprocessing leakage with `Pipeline` and `ColumnTransformer`.  
* Compare classification models using stratified 5-fold cross-validation.  
* Select a model using mean cross-validation F1-score.  
* Evaluate the selected model on unseen test data.

## **Dataset**

**Dataset:** Breast Cancer Wisconsin dataset  
**Source:** Scikit-learn built-in dataset  
**Task:** Binary classification

The original dataset contains 569 samples and 30 numerical features.

| Dataset property | Value |
| ----- | ----- |
| Original samples | 569 |
| Original numerical features | 30 |
| Training samples | 455 |
| Test samples | 114 |
| Classification classes | 2 |

The target was encoded as:

* `0` — Benign  
* `1` — Malignant

An 80/20 stratified train-test split was used with `random_state=42`.

### **Simulated Missing Values**

The original dataset was modified for educational purposes by introducing missing values into selected numerical and categorical features. These missing values were simulated rather than discovered in the original dataset.

The project also created two additional features:

* `radius_texture_ratio` — the ratio of mean radius to mean texture.  
* `radius_group` — a categorical feature derived from mean radius, with Small, Medium, and Large groups.

## **Technologies Used**

* Python  
* NumPy  
* pandas  
* Matplotlib  
* Seaborn  
* Scikit-learn  
* Joblib

## **Machine Learning Workflow**

1. Load and explore the dataset.  
2. Engineer additional features.  
3. Simulate missing values for preprocessing practice.  
4. Separate the features and target.  
5. Split the data into training and test sets.  
6. Build numerical and categorical preprocessing pipelines.  
7. Compare candidate classifiers using cross-validation.  
8. Select the model with the highest mean cross-validation F1-score.  
9. Evaluate the selected model on the held-out test set.  
10. Export evaluation results and save the fitted pipeline.

## **Feature Engineering and Preprocessing**

### **1\. Missing Value Imputation**

Numerical features were imputed using the median, while categorical features were imputed using the most frequent value.

### **2\. Categorical Encoding**

`OneHotEncoder(handle_unknown="ignore")` was used to transform categorical variables into numerical representations while handling previously unseen categories safely.

### **3\. Feature Scaling**

`StandardScaler` was applied to numerical features. Scaling is particularly relevant for distance-based and margin-based algorithms, such as K-Nearest Neighbors, Support Vector Machine, and Logistic Regression.

### **4\. Pipeline Architecture**

The preprocessing and classification stages were combined using Scikit-learn pipelines.

* Numerical pipeline: median imputation → standardization.  
* Categorical pipeline: most-frequent imputation → one-hot encoding.  
* Full pipeline: preprocessing → classifier.

This design ensures that preprocessing parameters are learned from the appropriate training data rather than from the held-out test set.

## **Models Compared**

The project evaluated five classification algorithms:

1. Logistic Regression  
2. K-Nearest Neighbors  
3. Support Vector Machine  
4. Decision Tree  
5. Random Forest

**Model-selection criterion:** Highest mean F1-score across stratified 5-fold cross-validation on the training data.

## **Cross-Validation Results**

The following table contains the cross-validation results available from the extracted notebook output. Results for K-Nearest Neighbors, Decision Tree, and Random Forest were not available in the extracted text and are therefore not reported here.

| Model | Mean CV Accuracy | Accuracy Std. | Mean CV Precision | Mean CV Recall | Mean CV F1 | F1 Std. | Mean CV ROC-AUC |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Support Vector Machine | 0.9780 | 0.0098 | 0.9823 | 0.9588 | 0.9700 | 0.0138 | 0.9951 |
| Logistic Regression | 0.9758 | 0.0108 | 0.9820 | 0.9529 | 0.9671 | 0.0150 | 0.9956 |

The Support Vector Machine achieved the highest mean F1-score among the available results and was selected as the best model.

## **Selected Model**

**Support Vector Machine (SVM)**

| Metric | Result |
| ----- | ----- |
| Mean CV F1-score | 0.9700 |
| Mean CV Accuracy | 0.9780 |
| Mean CV Precision | 0.9823 |
| Mean CV Recall | 0.9588 |
| Mean CV ROC-AUC | 0.9951 |

These values are cross-validation results, not held-out test-set results.

## **Test-Set Evaluation**

The selected SVM model was evaluated on the separate test set containing 114 samples.

| Metric | Test Result |
| ----- | ----- |
| Accuracy | 0.96 |
| Macro-average Precision | 0.95 |
| Macro-average Recall | 0.95 |
| Macro-average F1-score | 0.96 |
| Weighted-average Precision | 0.96 |
| Weighted-average Recall | 0.96 |
| Weighted-average F1-score | 0.96 |

The classification report showed the following class-level results:

| Class | Precision | Recall | F1-score | Support |
| ----- | ----- | ----- | ----- | ----- |
| Benign | 0.95 | 0.99 | 0.97 | 72 |
| Malignant | 0.97 | 0.90 | 0.94 | 42 |

Values are rounded as reported in the extracted classification report.

### **Interpretation**

* **Benign:** The model achieved a recall of 0.99 for benign samples.  
* **Malignant:** The model achieved a precision of 0.97 and a recall of 0.90 for malignant samples.  
* **Overall:** The reported test accuracy was approximately 0.96.

The malignant-class recall means that some malignant cases were missed by the model. This is an important consideration when interpreting a medical classification task. These experimental results are educational and do not establish clinical suitability.

The raw extracted test-metrics output was garbled, so a separate test ROC-AUC value is intentionally not reported.

## **Data Leakage Prevention**

Preventing data leakage was a central objective of this project.

The following practices were used:

* The test set was separated before cross-validation and model selection.  
* Imputation, encoding, and scaling were included inside the preprocessing pipeline.  
* Cross-validation was performed on the training set only.  
* Preprocessing was fitted independently within each cross-validation training fold.  
* The final selected pipeline was fitted on the full training set and evaluated on the held-out test set.

### **Transformation Verification**

| Check | Result |
| ----- | ----- |
| Transformed training shape | `(455, 34)` |
| All transformed values are finite | `True` |

The transformed shape confirms that the pipeline produced 34 processed features for the 455 training samples. The finite-value check confirmed that the transformed training matrix contained no NaN or infinite values.

## **Visualizations**

The notebook includes:

* Target class distribution.  
* Missing-value percentage by feature.  
* Cross-validation F1-score comparison.  
* Confusion matrix for the selected model.  
* ROC curve for the selected model.

These visualizations help examine class distribution, preprocessing, comparative performance, and classification behavior.

## **Exported Files**

The notebook generates the following files:

| File | Description |
| ----- | ----- |
| `feature_engineering_cv_results.csv` | Cross-validation comparison results |
| `feature_engineering_test_metrics.csv` | Selected model's test metrics |
| `feature_engineering_classification_report.txt` | Classification report |
| `feature_engineering_best_pipeline.joblib` | Serialized preprocessing and classification pipeline |

The exported metrics should be retained with the notebook to support reproducibility and transparent reporting.

## **Limitations**

* Missing values were artificially introduced for demonstration; they were not present in the original dataset in the form used by this experiment.  
* The engineered categorical feature was derived from an existing numerical feature.  
* The extracted cross-validation table was incomplete, so the available results do not establish the complete ranking of all five models.  
* The test ROC-AUC result could not be reliably recovered from the extracted output.  
* The dataset is a benchmark dataset and the model is not validated for clinical use.

## **Future Improvements**

* Recover and document the complete cross-validation results for all five models.  
* Add hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`.  
* Compare additional feature-selection methods.  
* Investigate model calibration and classification thresholds.  
* Evaluate the workflow on additional datasets.  
* Add reproducibility checks and experiment tracking.

## **Key Takeaways**

This project demonstrates how a structured preprocessing pipeline can make machine learning experiments more consistent and reduce the risk of data leakage.

The SVM model achieved a mean cross-validation F1-score of 0.9700 among the available reported results and a reported test accuracy of 0.96. The experiment also highlights why cross-validation and held-out test results must be reported separately.

##  **Author**

**Md. Munna Mia**

CSE Undergraduate \| AI/ML and Cybersecurity

* GitHub: [https\://github.com/munnamia1945](https://github.com/munnamia1945)  
* Portfolio: [https\://www\.munnamia.com](https://www.munnamia.com)

