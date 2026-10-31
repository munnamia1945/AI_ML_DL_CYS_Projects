# **Project 12: ML-Based Network Intrusion Detection Using CICIDS2017**

An experimental machine learning-based Network Intrusion Detection System (NIDS) using the CICIDS2017 dataset. This project evaluates four tree-based ensemble algorithms for both binary and multiclass network traffic classification.

The primary objective is to distinguish benign network traffic from malicious activity and investigate how effectively machine learning models identify individual attack categories.

## **Project Overview**

This project investigates two classification tasks:

* **Binary Classification:** Classifies network traffic as `BENIGN` or `ATTACK`.  
* **Multiclass Classification:** Identifies 15 traffic categories, including benign traffic and 14 attack categories.

Four machine learning algorithms are evaluated:

1. Random Forest  
2. XGBoost  
3. LightGBM  
4. CatBoost

## **Dataset**

**Dataset:** CICIDS2017  
**Source:** Canadian Institute for Cybersecurity, University of New Brunswick.

The experiment combines eight CICIDS2017 CSV files covering different days and attack scenarios.

| Dataset Information | Value |
| ----- | ----- |
| CSV files loaded | 8 |
| Rows before cleaning | 2,830,743 |
| Exact duplicate rows removed | 308,381 |
| Rows after cleaning | 2,522,362 |
| Numeric features used | 70 |
| Classification categories | 15 |
| Training samples | 2,017,889 |
| Testing samples | 504,473 |

### **Class Distribution**

The cleaned dataset contains one benign category and 14 attack categories.

| Class | Samples |
| ----- | ----- |
| BENIGN | 2,096,484 |
| DoS Hulk | 172,849 |
| DDoS | 128,016 |
| PortScan | 90,819 |
| DoS GoldenEye | 10,286 |
| FTP-Patator | 5,933 |
| DoS slowloris | 5,385 |
| DoS Slowhttptest | 5,228 |
| SSH-Patator | 3,219 |
| Bot | 1,953 |
| Web Attack Brute Force | 1,470 |
| Web Attack XSS | 652 |
| Infiltration | 36 |
| Web Attack Sql Injection | 21 |
| Heartbleed | 11 |

The dataset exhibits substantial class imbalance, particularly among less frequent attack categories.

## **Technologies Used**

* Python  
* Google Colab  
* Pandas  
* NumPy  
* Scikit-learn  
* XGBoost  
* LightGBM  
* CatBoost  
* Matplotlib  
* Joblib

## **Experimental Workflow**

1. Load and combine eight CICIDS2017 CSV files.  
2. Clean target labels and remove empty labels.  
3. Identify and remove exact duplicate records.  
4. Convert infinite values to missing values.  
5. Select numeric network traffic features.  
6. Exclude identifiers, metadata, entirely missing columns, and constant features.  
7. Create separate binary and multiclass targets.  
8. Apply stratified 80/20 train-test splitting.  
9. Apply median imputation through a training-fitted pipeline.  
10. Calculate balanced sample weights using training labels.  
11. Train and evaluate four machine learning models.  
12. Compare performance using classification metrics and confusion matrices.  
13. Analyze feature importance and prediction errors.  
14. Save trained models and experimental reports.

### **Evaluation Metrics**

* Accuracy  
* Macro Precision  
* Macro Recall  
* Macro F1-score  
* Macro False Positive Rate (FPR)  
* Confusion Matrix  
* Per-class Classification Report  
* Training Time  
* Inference Time

## **Binary Classification Results**

Binary classification distinguishes benign traffic from malicious traffic.

| Model | Accuracy | Macro-F1 | Training Time |
| ----- | ----- | ----- | ----- |
| XGBoost | 0.9990 | 0.9983 | 53.4 s |
| LightGBM | 0.9990 | 0.9982 | 40.2 s |
| CatBoost | 0.9989 | 0.9980 | 70.7 s |
| Random Forest | 0.9986 | 0.9975 | 442.1 s |

### **Best Binary Model: XGBoost**

XGBoost achieved the highest Macro-F1 score of 0.9983, while sharing the highest reported accuracy of 0.9990 with LightGBM.

Its reported macro-FPR was 0.0007, tied with LightGBM.

**XGBoost test-set error analysis:**

| Error Type | Count |
| ----- | ----- |
| False Positives | 515 |
| False Negatives | 33 |
| Total Incorrect Predictions | 548 |

These results indicate strong binary classification performance under the experimental split.

## **Multiclass Classification Results**

Multiclass classification evaluates the models' ability to distinguish benign traffic and individual attack categories.

| Model | Accuracy | Macro-F1 | Training Time |
| ----- | ----- | ----- | ----- |
| XGBoost | 0.9986 | 0.9077 | 420.0 s |
| Random Forest | 0.9983 | 0.8980 | 431.2 s |
| CatBoost | 0.9882 | 0.6920 | 1591.2 s |
| LightGBM | 0.4934 | 0.2122 | 231.8 s |

### **Best Multiclass Model: XGBoost**

XGBoost achieved the highest reported accuracy (0.9986) and Macro-F1 (0.9077) among the evaluated models.

However, the difference between accuracy and Macro-F1 highlights the difficulty of detecting minority attack categories in an imbalanced dataset.

### **Important Classification Findings**

* Heartbleed and SSH-Patator achieved reported F1-scores of 1.0000 with XGBoost.  
* Web Attack XSS was among the most difficult categories, with an F1-score of 0.4615.  
* Web Attack Brute Force achieved a recall of 0.7449.  
* Web Attack Brute Force was frequently confused with BENIGN traffic and Web Attack XSS.  
* LightGBM showed substantially weaker multiclass performance, with a Macro-F1 of 0.2122.

These findings demonstrate that strong binary detection performance does not necessarily guarantee reliable identification of individual attack categories.

## **Feature Importance**

Feature importance was extracted using the models' native `feature_importances_` attributes.

Some of the most important features identified in the experiments were:

**Random Forest — Binary Classification**

* Average Packet Size  
* Bwd Packet Length Std  
* Packet Length Variance  
* Bwd Packet Length Mean

**XGBoost — Binary Classification**

* Bwd Packet Length Std  
* Average Packet Size  
* Avg Bwd Segment Size  
* Bwd Header Length

Packet-length statistics appeared prominently among the important features across the reported models.

Feature importance indicates model reliance, not causal relationships between individual network features and attacks.

## **Key Findings**

1. All four models achieved high binary classification performance, with Macro-F1 scores above 0.997.  
2. XGBoost achieved the strongest reported results in both binary and multiclass experiments.  
3. Multiclass classification was substantially more challenging than binary classification.  
4. Minority attack categories, particularly web attacks, remained difficult to distinguish.  
5. LightGBM experienced a significant performance drop in the multiclass experiment.  
6. Class imbalance and the distinction between similar attack patterns require further investigation.

## **Generated Outputs**

The experiment generates model files and analysis artifacts in the `Project12_Outputs` directory.

| Artifact | Description |
| ----- | ----- |
| `source_file_audit.csv` | Loaded dataset file and row-count audit |
| `class_distribution.csv` | Class counts and percentages |
| `class_distribution.png` | Class distribution visualization |
| `binary_model_comparison.csv` | Binary model performance comparison |
| `binary_model_comparison.png` | Binary performance visualization |
| `binary_xgboost_classification_report.csv` | Binary XGBoost classification report |
| `multiclass_model_comparison.csv` | Multiclass model performance comparison |
| `multiclass_model_comparison.png` | Multiclass performance visualization |
| `best_binary_intrusion_model.joblib` | Selected binary classification model |
| `best_multiclass_intrusion_model.joblib` | Selected multiclass classification model |

Additional confusion matrices, classification reports, prediction files, feature importance reports, encoders, and other artifacts are generated by the notebook.

## **Limitations**

* **Temporal Leakage Risk:** The random stratified split may allow flows from the same capture periods to appear in both training and testing sets. Therefore, these results should be treated as a baseline rather than proof of temporal generalization.  
* **Class Imbalance:** Minority attack categories have very few samples, making reliable multiclass detection difficult.  
* **Model Performance Variation:** LightGBM showed a substantial performance decline in multiclass classification despite balanced sample weighting.  
* **External Validation:** Cross-dataset evaluation using an independent dataset such as UNSW-NB15 has not yet been performed.  
* **Live Traffic Evaluation:** The project has not been tested on live network traffic.

## **Future Improvements**

* Investigate capture-aware and time-based evaluation strategies.  
* Perform cross-dataset validation using additional intrusion detection datasets.  
* Improve minority attack detection through class-specific analysis and imbalance-handling techniques.  
* Investigate feature selection and hyperparameter optimization.  
* Explore deep learning approaches for comparison with traditional machine learning models.  
* Develop an offline prediction dashboard for CSV-based network traffic analysis.

## **How to Run**

1. Open the project notebook in Google Colab.  
2. Download the CICIDS2017 machine-learning CSV files from the official dataset source.  
3. Place the CSV files in the configured Google Drive dataset directory.  
4. Execute the notebook cells in order.  
5. Review the generated reports and trained models in `Project12_Outputs`.

**Dataset Source:**  
[https\://www\.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)

## **Disclaimer**

This project is an academic machine learning experiment using the CICIDS2017 dataset. Its reported performance reflects the specific dataset, preprocessing pipeline, and train-test split used in this experiment. It does not establish production-level intrusion detection capability or real-time network monitoring.

## **Author**

**Md Munna Mia**

* GitHub: [https\://github.com/munnamia1945](https://github.com/munnamia1945)  
* Portfolio: [https\://www\.munnamia.com](https://www.munnamia.com/)

---

*Project 12 — Machine Learning-Based Network Intrusion Detection Using CICIDS2017.*

