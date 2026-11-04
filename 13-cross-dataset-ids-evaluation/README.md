# **Cross-Dataset Intrusion Detection Evaluation**

### **Feature Compatibility, Label Mapping, and Generalization Between CICIDS2017 and UNSW-NB15**

## **1\. Project Overview**

Machine learning models can achieve strong performance when training and testing data originate from the same network intrusion detection dataset. However, their performance may decrease substantially when evaluated on traffic collected from a different environment or dataset.

This project investigates the cross-dataset generalization capability of machine learning-based Network Intrusion Detection Systems (NIDS) using two widely used benchmark datasets:

* CICIDS2017  
* UNSW-NB15

The study examines whether models trained on one dataset can effectively detect malicious network traffic in another dataset using a harmonized five-feature representation.

Rather than focusing only on within-dataset classification accuracy, this project evaluates both directions of cross-dataset transfer, investigates feature compatibility, applies leakage-aware splitting, and examines dataset-origin separability and multi-seed performance stability.

**Central research question:**

> To what extent can machine learning-based intrusion detection models trained on one network traffic dataset generalize to another dataset, and how do feature compatibility and dataset-specific characteristics affect their performance?

---

## **2\. Research Objectives**

The project has five primary objectives:

1. Evaluate intrusion detection models independently on CICIDS2017 and UNSW-NB15.  
2. Audit feature compatibility between the two datasets.  
3. Develop a common five-feature representation for cross-dataset experiments.  
4. Measure cross-dataset generalization in both transfer directions.  
5. Investigate dataset-origin separability, class imbalance, sampling limitations, and model stability.

---

## **3\. Datasets**

### **3.1 CICIDS2017**

CICIDS2017 is a network intrusion detection benchmark containing benign traffic and multiple categories of malicious network activity.

In this project, the original traffic records are cleaned, duplicate records are removed, and a common feature representation is constructed.

Dataset source:

* [Canadian Institute for Cybersecurity — CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

### **3.2 UNSW-NB15**

UNSW-NB15 is another network intrusion detection benchmark containing normal traffic and several categories of network attacks.

The official training and testing partitions are retained for the experiments.

Dataset source:

* [UNSW Research — UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

### **3.3 Dataset Preparation**

Both datasets are prepared using a binary classification target:

| Label | Meaning |
| ----- | ----- |
| 0 | Benign / Normal |
| 1 | Attack |

The original attack-category information from CICIDS2017 is also retained separately for sampling diagnostics. It is not used as a model input.

**Important:** Binary label mapping simplifies the cross-dataset comparison but does not preserve the full attack-category structure of either dataset.

---

## **4\. Feature Compatibility and Harmonization**

One of the main challenges in cross-dataset evaluation is that datasets use different feature names, formats, and measurement conventions.

This project uses five candidate common features.

| Common Feature | CICIDS2017 Feature | UNSW-NB15 Feature |
| ----- | ----- | ----- |
| `duration` | Flow Duration | `dur` |
| `forward_packets` | Total Fwd Packets | `spkts` |
| `backward_packets` | Total Backward Packets | `dpkts` |
| `forward_bytes` | Total Length of Fwd Packets | `sbytes` |
| `backward_bytes` | Total Length of Bwd Packets | `dbytes` |

### **Duration Conversion**

CICIDS2017 records flow duration in microseconds, whereas the UNSW-NB15 `dur` feature is expressed in seconds.

Therefore, CICIDS2017 duration is converted into seconds:

```
cic_data["duration"] = cic_data["duration"] / 1_000_000
```

### **Feature Compatibility Consideration**

The feature names and broad concepts are aligned, but matching names or concepts do not guarantee identical statistical distributions or measurement semantics.

The five-feature representation should therefore be interpreted as a practical harmonization strategy, not proof that both datasets describe identical traffic distributions.

---

## **5\. Data Cleaning and Preprocessing**

The preprocessing workflow includes the following operations.

### **CICIDS2017**

* Load and combine the available CSV files.  
* Normalize column names and labels.  
* Create binary labels.  
* Remove exact duplicate source records before splitting.  
* Convert selected features into numeric values.  
* Convert flow duration from microseconds to seconds.  
* Remove invalid labels and infinite values.  
* Remove records with negative values in the selected features.  
* Audit repeated five-feature signatures and conflicting labels.

### **UNSW-NB15**

* Load the official training and testing partitions.  
* Map the selected features to the common representation.  
* Convert selected columns into numeric values.  
* Remove infinite values and invalid labels.  
* Remove records containing negative values in the selected features.

### **Model Preprocessing Pipeline**

Each classifier is evaluated using a pipeline:

```
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    (
        "log_transform",
        FunctionTransformer(np.log1p)
    ),
    ("classifier", model)
])
```

The median imputer and log transformation are fitted within the training pipeline.

This ensures that preprocessing parameters are learned from training data rather than from the target test data.

---

## **6\. Leakage-Aware Experimental Splitting**

A key methodological concern is the possibility that identical model-input feature combinations appear in both training and testing partitions.

This can inflate evaluation performance and produce misleading estimates of generalization.

### **CICIDS2017 Split**

The project applies the following procedure:

1. Create a stratified working sample of 250,000 records.  
2. Group records using their five selected model-input features.  
3. Apply `GroupShuffleSplit` with a 20% test-size target.  
4. Verify that identical five-feature signatures do not overlap between training and testing partitions.

The grouping operation keeps identical feature signatures in the same partition, including cases where identical signatures have conflicting labels.

### **UNSW-NB15 Split**

The official UNSW-NB15 training and testing partitions are retained.

### **Leakage Audit**

The notebook explicitly checks whether identical five-feature signatures occur across the CICIDS2017 training and testing partitions.

The split is rejected if signature overlap is detected.

**Important distinction:** This procedure reduces leakage from repeated model-input signatures. It does not establish that every possible form of dataset, temporal, or source-level leakage has been eliminated.

---

## **7\. Machine Learning Models**

Four tree-based machine learning classifiers are evaluated.

| Model | Implementation |
| ----- | ----- |
| Random Forest | Scikit-learn |
| XGBoost | XGBoost |
| LightGBM | LightGBM |
| CatBoost | CatBoost |

The notebook uses 150 estimators or iterations for the four model configurations, with model-specific parameters and class-imbalance handling.

Class weighting or positive-class weighting is used where configured.

The same four model families are evaluated across all four experiments.

---

## **8\. Experimental Design**

The project contains four main experiments.

| Experiment | Training Dataset | Testing Dataset | Evaluation Type |
| ----- | ----- | ----- | ----- |
| EXP-01 | CICIDS2017 | CICIDS2017 | Within-dataset |
| EXP-02 | UNSW-NB15 | UNSW-NB15 | Within-dataset |
| EXP-03 | CICIDS2017 | UNSW-NB15 | Cross-dataset |
| EXP-04 | UNSW-NB15 | CICIDS2017 | Cross-dataset |

### **EXP-01: CICIDS2017 → CICIDS2017**

Evaluates the models using CICIDS2017 training and testing partitions.

### **EXP-02: UNSW-NB15 → UNSW-NB15**

Evaluates the models using the official UNSW-NB15 training and testing partitions.

### **EXP-03: CICIDS2017 → UNSW-NB15**

Trains models using CICIDS2017 training data and evaluates them on the UNSW-NB15 test partition.

This experiment investigates whether knowledge learned from CICIDS2017 transfers to UNSW-NB15.

### **EXP-04: UNSW-NB15 → CICIDS2017**

Trains models using UNSW-NB15 training data and evaluates them on the CICIDS2017 test partition.

This experiment investigates generalization in the opposite direction.

---

## **9\. Evaluation Metrics**

The project evaluates model performance using:

| Metric | Purpose |
| ----- | ----- |
| Accuracy | Overall classification correctness |
| Macro Precision | Average precision across both classes |
| Macro Recall | Average recall across both classes |
| Macro F1-score | Balanced F1 evaluation across both classes |
| False Positive Rate (FPR) | Proportion of benign samples incorrectly classified as attacks |
| Confusion Matrix | Detailed classification outcomes |
| Training Time | Time required to fit the model |
| Inference Time | Time required to generate predictions |

The False Positive Rate is calculated as:

\[  
FPR \= \\frac{FP}{FP+TN}  
\]

Macro-F1 is particularly important because it gives equal importance to both classes, rather than allowing the majority class to dominate the evaluation.

---

## **10\. Multi-Seed Robustness Evaluation**

To investigate sensitivity to random sampling and splitting, the project includes an additional multi-seed evaluation in Cell 48\.

### **Experimental Configuration**

Five random seeds are used:

```
ROBUSTNESS_SEEDS = [42, 7, 21, 100, 2026]
```

The evaluation repeats all four experiments for each seed.

| Configuration | Value |
| ----- | ----- |
| Random seeds | 5 |
| Experiments per seed | 4 |
| Models per experiment | 4 |
| Total model evaluations | 80 |

For each seed:

* A new stratified 250,000-row CICIDS2017 working sample is created.  
* A new group-aware CICIDS2017 split is generated.  
* Feature-signature overlap is checked.  
* The four experiments are repeated.  
* Accuracy, Macro Precision, Macro Recall, Macro-F1, FPR, confusion-matrix counts, training time, and test sample count are recorded.

The official UNSW-NB15 training and testing partitions remain fixed across seeds.

**Interpretation note:** This is a seed-sensitivity analysis, not five independent resamplings of the UNSW-NB15 official test set. The reported standard deviations should not be interpreted as independent test-set uncertainty estimates.

### **Multi-Seed Macro-F1 Results**

Values are reported as mean ± standard deviation.

| Experiment | Random Forest | XGBoost | LightGBM | CatBoost |
| ----- | ----- | ----- | ----- | ----- |
| EXP-01 CIC → CIC | 0.990936 ± 0.000546 | 0.990452 ± 0.000904 | 0.990125 ± 0.000895 | 0.985159 ± 0.000767 |
| EXP-02 UNSW → UNSW | 0.871588 ± 0.000407 | 0.899955 ± 0.000642 | 0.899541 ± 0.000000 | 0.884968 ± 0.002241 |
| EXP-03 CIC → UNSW | 0.310059 ± 0.000000 | 0.309999 ± 0.001049 | 0.310931 ± 0.000707 | 0.311004 ± 0.001104 |
| EXP-04 UNSW → CIC | 0.352568 ± 0.022158 | 0.375148 ± 0.021608 | 0.436366 ± 0.001886 | 0.345778 ± 0.020597 |

### **Key Observations**

1. All four models achieve very high Macro-F1 scores in the CICIDS2017 within-dataset experiment.  
2. XGBoost and LightGBM achieve the strongest Macro-F1 results in the UNSW-NB15 within-dataset experiment.  
3. All models experience a substantial performance reduction when transferred from CICIDS2017 to UNSW-NB15.  
4. UNSW-NB15 to CICIDS2017 transfer performs somewhat better than the opposite direction for these experiments.  
5. The low standard deviations in some experiments indicate stable results under the tested seed configurations, but they do not establish broad generalization across independent datasets.

---

## **11\. Cross-Dataset Performance Analysis**

The central finding is the difference between within-dataset and cross-dataset performance.

### **Within-Dataset Performance**

The multi-seed results show:

* CICIDS2017 → CICIDS2017 achieves approximately 0.985–0.991 Macro-F1 across the four models.  
* UNSW-NB15 → UNSW-NB15 achieves approximately 0.872–0.900 Macro-F1.

These results indicate strong performance when training and testing data originate from the same benchmark dataset.

### **Cross-Dataset Performance**

The cross-dataset results are considerably lower:

* CICIDS2017 → UNSW-NB15: approximately 0.310 Macro-F1.  
* UNSW-NB15 → CICIDS2017: approximately 0.346–0.436 Macro-F1.

This demonstrates that high within-dataset performance does not automatically translate into cross-dataset generalization.

### **Important Interpretation**

The observed performance reduction is consistent with substantial differences between the two benchmark datasets.

However, the current experiments do not isolate the individual causal contribution of feature distributions, traffic-generation environments, attack composition, labeling conventions, or other dataset-specific factors.

---

## **12\. Dataset-Origin Separability Analysis**

Cell 47 introduces an additional experiment to investigate whether CICIDS2017 and UNSW-NB15 can be distinguished using the same five harmonized features.

### **Method**

* Sample 175,000 records from each dataset.  
* Construct a balanced dataset-origin classification problem.  
* Assign CICIDS2017 the origin label 0\.  
* Assign UNSW-NB15 the origin label 1\.  
* Group identical five-feature signatures.  
* Apply a group-aware train/test split.  
* Train a LightGBM classifier.  
* Evaluate dataset-origin classification using ROC-AUC and accuracy.

### **Reported Results**

| Metric | Result |
| ----- | ----- |
| Balanced sample per dataset | 175,000 |
| Training rows | 244,482 |
| Testing rows | 105,518 |
| Test CICIDS2017 samples | 52,042 |
| Test UNSW-NB15 samples | 53,476 |
| ROC-AUC | 0.999926 |
| Accuracy | 0.997356 |
| Total classification errors | 279 |

Confusion matrix:

```
[[51899,   143],
 [  136, 53340]]
```

### **Interpretation**

The very high ROC-AUC indicates that the five selected features contain substantial information about dataset origin.

In other words, a classifier can distinguish the two benchmark datasets with very high accuracy using the harmonized feature representation.

This finding supports the observation that the datasets are highly separable in the selected feature space.

However, dataset-origin separability alone does not identify a specific distribution-shift mechanism or establish causality. Further feature-distribution and domain-adaptation analyses would be needed.

---

## **13\. CICIDS2017 Sampling and Rare Attack-Category Validation**

Cell 46 examines whether the 250,000-row CICIDS2017 working sample retains the attack categories present in the full cleaned dataset.

### **Reported Sampling Information**

| Item | Result |
| ----- | ----- |
| Full cleaned CICIDS2017 rows | 2,522,255 |
| Working sample rows | 250,000 |
| Attack categories in full cleaned data | 15 |
| Categories absent from working sample | 1 |

The absent category is:

```
Heartbleed
```

The full cleaned dataset contains only 11 Heartbleed records, and none of them appears in the 250,000-row working sample.

### **Interpretation**

This sampling limitation is important when interpreting the CICIDS2017 experiments.

The model evaluation cannot provide meaningful category-specific performance evidence for an attack category that is absent from the working sample.

Although the primary experiments use binary labels, the absence of a rare attack category limits the diversity of attack examples represented in the sampled data.

---

## **14\. Main Findings**

The experiments support the following observations:

**Finding 1 — Strong within-dataset performance**

The evaluated models perform strongly when training and testing on the same dataset.

**Finding 2 — Weak cross-dataset generalization**

All four models show substantial performance degradation when transferred between CICIDS2017 and UNSW-NB15.

**Finding 3 — Asymmetric transfer performance**

UNSW-NB15 → CICIDS2017 produces somewhat higher Macro-F1 scores than CICIDS2017 → UNSW-NB15 in the multi-seed results.

**Finding 4 — Strong dataset-origin separability**

The dataset-origin classifier achieves ROC-AUC 0.999926, demonstrating that the selected feature representation strongly distinguishes the two datasets.

**Finding 5 — Rare-category sampling limitation**

The CICIDS2017 working sample excludes Heartbleed, which has only 11 records in the full cleaned dataset.

**Finding 6 — Seed sensitivity is not the same as external validation**

Five-seed evaluation provides useful evidence about sensitivity to the tested random configurations, but it does not replace evaluation on additional independent datasets.

---

## **15\. Limitations**

The current study has several limitations that should be considered when interpreting the findings.

1. **Limited dataset coverage:** Only CICIDS2017 and UNSW-NB15 are evaluated.  
2. **Five-feature representation:** The harmonized feature space contains only five selected network-flow features.  
3. **Binary classification:** The original attack categories are collapsed into Normal and Attack labels for the main experiments.  
4. **Semantic compatibility:** Similar feature names and concepts do not guarantee identical measurement semantics.  
5. **Rare attack representation:** Heartbleed is absent from the CICIDS2017 working sample.  
6. **Sampling limitation:** The main CICIDS2017 experiment uses a 250,000-row working sample rather than the full cleaned dataset.  
7. **Dataset-origin analysis:** High dataset-origin separability does not independently prove the cause of cross-dataset performance degradation.  
8. **Limited seed interpretation:** The official UNSW-NB15 split remains fixed across all five seeds.  
9. **Model scope:** The current experiments evaluate four tree-based machine learning models; they do not establish the performance of deep learning or domain-adaptation methods.  
10. **No operational deployment claim:** The experiments evaluate offline benchmark datasets and do not demonstrate live network monitoring or production-ready intrusion prevention.

These limitations define important directions for future investigation.

---

## **16\. Future Research Directions**

Potential extensions include:

* Evaluating additional benchmark datasets such as CSE-CIC-IDS2018, ToN-IoT, and BoT-IoT.  
* Investigating a broader set of semantically compatible network-flow features.  
* Conducting detailed feature-distribution and domain-shift analysis.  
* Evaluating domain adaptation and transfer learning techniques.  
* Investigating domain generalization methods.  
* Evaluating deep learning-based intrusion detection models.  
* Studying rare and previously unseen attack categories.  
* Performing more comprehensive leakage and duplicate-signature audits.  
* Testing models on additional independent datasets and evaluation environments.

These are proposed research directions, not completed experiments in the current notebook.

---

## **17\. Technologies and Libraries**

| Technology | Purpose |
| ----- | ----- |
| Python | Main programming language |
| Google Colab | Experimental environment |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, splitting, pipelines, and evaluation |
| XGBoost | Gradient boosting classifier |
| LightGBM | Gradient boosting classifier |
| CatBoost | Gradient boosting classifier |
| Matplotlib | Visualization |
| Joblib | Model serialization |

---

## **18\. Repository Structure**

Suggested project organization:

```
project-13-cross-dataset-ids/
│
├── README.md
│
├── notebooks/
│   └── cross_dataset_ids_evaluation_leakage_audited_updated.ipynb
│
├── results/
│   ├── project13_all_experiment_results.csv
│   ├── project13_best_models.csv
│   ├── project13_generalization_gap.csv
│   ├── project13_final_summary.json
│   ├── multi_seed_robustness_results.csv
│   ├── multi_seed_robustness_summary.csv
│   ├── domain_origin_separability_summary.csv
│   └── cicids_sampling_category_validation.csv
│
└── figures/
    ├── project13_macro_f1_comparison.png
    ├── project13_accuracy_comparison.png
    └── project13_fpr_comparison.png
```

This is a suggested structure. Keep only the files and folders that actually exist in the repository.

The original datasets are not included because of their size and distribution requirements. Download them from their official sources.

---

## **19\. How to Run the Project**

### **Step 1 — Open the Notebook**

Open:

```
cross_dataset_ids_evaluation_leakage_audited_updated.ipynb
```

using Google Colab.

### **Step 2 — Install Required Libraries**

The notebook installs the required model libraries:

```
!pip -q install xgboost lightgbm catboost joblib
```

### **Step 3 — Prepare Dataset Files**

Download CICIDS2017 and UNSW-NB15 from their official sources.

Place the files in the appropriate Google Drive directories or update the dataset paths in the notebook.

### **Step 4 — Configure Dataset Paths**

Review the dataset path definitions in the notebook before running the cells.

The current paths refer to a Google Drive directory used in the original experimental environment. They may need to be changed for another user.

### **Step 5 — Execute the Notebook**

Run the notebook sequentially from the beginning.

The additional validation cells depend on variables and outputs created by the preceding cells.

In particular:

* Cell 46 performs CICIDS2017 sampling and rare-category validation.  
* Cell 47 performs dataset-origin separability analysis.  
* Cell 48 performs the five-seed robustness evaluation.

Cell 48 trains 80 models and may require substantial computational time.

---

## **20\. Reproducibility and Generated Outputs**

The notebook saves experimental outputs under its configured `OUTPUT_DIR`.

Major output categories include:

| Output | Description |
| ----- | ----- |
| `feature_mapping_audit.csv` | Candidate feature mapping audit |
| `verified_feature_mapping.csv` | Resolved feature mapping report |
| `project13_all_experiment_results.csv` | Combined results from four experiments |
| `project13_best_models.csv` | Best model per experiment |
| `project13_generalization_gap.csv` | Within- and cross-dataset Macro-F1 comparison |
| `project13_final_summary.json` | Consolidated experiment summary |
| `multi_seed_robustness_results.csv` | Individual seed-level evaluation results |
| `multi_seed_robustness_summary.csv` | Mean and standard deviation across seeds |
| `domain_origin_separability_summary.csv` | Dataset-origin classification summary |
| `cicids_sampling_category_validation.csv` | Attack-category sampling comparison |

The notebook also generates confusion-matrix images and performance comparison figures.

Generated files depend on successful execution of the relevant cells and are not automatically included in the repository.

---

## **21\. Conclusion**

This project investigates an important limitation of machine learning-based network intrusion detection: strong performance on one benchmark dataset does not necessarily indicate reliable performance on another.

Using CICIDS2017 and UNSW-NB15, four tree-based machine learning classifiers, a harmonized five-feature representation, leakage-aware CICIDS2017 splitting, bidirectional cross-dataset experiments, dataset-origin separability analysis, and five-seed evaluation, the study demonstrates a substantial gap between within-dataset performance and cross-dataset generalization.

The results highlight the importance of evaluating intrusion detection models beyond a single benchmark dataset.

**The central takeaway is that reliable NIDS research should consider cross-dataset generalization, feature compatibility, sampling limitations, and false-positive behavior—not merely high within-dataset accuracy.**

---

## **22\. Project Information**

**Project:** 13 of the AI/ML/DL/Cybersecurity Portfolio

**Research Area:** Machine Learning, Cybersecurity, Network Intrusion Detection, Cross-Dataset Generalization

**Datasets:** CICIDS2017 and UNSW-NB15

**Models:** Random Forest, XGBoost, LightGBM, CatBoost

**Evaluation:** Four experimental configurations and five-seed robustness analysis

**Primary Notebook:**

`cross_dataset_ids_evaluation_leakage_audited_updated.ipynb`

**Repository:** [AI\_ML\_DL\_CYS\_Projects](https://github.com/munnamia1945/AI_ML_DL_CYS_Projects)

---

*This repository documents an experimental portfolio project. Reported results correspond to the notebook outputs and experimental summaries used in this project. They should not be interpreted as evidence of live-network deployment or production-level intrusion detection performance.*

## **Author**

**Md Munna Mia**

* GitHub: [https\://github.com/munnamia1945](https://github.com/munnamia1945)  
* Portfolio: [https\://www\.munnamia.com](https://www.munnamia.com/)

