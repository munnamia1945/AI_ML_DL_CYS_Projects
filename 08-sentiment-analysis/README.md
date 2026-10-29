# **Project 08: Sentiment Analysis**

A Natural Language Processing (NLP) project comparing a traditional machine-learning approach, TF-IDF with Logistic Regression, against a fine-tuned pretrained Transformer, DistilBERT, for binary sentiment classification.

## **Project Overview**

This project classifies movie reviews as **positive** or **negative** using the IMDb dataset. It evaluates two approaches under the same test-set setup and compares their classification performance.

The goal is to understand the trade-offs between a lightweight traditional NLP pipeline and a contextual Transformer model.

## **Objectives**

* Build a TF-IDF \+ Logistic Regression sentiment classifier.  
* Fine-tune a pretrained DistilBERT model.  
* Compare accuracy, macro precision, macro recall, and macro F1-score.  
* Evaluate per-class performance using classification reports.  
* Visualize confusion matrices and compare model predictions.  
* Export evaluation results and save trained model artifacts.

## **Dataset**

**Dataset:** IMDb Movie Reviews  
**Task:** Binary sentiment classification

| Dataset split | Samples |
| ----- | ----- |
| Training | 22,500 |
| Validation | 2,500 |
| Test | 25,000 |
| Total | 50,000 |

The dataset contains two sentiment classes:

* Negative  
* Positive

The training data was divided into training and validation subsets. The test set was used for final model evaluation.

## **Technologies Used**

* Python  
* Google Colab  
* Pandas and NumPy  
* Scikit-learn  
* TensorFlow/PyTorch ecosystem: PyTorch  
* Hugging Face Transformers and Datasets  
* Matplotlib and Seaborn  
* Joblib

## **Models**

### **1\. TF-IDF \+ Logistic Regression**

The traditional NLP pipeline uses:

* TF-IDF vectorization  
* Unigrams and bigrams  
* A maximum vocabulary size of 50,000 features  
* Logistic Regression for classification

This model serves as a computationally efficient baseline.

### **2\. Fine-tuned DistilBERT**

The Transformer approach uses the pretrained `distilbert/distilbert-base-uncased` model.

The workflow includes:

* Tokenization with the pretrained tokenizer  
* Truncation to a maximum sequence length of 256 tokens  
* Fine-tuning for two epochs  
* AdamW-style Transformer optimization through the Hugging Face Trainer  
* Validation-based model selection using macro F1-score

Unlike TF-IDF, DistilBERT uses contextual text representations to learn relationships between words in a review.

## **Experimental Results**

Both models were evaluated on the same 25,000-review test set.

### **Model Comparison**

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-score |
| ----- | ----- | ----- | ----- | ----- |
| TF-IDF \+ Logistic Regression | 0.8984 | 0.8984 | 0.8984 | 0.8984 |
| Fine-tuned DistilBERT | 0.9108 | 0.9110 | 0.9108 | 0.9107 |

### **Key Findings**

* **Best overall performance:** Fine-tuned DistilBERT.  
* **Accuracy improvement:** 1.24 percentage points over the TF-IDF baseline.  
* **Macro F1-score improvement:** 1.23 percentage points.  
* **Traditional baseline:** TF-IDF \+ Logistic Regression achieved 89.84% accuracy.  
* **Transformer performance:** DistilBERT achieved 91.08% accuracy.

The results show that the fine-tuned Transformer outperformed the traditional baseline on the reported test metrics. However, the improvement should be considered alongside its greater computational requirements.

### **Classification Report Summary**

| Model | Class | Precision | Recall | F1-score |
| ----- | ----- | ----- | ----- | ----- |
| TF-IDF \+ Logistic Regression | Negative | 0.90 | 0.89 | 0.90 |
| TF-IDF \+ Logistic Regression | Positive | 0.89 | 0.90 | 0.90 |
| Fine-tuned DistilBERT | Negative | 0.92 | 0.90 | 0.91 |
| Fine-tuned DistilBERT | Positive | 0.90 | 0.92 | 0.91 |

Each class had 12,500 test examples.

## **Evaluation Methods**

The experiment uses:

* Accuracy  
* Macro precision  
* Macro recall  
* Macro F1-score  
* Per-class precision, recall, and F1-score  
* Confusion matrices  
* A model-comparison visualization

Macro metrics give equal weight to the two sentiment classes.

## **Output Artifacts**

The notebook saves the following files under `project08_outputs/`:

* `model_comparison.csv` — evaluation metrics for both models  
* `test_predictions.csv` — true labels and model predictions  
* `tfidf_logistic_regression.joblib` — serialized TF-IDF and Logistic Regression pipeline  
* `project_summary.json` — experiment summary  
* `distilbert_checkpoints/` — training checkpoints  
* `distilbert_model/` — saved fine-tuned Transformer and tokenizer

## **How to Run**

1. Open `08-sentiment-analysis.ipynb` in Google Colab.  
2. Enable a GPU runtime if available.  
3. Run the notebook cells in order.  
4. Wait for dataset loading and model training to complete.  
5. Review the comparison table, classification reports, and confusion matrices.  
6. Export the generated artifacts for further analysis.

## **Limitations and Future Improvements**

* Compare inference latency and memory usage for both models.  
* Evaluate performance on a separate external sentiment dataset.  
* Test robustness against negation, sarcasm, and ambiguous reviews.  
* Experiment with different learning rates and training epochs.  
* Compare additional pretrained Transformer models.  
* Perform repeated runs or statistical testing to assess whether the observed performance difference is consistent.

## **Key Learnings**

* Implemented a complete sentiment-classification workflow.  
* Built a traditional TF-IDF-based NLP pipeline.  
* Fine-tuned a pretrained Transformer for binary classification.  
* Compared models using consistent test data and multiple evaluation metrics.  
* Exported predictions, evaluation results, and trained models.

## **Disclaimer**

This is an educational machine-learning project using the IMDb benchmark dataset. The reported metrics correspond to the experiment described in the notebook and do not establish state-of-the-art performance.

## **Author**

**Md. Munna Mia**

* GitHub: [https\://github.com/munnamia1945](https://github.com/munnamia1945)  
* Portfolio: [https\://www\.munnamia.com](https://www.munnamia.com/)

---

*Part of my AI/ML project portfolio.*

