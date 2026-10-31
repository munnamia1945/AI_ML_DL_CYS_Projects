# **Project 11: Transformer-Based Text Classification Using DistilBERT**

A transformer-based NLP project that fine-tunes **DistilBERT** for multiclass news classification using the AG News dataset. The model classifies news articles into four categories: World, Sports, Business, and Sci/Tech.

The project includes dataset exploration, tokenization, transformer fine-tuning, model evaluation, confusion matrix analysis, error analysis, and custom text predictions.

## **Project Overview**

| Item | Details |
| ----- | ----- |
| Project | Transformer-Based Text Classification |
| Project Number | 11 |
| Task | Multiclass News Classification |
| Dataset | AG News |
| Dataset ID | `fancyzhx/ag_news` |
| Pretrained Model | `distilbert/distilbert-base-uncased` |
| Number of Classes | 4 |
| Framework | PyTorch |
| NLP Library | Hugging Face Transformers |
| Environment | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Test Accuracy | **94.51%** |
| Test Macro-F1 | **94.52%** |

## **1\. Project Objectives**

* Fine-tune a pretrained DistilBERT model for multiclass text classification.  
* Classify news articles into four distinct categories.  
* Apply transformer-based tokenization and dynamic padding.  
* Evaluate model performance using accuracy, precision, recall, and macro-F1.  
* Analyze class-wise performance and prediction errors.  
* Test the trained model with custom news text.

## **2\. Dataset Description**

The project uses the **AG News dataset**, a benchmark dataset for news topic classification.

Dataset: [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news)

The dataset contains four balanced news categories.

### **Dataset Distribution**

| Dataset Split | Samples | Samples per Class |
| ----- | ----- | ----- |
| Training | 108,000 | 27,000 |
| Validation | 12,000 | 3,000 |
| Official Test | 7,600 | 1,900 |
| **Total** | **127,600** | — |

The original training set was divided into training and validation subsets using a stratified 90/10 split. The official test set was kept separate for final evaluation.

### **Class Labels**

| Label | Category |
| ----- | ----- |
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

All four classes have equal representation in each dataset split.

## **3\. Technologies and Libraries**

* **Programming Language:** Python  
* **Environment:** Google Colab  
* **Deep Learning Framework:** PyTorch  
* **Transformer Library:** Hugging Face Transformers  
* **Dataset Management:** Hugging Face Datasets  
* **Training Acceleration:** Accelerate  
* **Data Processing:** Pandas, NumPy  
* **Visualization:** Matplotlib  
* **Evaluation:** Scikit-learn

### **Main Software Versions**

| Library | Version |
| ----- | ----- |
| PyTorch | 2.11.0+cu130 |
| Transformers | 4.46.3 |
| Datasets | 2.21.0 |
| Accelerate | 0.34.2 |

## **4\. Model Architecture**

The project uses `distilbert-base-uncased`, a distilled version of the BERT transformer architecture.

A sequence classification head was added to the pretrained model to predict one of four news categories.

**Model:** `AutoModelForSequenceClassification`

The model was fine-tuned using the Hugging Face `Trainer` API with validation-based model selection and early stopping.

## **5\. Data Preprocessing and Tokenization**

The following preprocessing steps were applied:

1. Loaded the AG News dataset.  
2. Created a stratified training-validation split.  
3. Initialized the DistilBERT tokenizer.  
4. Tokenized news text using the pretrained tokenizer.  
5. Limited the maximum sequence length to 128 tokens.  
6. Applied truncation to longer sequences.  
7. Used dynamic padding through `DataCollatorWithPadding`.

## **6\. Training Configuration**

| Hyperparameter | Value |
| ----- | ----- |
| Pretrained Model | `distilbert-base-uncased` |
| Maximum Sequence Length | 128 |
| Number of Labels | 4 |
| Epochs | 3 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| Random Seed | 42 |
| Mixed Precision | FP16 |
| Early Stopping Patience | 2 |
| Best Model Selection | Validation Macro-F1 |
| Training Hardware | Tesla T4 GPU |

## **7\. Training and Validation Results**

The model was evaluated on the validation set after each training epoch.

The best checkpoint was `checkpoint-20250`, selected at Epoch 3 based on validation Macro-F1.

| Metric | Validation Result |
| ----- | ----- |
| Validation Loss | 0.226819 |
| Validation Accuracy | 0.945500 |
| Validation Macro-F1 | 0.945551 |

## **8\. Final Test Performance**

The selected model was evaluated on the official test set containing 7,600 previously unseen samples.

\<details\>  
\<summary\>Final test metrics\</summary\>

| Metric | Score |
| ----- | ----- |
| Test Loss | 0.2240 |
| Accuracy | **0.9451** |
| Macro Precision | 0.9452 |
| Macro Recall | 0.9451 |
| Macro F1-Score | **0.9452** |

\</details\>

The model achieved approximately **94.51% test accuracy** and **94.52% macro-F1**, demonstrating strong classification performance across the four news categories.

## **9\. Class-Wise Classification Report**

| Class | Precision | Recall | F1-Score | Support |
| ----- | ----- | ----- | ----- | ----- |
| World | 0.9598 | 0.9542 | 0.9570 | 1,900 |
| Sports | 0.9879 | 0.9863 | 0.9871 | 1,900 |
| Business | 0.9170 | 0.9126 | 0.9148 | 1,900 |
| Sci/Tech | 0.9163 | 0.9274 | 0.9218 | 1,900 |
| **Macro Average** | **0.9452** | **0.9451** | **0.9452** | **7,600** |

### **Performance Observations**

* **Sports** achieved the highest F1-score at 98.71%.  
* **World** achieved an F1-score of 95.70%.  
* **Sci/Tech** achieved an F1-score of 92.18%.  
* **Business** achieved an F1-score of 91.48%.

The model performed particularly well on Sports news, while Business and Sci/Tech were comparatively more challenging.

## **10\. Confusion Matrix**

The following confusion matrix shows actual versus predicted categories.

| Actual / Predicted | World | Sports | Business | Sci/Tech |
| ----- | ----- | ----- | ----- | ----- |
| **World** | 1,813 | 11 | 46 | 30 |
| **Sports** | 12 | 1,874 | 7 | 7 |
| **Business** | 36 | 6 | 1,734 | 124 |
| **Sci/Tech** | 28 | 6 | 104 | 1,762 |

The diagonal values represent correctly classified samples.

## **11\. Error Analysis**

Out of 7,600 test samples:

* Correct predictions: **7,183**  
* Incorrect predictions: **417**  
* Overall error rate: **5.49%**

### **Class-Wise Errors**

| Class | Misclassified Samples | Error Rate |
| ----- | ----- | ----- |
| Business | 166 | 8.74% |
| Sci/Tech | 138 | 7.26% |
| World | 87 | 4.58% |
| Sports | 26 | 1.37% |

### **Key Findings**

**1\. Strong Sports Classification**

Sports news was the easiest category for the model to distinguish, with only 26 misclassified samples.

**2\. Business and Sci/Tech Confusion**

The most noticeable classification errors occurred between Business and Sci/Tech:

* 124 Business articles were predicted as Sci/Tech.  
* 104 Sci/Tech articles were predicted as Business.

This overlap may reflect shared terminology in technology companies, financial markets, business developments, and technological innovation.

**3\. Opportunities for Improvement**

Future experiments could investigate longer input sequences, alternative transformer architectures, and additional preprocessing or fine-tuning strategies.

## **12\. Custom Text Predictions**

The fine-tuned model was also tested with custom news text.

| Input Text | Predicted Class | Confidence |
| ----- | ----- | ----- |
| The United Nations discussed a new international agreement. | World | \~98.9% |
| The team won the championship after an exciting final. | Sports | \~96.5% |
| The stock market rose following strong company earnings. | Business | \~99.8% |
| Researchers announced a new breakthrough in artificial intelligence. | Sci/Tech | \~98.6% |

These examples demonstrate the model's ability to classify new text inputs. Confidence scores are model prediction probabilities, not guarantees of correctness.

## **13\. Project Structure**

The generated outputs are stored in the `project11_outputs/` directory.

```
project11_outputs/
│
├── best_checkpoint/
├── fine_tuned_distilbert/
│
├── class_distribution.csv
├── class_distribution.png
│
├── training_history.csv
├── training_curves.png
│
├── classification_report.csv
├── test_metrics.json
│
├── confusion_matrix.csv
├── confusion_matrix.png
│
├── misclassified_examples.csv
│
├── class_error_analysis.csv
├── class_error_analysis.png
│
└── custom_predictions.csv
```

### **Important Output Files**

| File | Description |
| ----- | ----- |
| `best_checkpoint/` | Best Trainer checkpoint |
| `fine_tuned_distilbert/` | Saved fine-tuned model and tokenizer |
| `training_history.csv` | Epoch-wise training and validation metrics |
| `training_curves.png` | Training and validation performance plots |
| `classification_report.csv` | Class-wise evaluation metrics |
| `test_metrics.json` | Overall test metrics |
| `confusion_matrix.csv` | Confusion matrix data |
| `confusion_matrix.png` | Confusion matrix visualization |
| `misclassified_examples.csv` | Incorrect predictions and confidence scores |
| `class_error_analysis.csv` | Class-wise error statistics |
| `class_error_analysis.png` | Class-wise error visualization |
| `custom_predictions.csv` | Predictions for custom text inputs |

## **14\. How to Run the Project**

The notebook was developed for Google Colab.

1. Open Google Colab.  
2. Upload or open the project notebook.  
3. Select a GPU runtime.  
4. Execute the notebook cells sequentially.  
5. Allow the required libraries and AG News dataset to load.  
6. Train and evaluate the model.  
7. Review the generated metrics, plots, and prediction files.

**Note:** GPU availability and training time may vary depending on the Colab runtime.

## **15\. Limitations**

* The model was evaluated on the AG News dataset and may not generalize equally well to every news source or writing style.  
* Business and Sci/Tech articles show noticeable category overlap.  
* The maximum input sequence length was limited to 128 tokens.  
* Custom prediction examples are limited and do not constitute a comprehensive real-world evaluation.  
* The project focuses on four predefined news categories rather than open-ended topic discovery.

## **16\. Future Improvements**

* Experiment with BERT, RoBERTa, and DeBERTa architectures.  
* Investigate longer token sequence lengths.  
* Perform hyperparameter optimization.  
* Evaluate robustness on additional news datasets.  
* Investigate class-specific errors and ambiguous news articles.  
* Develop an interactive web interface for news classification.

## **17\. Conclusion**

This project demonstrates the practical application of transformer-based NLP through fine-tuning DistilBERT on the AG News dataset.

The model achieved **94.51% test accuracy** and **94.52% macro-F1** across four balanced news categories. Sports classification was particularly strong, while Business and Sci/Tech presented the main classification challenges.

The project provides hands-on experience with pretrained transformer models, tokenization, GPU-based fine-tuning, evaluation, and error analysis using the Hugging Face ecosystem.

## **18\. Author**

**Md. Munna Mia**

CSE Undergraduate \| AI/ML and Cybersecurity

* GitHub: [https\://github.com/munnamia1945](https://github.com/munnamia1945)  
* Portfolio: [https\://www\.munnamia.com](https://www.munnamia.com/)

\[Project 11 — Transformer-Based Text Classification Using DistilBERT

Part of an AI/ML learning and implementation portfolio.\]

