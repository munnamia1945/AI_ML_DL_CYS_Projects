# **Project 10: Sequence Classification with LSTM**

## **1\. Project Overview**

This project implements a Long Short-Term Memory (LSTM) neural network for binary sentiment classification of IMDb movie reviews. The model processes sequences of word-token IDs and predicts whether a review expresses positive or negative sentiment.

The project demonstrates sequence preprocessing, embedding layers, recurrent neural networks, regularization, early stopping, and test-set evaluation.

## **2\. Objectives**

* Build an LSTM-based sentiment classification model.  
* Prepare variable-length text sequences using padding and truncation.  
* Apply dropout and early stopping during training.  
* Evaluate performance using accuracy, precision, recall, F1-score, and a confusion matrix.  
* Test the trained model on custom movie reviews.

## **3\. Dataset**

* **Dataset:** IMDb Movie Reviews  
* **Task:** Binary sentiment classification  
* **Total samples:** 50,000  
* **Training samples:** 20,000  
* **Validation samples:** 5,000  
* **Test samples:** 25,000  
* **Vocabulary limit:** 20,000 words  
* **Maximum sequence length:** 250 tokens

The training, validation, and test sets are balanced between the two sentiment classes.

The official test set was kept separate from the training and validation data.

## **4\. Technologies Used**

* Python  
* TensorFlow / Keras  
* NumPy  
* Pandas  
* Matplotlib  
* Scikit-learn  
* Google Colab

**Execution environment:** CPU; GPU was unavailable during the reported run.

## **5\. Model Architecture**

The sequential model contains the following layers:

1. **Embedding:** Maps token IDs to 128-dimensional dense vectors; `mask_zero=True`.  
2. **SpatialDropout1D:** Dropout rate of 0.4.  
3. **LSTM:** 64 units, with dropout of 0.2 and recurrent dropout of 0.0.  
4. **Dropout:** Dropout rate of 0.4.  
5. **Dense:** One output unit with sigmoid activation.

The sigmoid output represents the predicted probability of the positive class.

The exact parameter count is omitted because the available model-summary output did not verify it.

## **6\. Data Preprocessing**

1. Load the IMDb dataset with a vocabulary limit of 20,000 words.  
2. Split the original training portion into 20,000 training and 5,000 validation samples using stratification.  
3. Pad or truncate sequences to 250 tokens.  
4. Use pre-padding and pre-truncation to standardize input lengths.

## **7\. Training Configuration**

| Parameter | Configuration |
| ----- | ----- |
| Optimizer | Adam |
| Loss function | Binary cross-entropy |
| Batch size | 128 |
| Maximum epochs | 10 |
| Early stopping monitor | Validation loss |
| Early stopping patience | 2 |
| Restore best weights | Enabled |
| Model checkpoint | Best validation-loss checkpoint |
| Random seed | 42 |

Training stopped after three epochs. The best validation loss occurred at Epoch 1, with a value of **0.3226**. The early-stopping callback restored the best weights.

## **8\. Training and Validation Results**

| Metric | Result |
| ----- | ----- |
| Epochs executed | 3 |
| Best epoch | 1 |
| Best validation loss | 0.3226 |
| Best validation accuracy | 0.8700 |

These are validation results, not test-set results.

## **9\. Test Evaluation**

The model was evaluated on the held-out test set of 25,000 reviews.

| Metric | Score |
| ----- | ----- |
| Test loss | 0.3333 |
| Accuracy | 0.8605 |
| Positive-class precision | 0.9118 |
| Positive-class recall | 0.7982 |
| Positive-class F1-score | 0.8513 |

The reported accuracy was **86.05%**. Positive-class precision was higher than positive-class recall, indicating that the model missed some positive reviews.

## **10\. Classification Report**

| Class | Precision | Recall | F1-score | Support |
| ----- | ----- | ----- | ----- | ----- |
| Negative (0) | 0.8206 | 0.9228 | 0.8687 | 12,500 |
| Positive (1) | 0.9118 | 0.7982 | 0.8513 | 12,500 |
| Macro average | 0.8662 | 0.8605 | 0.8600 | 25,000 |
| Weighted average | 0.8662 | 0.8605 | 0.8600 | 25,000 |

The negative class achieved higher recall, while the positive class achieved higher precision.

## **11\. Confusion Matrix**

|  | Predicted Negative | Predicted Positive |
| ----- | ----- | ----- |
| Actual Negative | 11,535 | 965 |
| Actual Positive | 2,522 | 9,978 |

*   
  **True negatives:** 11,535  
* **False positives:** 965  
* **False negatives:** 2,522  
* **True positives:** 9,978

The model incorrectly classified 3,487 of the 25,000 test samples.

## **12\. Custom Review Predictions**

The model was also tested on four manually written reviews.

| Review | Predicted sentiment | Positive-class score |
| ----- | ----- | ----- |
| This movie was wonderful and the acting was excellent. | Positive | 0.5983 |
| The film was boring and the story was terrible. | Negative | 0.3118 |
| I enjoyed the performances and would watch it again. | Positive | 0.6890 |
| The plot was confusing and I did not like this movie. | Negative | 0.2207 |

These examples demonstrate the inference pipeline only. Four predictions are not sufficient to establish general performance.

The custom-review tokenizer uses a simple IMDb vocabulary-based implementation. Its handling of punctuation, unknown words, and text normalization is limited, so predictions on arbitrary real-world text may be less reliable.

## **13\. How to Run in Google Colab**

1. Open `10-sequence-classification-with-lstm.ipynb` in Google Colab.  
2. Install any required dependencies if they are not already available.  
3. Run the notebook cells sequentially.  
4. Review the evaluation metrics, classification report, confusion matrix, and training curves.  
5. Find the generated files in the `project10_outputs` directory.

A GPU runtime may reduce training time, but the notebook was executed using a CPU.

## **14\. Generated Output Files**

The notebook generated the following 14 files:

* `best_lstm_model.keras`  
* `class_distribution.csv`  
* `class_distribution.png`  
* `classification_report.csv`  
* `confusion_matrix.png`  
* `custom_predictions.csv`  
* `final_lstm_model.keras`  
* `misclassified_examples.csv`  
* `project_summary.json`  
* `sequence_length_distribution.png`  
* `test_metrics.csv`  
* `test_predictions.csv`  
* `training_curves.png`  
* `training_history.csv`

## **15\. Limitations**

* The model was trained without GPU acceleration.  
* The custom inference tokenizer uses a basic vocabulary-based approach and limited text normalization.  
* Truncating reviews to 250 tokens can remove useful context.  
* The experiment does not include a controlled comparison against other models.  
* No conclusion about superiority over other architectures can be drawn from this experiment alone.  
* Custom predictions illustrate functionality but do not constitute an independent evaluation dataset.

## **16\. Future Improvements**

* Compare LSTM with a simple TF-IDF and Logistic Regression baseline.  
* Experiment with Bidirectional LSTM and GRU architectures.  
* Evaluate pretrained word embeddings such as GloVe.  
* Improve tokenization and text normalization for custom inputs.  
* Analyze misclassified reviews and investigate class-specific errors.  
* Experiment with transformer-based models for a controlled comparison.

## **17\. Learning Outcomes**

* Preparing sequential text data for neural networks.  
* Understanding embedding layers and LSTM-based sequence modelling.  
* Applying dropout, early stopping, and model checkpointing.  
* Evaluating binary classifiers using multiple metrics.  
* Interpreting confusion matrices and class-specific performance.  
* Exporting model predictions, metrics, and training visualizations.

## **18\. Author**

**Md. Munna Mia**

CSE Undergraduate \| AI/ML and Cybersecurity

* GitHub: [https\://github.com/munnamia1945](https://github.com/munnamia1945)  
* Portfolio: [https\://www\.munnamia.com](https://www.munnamia.com/)

