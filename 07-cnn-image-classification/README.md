# **Project 07: CNN Image Classification**

A Convolutional Neural Network (CNN) image classification project using TensorFlow/Keras and the CIFAR-10 dataset. This project explores image augmentation, validation, model evaluation, and practical techniques for controlling overfitting.

## **Project Overview**

The goal is to build a CNN that classifies small RGB images into 10 object categories. The workflow includes dataset preparation, image normalization, data augmentation, CNN training, performance evaluation, and model export.

## **Objectives**

* Build a CNN for multi-class image classification.  
* Apply image normalization and data augmentation.  
* Use a separate validation set for model monitoring.  
* Apply EarlyStopping, ReduceLROnPlateau, Batch Normalization, Dropout, and L2 regularization.  
* Evaluate performance using test metrics, a classification report, and a confusion matrix.  
* Save the trained model and export evaluation results.

## **Dataset**

**CIFAR-10** contains 60,000 color images of size 32 × 32 pixels, distributed across 10 classes.

| Dataset split | Number of images |
| ----- | ----- |
| Original training set | 50,000 |
| Training set after splitting | 45,000 |
| Validation set | 5,000 |
| Test set | 10,000 |
| Number of classes | 10 |

The training data is split into training and validation sets using stratified sampling. Each class has 4,500 training examples after the split.

### **CIFAR-10 Classes**

1. Airplane  
2. Automobile  
3. Bird  
4. Cat  
5. Deer  
6. Dog  
7. Frog  
8. Horse  
9. Ship  
10. Truck

## **Technologies Used**

* Python  
* Google Colab  
* TensorFlow / Keras  
* NumPy  
* Matplotlib  
* Seaborn  
* Pandas  
* Scikit-learn

## **Methodology**

### **1\. Data Preparation**

* Loaded the CIFAR-10 dataset.  
* Created training, validation, and test splits.  
* Normalized pixel values to the range `[0, 1]`.  
* Visualized sample images and class distributions.

### **2\. Data Augmentation**

The augmentation pipeline includes:

* Random horizontal flipping  
* Random rotation  
* Random translation

These transformations help the model learn from varied image representations and can improve generalization.

### **3\. CNN Architecture**

The model uses a sequential CNN architecture with:

* Convolutional layers with 32, 64, and 128 filters  
* Batch Normalization  
* Max Pooling  
* Dropout and L2 regularization  
* Global Average Pooling  
* A dense hidden layer with 128 units  
* A 10-unit softmax output layer

**Total parameters:** 158,122  
**Trainable parameters:** 157,674

### **4\. Training Configuration**

* Optimizer: Adam  
* Initial learning rate: 0.001  
* Loss function: Sparse Categorical Crossentropy  
* Batch size: 128  
* Maximum epochs: 20  
* EarlyStopping with restoration of the best weights  
* ReduceLROnPlateau for learning-rate adjustment  
* ModelCheckpoint for saving the best model

### **5\. Evaluation**

The trained model was evaluated using:

* Test loss and accuracy  
* Macro and weighted precision, recall, and F1-score  
* Per-class classification report  
* Confusion matrix  
* Sample predictions with confidence scores  
* Training and validation accuracy/loss curves

## **Results**

### **Test Performance**

| Metric | Result |
| ----- | ----- |
| Test Loss | 0.7919 |
| Test Accuracy | 0.7487 (74.87%) |
| Macro F1-score | 0.7434 (74.34%) |

### **Per-Class Performance Highlights**

| Class | F1-score |
| ----- | ----- |
| Automobile | 0.9118 |
| Ship | 0.8717 |
| Cat | 0.5445 |

The model performs particularly well on automobile and ship images. Cat classification remains a challenging area and may benefit from further experimentation with model architecture, augmentation strategies, or training configuration.

### **Overfitting Analysis**

| Metric | Result |
| ----- | ----- |
| Best epoch by validation loss | 20 |
| Best validation loss | 0.7559 |
| Final training accuracy | 0.7526 |
| Final validation accuracy | 0.7596 |
| Training − validation accuracy gap | \-0.0070 |

The small negative accuracy gap indicates that validation accuracy was slightly higher than training accuracy at the final epoch. This is consistent with regularization and data augmentation helping to control the train–validation gap. However, the gap alone does not prove that overfitting is entirely absent.

## **Output Files**

The notebook exports the following artifacts:

* `cnn_test_metrics.csv` — test evaluation metrics  
* `cnn_training_history.csv` — training and validation history  
* `cnn_classification_report.csv` — per-class precision, recall, and F1-score  
* `cnn_overfitting_summary.csv` — summary of validation performance and train–validation gap  
* `best_cnn_model.keras` — checkpoint of the best model according to validation loss  
* `final_cnn_model.keras` — exported final model

The output files are stored in the `project07_outputs/` directory.

## **How to Run**

1. Open the notebook in Google Colab.  
2. Enable a GPU runtime if available.  
3. Run the cells in order.  
4. Allow the dataset to load and the model to train.  
5. Review the training curves, test metrics, classification report, and confusion matrix.  
6. Download the exported model and CSV files if needed.

**Notebook:** `07-cnn-image-classification.ipynb`

## **Limitations and Future Improvements**

* Improve performance on the cat class and other challenging categories.  
* Experiment with deeper CNN architectures and alternative augmentation policies.  
* Tune learning rate, batch size, dropout, and regularization.  
* Compare the baseline CNN with transfer-learning models where appropriate.  
* Evaluate robustness and repeat experiments with additional random seeds.

## **Key Learnings**

* Implemented a CNN image classification workflow using TensorFlow/Keras.  
* Practiced image normalization and augmentation.  
* Used validation-based callbacks to monitor training.  
* Evaluated multi-class performance using classification metrics and a confusion matrix.  
* Exported trained models and structured experiment results.

## **Disclaimer**

This is an educational machine-learning project using the CIFAR-10 benchmark dataset. Reported results correspond to the experiment described in this notebook and are not a claim of state-of-the-art performance.

## **Author**

**Md. Munna Mia**

* GitHub: [https\://github.com/munnamia1945](https://github.com/munnamia1945)  
* Portfolio: [https\://www\.munnamia.com](https://www.munnamia.com/)

---

*Part of my AI/ML project practice and portfolio.*

