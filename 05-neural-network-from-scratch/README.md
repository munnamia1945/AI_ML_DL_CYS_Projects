# **Project 05: Neural Network from Scratch Using NumPy**

## **1\. Overview**

This project implements a simple feedforward neural network from scratch using NumPy. Unlike implementations that rely on deep learning frameworks such as TensorFlow or PyTorch, this project manually implements the core mathematical operations required to train a neural network.

The network learns to classify a nonlinear, two-class dataset using forward propagation, binary cross-entropy loss, backpropagation, and gradient descent.

The primary goal is to understand how neural networks learn by updating weights and biases to reduce prediction error.

## **2\. Objectives**

* Build a feedforward neural network using NumPy.  
* Implement forward propagation manually.  
* Calculate binary cross-entropy loss.  
* Implement backpropagation to compute gradients.  
* Train the network using gradient descent.  
* Visualize training loss over multiple epochs.  
* Evaluate classification performance on unseen test data.  
* Visualize the learned decision boundary.  
* Understand the limitations of a small neural network trained on synthetic data.

## **3\. Dataset**

**Dataset:** Synthetic nonlinear binary classification dataset

**Generation method:** `make_moons()` from Scikit-learn

The dataset contains two classes arranged in a nonlinear, moon-shaped pattern. It is suitable for demonstrating how a neural network can learn nonlinear decision boundaries.

### **Dataset configuration**

| Parameter | Value |
| ----- | ----- |
| Total samples | 1,000 |
| Input features | 2 |
| Target classes | 2 |
| Noise | 0.20 |
| Test size | 20% |
| Training samples | 800 |
| Testing samples | 200 |
| Random state | 42 |

### **Data preprocessing**

The following steps were performed:

1. Generated the synthetic dataset.  
2. Converted the features and labels into NumPy arrays.  
3. Reshaped the target labels into a column vector.  
4. Split the dataset into training and testing sets using stratified sampling.  
5. Standardized the input features using the training-set mean and standard deviation.  
6. Applied the same training-set scaling parameters to the test set.

Using training-set statistics for scaling helps prevent information from the test set from leaking into preprocessing.

## **4\. Neural Network Architecture**

The model is a fully connected feedforward neural network with one hidden layer.

| Layer | Number of neurons | Activation |
| ----- | ----- | ----- |
| Input layer | 2 | None |
| Hidden layer | 8 | Tanh |
| Output layer | 1 | Sigmoid |

### **Architecture**

Input Features (2)  
        \|  
        v  
Hidden Layer (8 neurons)  
     Tanh Activation  
        \|  
        v  
Output Layer (1 neuron)  
    Sigmoid Activation  
        \|  
        v  
Predicted Probability  
        \|  
        v  
Binary Classification

The output layer produces a probability between 0 and 1\. A threshold of 0.5 is used to assign the predicted class.

## **5\. Technologies Used**

* Python  
* NumPy  
* Pandas  
* Matplotlib  
* Scikit-learn  
* Google Colab

**Important:** Scikit-learn was used to generate and split the dataset and evaluate the model. The neural network's forward propagation, backpropagation, and gradient-descent updates were implemented manually using NumPy.

## **6\. Implementation Details**

### **6.1 Weight and Bias Initialization**

Weights were initialized using random values scaled according to the input dimensions. Biases were initialized to zero.

This provides the network with different initial weights for its connections and allows training to begin.

### **6.2 Forward Propagation**

Forward propagation passes input features through the neural network to generate predictions.

**Hidden Layer:**

Z1=XW1+b1

A1=tanh⁡(Z1)

**Output Layer:**

Z2=A1W2+b2

y^=σ(Z2)

Here, X represents the input features, W the weights, b the biases, and y^ the predicted probability. The sigmoid function ensures the output lies between 0 and 1\.

Here, (X) represents the input data, (W) represents weights, (b) represents biases, and (\\hat{y}) represents the predicted probability.

### **6.3 Loss Function**

### Binary cross-entropy loss measures the difference between actual labels and predicted probabilities.

### L=−1m∑i=1m\[yilog⁡(y^i)+(1−yi)log⁡(1−y^i)\]

### Here, m is the number of samples, yi is the actual label, and y^i is the predicted probability. The training objective is to minimize the loss by updating the network's weights and biases.

### **6.4 Backpropagation**

Backpropagation calculates the gradients of the loss with respect to the network's weights and biases.

The gradients are propagated backward from the output layer to the hidden layer using the chain rule.

For the sigmoid output layer with binary cross-entropy loss:

\[  
dZ\_2=\\frac{\\hat{y}-y}{m}  
\]

The gradients are then calculated for the output and hidden layers. The derivative of the tanh activation is used when calculating the hidden-layer gradients.

### **6.5 Gradient Descent**

The model updates its weights and biases using gradient descent:

\[  
W \\leftarrow W-\\eta\\nabla\_W L  
\]

\[  
b \\leftarrow b-\\eta\\nabla\_b L  
\]

Here, (\\eta) is the learning rate and (\\nabla L) represents the relevant gradient.

The training loop repeats forward propagation, loss calculation, backpropagation, and parameter updates for the specified number of epochs.

## **7\. Training Configuration**

| Parameter | Value |
| ----- | ----- |
| Hidden neurons | 8 |
| Learning rate | 0.05 |
| Training epochs | 5,000 |
| Loss function | Binary Cross-Entropy |
| Hidden activation | Tanh |
| Output activation | Sigmoid |
| Classification threshold | 0.5 |
| Random seed | 42 |

## **8\. Model Performance**

The trained network was evaluated on the held-out test set containing 200 samples.

### **Test-set results**

| Metric | Score |
| ----- | ----- |
| Accuracy | 0.980000 |
| Precision | 0.980000 |
| Recall | 0.980000 |
| F1-score | 0.980000 |

**Test accuracy: 98.00%**

These results indicate that the network classified most of the test samples correctly in this experiment.

### **Classification report**

| Class | Precision | Recall | F1-score | Support |
| ----- | ----- | ----- | ----- | ----- |
| Class 0 | 0.98 | 0.98 | 0.98 | 100 |
| Class 1 | 0.98 | 0.98 | 0.98 | 100 |
| Macro average | 0.98 | 0.98 | 0.98 | 200 |
| Weighted average | 0.98 | 0.98 | 0.98 | 200 |

The reported accuracy was 0.98 across 200 test samples. Precision, recall, and F1-score were also 0.98 for both classes.

Precision measures the proportion of positive predictions that were correct. Recall measures the proportion of actual positive samples correctly identified. The F1-score combines precision and recall into a single metric.

## **9\. Training Loss Analysis**

The final recorded training loss was:

**Final training loss: 0.123098**

The training loss curve showed a downward trend during training, reaching approximately 0.123098 by the end of the 5,000-epoch run.

A decreasing loss indicates that the model was reducing its error on the training data. However, training loss alone does not establish generalization performance; the held-out test metrics are also important.

## **10\. Decision Boundary Visualization**

The decision boundary plot illustrates how the trained neural network separates the two classes in the two-dimensional feature space.

The plot displays:

* The learned classification regions as a colored background.  
* Test samples as individual points.  
* Different colors for the two actual classes.

Because the dataset is nonlinear, a linear decision boundary would generally be insufficient to represent its structure. The hidden layer and tanh activation allow the network to learn a nonlinear separation.

The visualization provides an intuitive way to examine the network's classification behavior.

## **11\. Key Findings**

1. A feedforward neural network was implemented using NumPy without TensorFlow or PyTorch.  
2. Forward propagation was used to generate predictions.  
3. Binary cross-entropy measured prediction error.  
4. Backpropagation calculated gradients for weights and biases.  
5. Gradient descent updated model parameters over 5,000 epochs.  
6. The network achieved 98.00% test accuracy on the synthetic dataset.  
7. The final recorded training loss was 0.123098.  
8. The decision boundary visualization demonstrated the network's ability to learn a nonlinear classification pattern.

## **12\. Limitations**

* The experiment used a synthetic dataset rather than a real-world dataset.  
* The network has only one hidden layer and eight hidden neurons.  
* Performance may vary with different random seeds, initialization values, and training configurations.  
* The results do not establish performance on more complex datasets.  
* A single train-test split provides limited evidence about performance stability.  
* The training loss does not directly measure generalization.  
* The implementation does not include advanced optimizers such as Adam, regularization methods, or automated hyperparameter tuning.  
* Gradient correctness was not independently verified using numerical gradient checking.

## **13\. Future Improvements**

Possible extensions include:

* Compare different hidden-layer sizes and network architectures.  
* Experiment with ReLU, sigmoid, and tanh activation functions.  
* Implement mini-batch and stochastic gradient descent.  
* Add the Adam optimizer from scratch.  
* Implement gradient checking using finite differences.  
* Compare performance across multiple random seeds.  
* Track and visualize training and validation loss separately.  
* Apply the implementation to a real-world binary classification dataset.  
* Compare the NumPy implementation with a Scikit-learn or deep learning framework baseline.

These are proposed extensions and should not be considered completed experiments.

## **14\. Repository Structure**

05-neural-network-from-scratch/  
├── 05-neural-network-from-scratch.ipynb  
├── README.md  
├── neural\_network\_metrics.csv  
└── neural\_network\_loss\_history.csv

### **File descriptions**

* `05-neural-network-from-scratch.ipynb`: Notebook containing the implementation, training, evaluation, and visualizations.  
* `README.md`: Project documentation.  
* `neural_network_metrics.csv`: Exported model evaluation metrics.  
* `neural_network_loss_history.csv`: Training loss values recorded across epochs.

Include the CSV files in the repository if they were generated and saved successfully.

## **15\. Conclusion**

This project demonstrates the fundamental mechanics of a neural network through a manual NumPy implementation. By implementing forward propagation, binary cross-entropy, backpropagation, and gradient descent, the project provides a practical understanding of how network parameters are optimized during training.

The model achieved 98.00% accuracy, precision, recall, and F1-score on the reported test set, with a final training loss of 0.123098. These results demonstrate effective classification on the selected synthetic dataset, while further evaluation on different datasets and configurations would be necessary to assess robustness.

The project serves as a foundation for understanding neural network mathematics and preparing for more advanced machine learning and deep learning experiments.

---

**Project:** 05 — Neural Network from Scratch  
**Domain:** Machine Learning, Neural Networks, Deep Learning Fundamentals  
**Implementation:** NumPy  
**Environment:** Google Colab  
**Status:** Initial implementation, training, and evaluation completed

## **16\. Author**

**Md. Munna Mia**

CSE Undergraduate \| AI/ML and Cybersecurity

* GitHub: [https\://github.com/munnamia1945](https://github.com/munnamia1945)  
* Portfolio: [https\://www\.munnamia.com](https://www.munnamia.com)

