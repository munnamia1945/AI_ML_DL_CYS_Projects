# **Project 02: House Price Prediction**

## **1\. Project Overview**

This project uses machine learning regression models to predict median house values in California using the California Housing dataset from Scikit-learn.

The workflow includes data preprocessing, feature engineering, missing-value imputation, model training, cross-validation, and performance comparison across five regression algorithms.

## **2\. Objectives**

* Explore housing-related demographic and geographic features.  
* Engineer additional features from existing variables.  
* Handle missing and infinite values through preprocessing.  
* Train and compare multiple regression models.  
* Evaluate model performance using MAE, RMSE, and R².  
* Apply cross-validation to assess model performance on the training data.

## **3\. Dataset Description**

* **Dataset:** California Housing  
* **Source:** Scikit-learn (`fetch_california_housing`)  
* **Total samples:** 20,640  
* **Target variable:** `MedHouseVal`  
* **Original features:** 8  
* **Engineered features:** 2  
* **Total features after engineering:** 10

The original features are:

* `MedInc` — Median income  
* `HouseAge` — Median house age  
* `AveRooms` — Average rooms per household  
* `AveBedrms` — Average bedrooms per household  
* `Population` — Population  
* `AveOccup` — Average household occupancy  
* `Latitude` — Geographic latitude  
* `Longitude` — Geographic longitude

The target variable `MedHouseVal` is expressed in units of \$100,000 in the original dataset.

## **4\. Technologies Used**

* Python  
* Pandas  
* NumPy  
* Matplotlib  
* Seaborn  
* Scikit-learn  
* Google Colab

## **5\. Data Preprocessing and Feature Engineering**

The following steps were implemented:

**Feature engineering**

* `BedroomsPerRoom`: Average bedrooms divided by average rooms.  
* `RoomsPerOccupant`: Average rooms divided by average occupancy.

**Data cleaning**

* Infinite values were replaced with `NaN`.  
* `SimpleImputer` with the median strategy was used to handle missing values.

**Feature scaling**

* `StandardScaler` was applied to Linear Regression and Ridge Regression pipelines.  
* Tree-based models were trained without standard scaling.

## **6\. Models Implemented**

Five regression algorithms were evaluated:

1. Linear Regression  
2. Ridge Regression  
3. Decision Tree Regressor  
4. Random Forest Regressor  
5. Gradient Boosting Regressor

The reported configurations included Ridge Regression with `alpha=1.0`, a Decision Tree with `max_depth=15`, a Random Forest with 100 estimators, and Gradient Boosting with 160 estimators.

## **7\. Experimental Setup**

| Setting | Value |
| ----- | ----- |
| Training split | 80% |
| Test split | 20% |
| Training samples | 16,512 |
| Test samples | 4,128 |
| Random state | 42 |
| Cross-validation | 5-fold K-Fold |

The test set was reserved for final model evaluation. Cross-validation was performed on the training data.

## **8\. Model Evaluation Metrics**

* **MAE (Mean Absolute Error):** Measures the average absolute difference between predictions and actual values.  
* **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more heavily.  
* **R² (Coefficient of Determination):** Measures how much variation in the target is explained by the model.

Lower MAE and RMSE values indicate smaller prediction errors, while a higher R² generally indicates a better fit.

## **9\. Actual Model Comparison**

The following results were reported for the held-out test set.

| Model | MAE | RMSE | R² |
| ----- | ----- | ----- | ----- |
| Random Forest | 32,803.93 | 50,544.29 | 0.81 |
| Gradient Boosting | 36,602.65 | 53,557.32 | 0.78 |
| Linear Regression | 48,616.44 | 67,534.72 | 0.65 |
| Ridge Regression | 48,616.43 | 67,535.59 | 0.65 |
| Decision Tree | 44,038.44 | 68,568.15 | 0.64 |

**Note:** Confirm the units and scale of MAE and RMSE against the notebook's evaluation code before interpreting these values as dollar amounts. The target variable in the original dataset is measured in units of \$100,000.

## **10\. Key Findings**

* Random Forest achieved the highest reported R² score of 0.81.  
* Random Forest also achieved the lowest reported MAE and RMSE.  
* Gradient Boosting was the second-best model by the reported R² score.  
* Linear Regression and Ridge Regression produced similar results.  
* In this experiment, the tree-based ensemble models performed better than the linear models on the held-out test set.

These findings apply to the models and experimental configuration used in this project; they do not establish that the same ranking will hold for every housing dataset.

## **11\. How to Run in Google Colab**

1. Open `house-price-prediction.ipynb` in Google Colab.  
2. Ensure the required Python libraries are available.  
3. Run the notebook cells sequentially.  
4. Review the model comparison and cross-validation outputs.  
5. Inspect the generated CSV files.

The dataset is loaded through Scikit-learn's dataset utility.

## **12\. Generated Files**

The notebook exports:

* `house_price_model_comparison.csv` — Reported test-set metrics for the evaluated models.  
* `house_price_cross_validation.csv` — Cross-validation results.

## **13\. Limitations**

* The experiment evaluates a limited set of regression algorithms.  
* Extensive hyperparameter optimization was not documented in the provided results.  
* Latitude and longitude were used as numerical features without more advanced spatial feature engineering.  
* Model performance may vary on housing data from other locations or time periods.  
* The dataset's target scale must be considered when interpreting error metrics.

## **14\. Future Improvements**

* Perform systematic hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.  
* Explore additional ensemble models, such as XGBoost or LightGBM.  
* Investigate more advanced geographic features.  
* Add prediction-versus-actual plots and residual analysis.  
* Evaluate model stability using additional validation strategies.

## **15\. Learning Outcomes**

* Building regression pipelines with Scikit-learn.  
* Performing feature engineering using existing dataset variables.  
* Handling missing and infinite values.  
* Comparing regression algorithms using multiple metrics.  
* Applying cross-validation to training data.  
* Interpreting model evaluation results.

## **16\. Author**

**Md. Munna Mia**

CSE Undergraduate \| AI/ML and Cybersecurity

* GitHub: [https\://github.com/munnamia1945](https://github.com/munnamia1945)  
* Portfolio: [https\://www\.munnamia.com](https://www.munnamia.com/)

