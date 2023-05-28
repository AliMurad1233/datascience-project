# datascience-project
# Stroke Prediction

Stroke Prediction Model

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Ensemble Model](#ensemble-model)
- [Prediction](#prediction)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

Stroke is a serious medical condition that requires early detection and intervention for effective treatment. Predictive models can play a crucial role in identifying individuals at high risk of stroke, enabling proactive measures to prevent or mitigate its occurrence.

In this project, we leverage datasets consisting of demographic information, health indicators, and lifestyle attributes to build predictive models for stroke detection. By analyzing this data and training machine learning algorithms, we aim to create models that can accurately classify individuals as either prone to stroke or not.



## Data

- The project utilizes a dataset obtained from the Kaggle competition "Playground Series - Stroke Prediction" (provided in the `train.csv` and `test.csv` files).
- The dataset contains information about individuals' age, gender, work type, smoking status, residence type, average glucose level, body mass index (BMI), and other factors.
- Prior to model training, the dataset was divided into a training set and a test set using the `train_test_split` function from scikit-learn.
- [Download train.csv](/train.csv)
- [Download test.csv](/test.csv)

## Exploratory Data Analysis

Before building the predictive models, it's essential to gain insights into the data through exploratory data analysis. In this section, we examine the characteristics of the dataset and visualize the relationships between different features.

### Dataset Overview

The datasets consist of the following features:

- `id`: Unique identifier for each record.
- `age`: Age of the individual.
- `gender`: Gender of the individual (Male, Female, or Other).
- `hypertension`: Whether the individual has hypertension (1 for yes, 0 for no).
- `heart_disease`: Whether the individual has heart disease (1 for yes, 0 for no).
- `ever_married`: Marital status of the individual (Yes or No).
- `work_type`: Type of work the individual is engaged in.
- `Residence_type`: Type of residence of the individual (Urban or Rural).
- `avg_glucose_level`: Average glucose level in the individual's blood.
- `bmi`: Body mass index (BMI) of the individual.
- `smoking_status`: Smoking status of the individual.
- `stroke`: Target variable indicating whether the individual had a stroke (1 for yes, 0 for no).

### Data Visualization

To gain insights into the data, various visualizations are created:

- Distribution of Categorical Features: Bar plots are used to visualize the distribution of categorical features (`gender`, `work_type`, `smoking_status`, `Residence_type`, `ever_married`, `heart_disease`, and `hypertension`) and the occurrence of strokes.

- Distribution of Numerical Features: Histograms with kernel density estimation (KDE) are used to visualize the distribution of numerical features (`age`, `avg_glucose_level`, and `bmi`) and the occurrence of strokes.

- Relationship Between Categorical Features and Stroke: Grouped bar plots are created to explore the relationship between different categorical features and the occurrence of strokes.

- Correlation Matrix: A heatmap is generated to visualize the correlation between numerical features (`age`, `avg_glucose_level`, and `bmi`).

These visualizations provide valuable insights into the distribution of features and their relationships with the target variable, helping us understand the data better and identify any potential patterns or trends.

For detailed code and visualizations, refer to the `stroke_prediction.ipynb` notebook in this repository.


## Data Preprocessing

- Before training the models, some preprocessing steps were performed on the dataset:

- Numerical features were standardized using the StandardScaler from scikit-learn to ensure they have similar scales.

- **Handling Missing Values**: The dataset was checked for missing values and appropriate strategies were applied, such as imputation or removal of missing data, depending on the feature and the extent of missingness.

- **Encoding Categorical Variables**: Categorical variables, including `gender`, `ever_married`, `work_type`, `Residence_type`, and `smoking_status`, were encoded using one-hot encoding or label encoding to convert them into numerical representations suitable for machine learning models.

- **Feature Scaling**: Numerical features (`age`, `avg_glucose_level`, `bmi`) were scaled using standardization or normalization techniques to ensure all features have a similar scale and prevent any particular feature from dominating the learning process.

- **Train-Test Split**: The preprocessed dataset was split into training and testing sets using the `train_test_split` function from `sklearn.model_selection`. This division allows us to evaluate the performance of the trained models on unseen data.


## Model Training

- Multiple machine learning models were utilized to build the stroke prediction model.
- The models used include Logistic Regression, Random Forest Classifier, K-Nearest Neighbors Classifier, Gaussian Naive Bayes, and Support Vector Machines.
- The models were trained using the training dataset and evaluated using various performance metrics such as accuracy, precision, recall, F1-score, and ROC AUC score.


In this project, we train several machine learning models to predict the occurrence of strokes based on the provided dataset. The following models were used:

- **K-Nearest Neighbors (KNN):**
  - Parameters: leaf_size=30, n_neighbors=15, p=2, weights='distance'
  - Code:
    ```python
    knn = KNeighborsClassifier(leaf_size=30, n_neighbors=15, p=2, weights='distance')
    knn.fit(X_train, y_train)
    ```

- **Logistic Regression:**
  - Parameters: C=1.0, solver="lbfgs", penalty="l2", class_weight="balanced"
  - Code:
    ```python
    logic = LogisticRegression(C=1.0, solver="lbfgs", penalty="l2", class_weight="balanced")
    logic.fit(X_train, y_train)
    ```

- **Random Forest:**
  - Parameters: n_estimators=100, min_samples_leaf=20, max_depth=8, max_samples=0.8, class_weight='balanced'
  - Code:
    ```python
    random = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, max_depth=8, max_samples=0.8, class_weight='balanced')
    random.fit(X_train, y_train)
    ```

- **Gaussian Naive Bayes:**
  - Parameters: var_smoothing=1
  - Code:
    ```python
    gnb = GaussianNB(var_smoothing=1)
    gnb.fit(X_train, y_train)
    ```

- **Ensemble Model:**
  - Code:
    ```python
    models = create_models()
    blender = train_ensemble(models, X_train1, y_train1, X_val1, y_val1)
    ```

Feel free to experiment wi

## Ensemble Model

- An ensemble model was created to combine the predictions of multiple individual models.
- The individual models used in the ensemble were Logistic Regression, Random Forest Classifier, and Gaussian Naive Bayes.
- The ensemble technique involved training the individual models on a portion of the training data and using the predictions from these models as inputs to a logistic regression model, which learned to combine the predictions effectively.

## Prediction

- The trained ensemble model was used to make predictions on the test dataset.
- The test dataset was preprocessed in the same way as the training dataset (e.g., encoding categorical variables, scaling numerical features).
- The final predictions were obtained by applying the ensemble model to the preprocessed test data.
- The predictions were stored in the "submission.csv" file, which contains the stroke predictions for each individual in the test dataset.

## Conclusion

- The stroke prediction model achieved satisfactory performance on the test dataset, as evidenced by the evaluation metrics (e.g., accuracy, precision, recall, F1-score, ROC AUC score).
- The model can be utilized to identify individuals who are at a higher risk of having a stroke, allowing for timely intervention and preventive measures.
- Further improvements can be made by incorporating more sophisticated feature engineering techniques, exploring different ensemble methods, and obtaining a larger and more diverse dataset for training.

## References

- Kaggle competition: [Playground Series - Stroke Prediction](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)
- Documentation for scikit-learn: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
- Documentation for matplotlib: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
- Documentation for seaborn: [https://seaborn.pydata.org/tutorial.html](https://seaborn.pydata.org/tutorial.html)
