# CODSOFT Data Science Internship Projects

This repository contains projects completed as part of the CODSOFT Data Science Internship.

---

# Task 1: Iris Flower Classification

## Task Description

The objective of this project is to build a machine learning model that can classify iris flowers into different species based on their measurements.

The dataset contains the following features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The model predicts the flower species:

- Setosa
- Versicolor
- Virginica

This is a classification machine learning problem.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Model Used

- K-Nearest Neighbors (KNN)

## Steps Performed

- Loaded the Iris dataset
- Checked dataset structure
- Checked missing values
- Visualized data using pairplot
- Encoded species labels
- Split dataset into training and testing sets
- Trained the machine learning model
- Made predictions
- Evaluated performance

## Output

- Dataset preview
- Dataset summary
- Class distribution
- Pairplot visualization
- Accuracy score
- Confusion matrix
- Classification report

---

# Task 2: Titanic Survival Prediction

## Task Description

The objective of this project is to build a machine learning model that predicts whether a passenger survived the Titanic disaster based on passenger details.

The dataset includes information such as:

- Passenger Class
- Gender
- Age
- Number of Siblings/Spouses
- Number of Parents/Children
- Fare
- Embarked Port

The target prediction is:

- Survived
- Did Not Survive

This is a binary classification machine learning problem.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Model Used

- Logistic Regression

## Steps Performed

- Loaded the Titanic dataset
- Checked dataset structure
- Checked missing values
- Removed unnecessary columns
- Filled missing values
- Encoded categorical columns
- Split dataset into training and testing sets
- Trained Logistic Regression model
- Made predictions
- Evaluated performance
- Created scatter plot visualization

## Output

- Dataset preview
- Missing value analysis
- Survival class distribution
- Accuracy score
- Confusion matrix
- Classification report
- Scatter plot visualization

---

# Repository Structure

```text
CODSOFT/
│
├── Task1_Iris_Classification/
│   ├── IRIS.csv
│   ├── iris_classification.py
│
├── Task2_Titanic_Survival/
│   ├── Titanic-Dataset.csv
│   ├── titanic_prediction.py
│
└── README.md
