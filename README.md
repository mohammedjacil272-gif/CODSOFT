# Iris Flower Classification

This is a simple machine learning project created as part of the CODSOFT Data Science Internship.

The goal of this project is to classify iris flowers into different species based on their measurements.

## Dataset

The dataset used is the Iris dataset.

It contains the following features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target classes:

- Setosa
- Versicolor
- Virginica

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Steps Performed

1. Loaded the dataset from CSV file
2. Checked dataset shape and columns
3. Checked for missing values
4. Checked class distribution
5. Visualized the data using pairplot
6. Converted species names into numeric values
7. Split dataset into training and testing data
8. Trained the model using K-Nearest Neighbors
9. Made predictions
10. Evaluated model performance using accuracy, confusion matrix, and classification report

## Model Used

K-Nearest Neighbors (KNN)

## Results

The model achieved high accuracy in classifying iris flower species.

## Output

The project generates:

- Dataset preview
- Class distribution
- Pairplot visualization
- Confusion matrix
- Classification report
- Accuracy score

## How to Run

Install required libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn
