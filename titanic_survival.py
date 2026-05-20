import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
print(df["Survived"].value_counts())

df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])
df["Embarked"] = encoder.fit_transform(df["Embarked"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

sns.scatterplot(x=X_test["Age"], y=X_test["Fare"], hue=predictions)
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Titanic Survival Prediction Scatter Plot")
plt.show()