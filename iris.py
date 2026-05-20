import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("IRIS.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
print(df["species"].value_counts())

sns.pairplot(df, hue="species")
plt.show()

encoder = LabelEncoder()
df["species"] = encoder.fit_transform(df["species"])

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, predictions)
print(cm)

print(classification_report(y_test, predictions))

sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()