import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

titanic = pd.read_csv(r'C:\Users\Dear User\Downloads\tested.csv')

print(titanic.head())
print(titanic.info())
print(titanic.describe())
print(titanic.isnull().sum())

sns.countplot(x="Survived",hue="Sex", data=titanic)
plt.show()

sns.countplot(x="Survived",hue="Pclass", data=titanic)
plt.show()

sns.histplot(titanic["Age"].dropna(), kde=True, bins=30)
plt.show()

sns.catplot(x="Pclass", hue="Sex", data=titanic, kind="count")
plt.show()


titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())


titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())


if "Cabin" in titanic.columns:
    titanic = titanic.drop("Cabin", axis=1)


print(titanic.isnull().sum())

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"] + 1

titanic["IsAlone"] = 0 
titanic.loc[titanic["FamilySize"]==1,
"IsAlone"] = 1

titanic["Sex"] = titanic["Sex"].map({"male":0,"female":1})

titanic = pd.get_dummies(titanic, columns=["Embarked"], drop_first=True)

print(titanic.head())
print(titanic.columns)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

columns_to_drop = ["Survived", "Name", "Ticket", "PassengerId"]
titanic_cleaned = titanic.drop(columns_to_drop, axis=1, errors='ignore')
x = titanic_cleaned.select_dtypes(include=['number'])
y = titanic["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Missing values after preprocessing:")
print(titanic.isnull().sum())









