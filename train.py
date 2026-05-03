import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("Titanic-Dataset.csv")

# -----------------------
# CLEANING
# -----------------------
df['Embarked'].fillna('S', inplace=True)
df['Age'] = df['Age'].fillna(df['Age'].median())

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Drop useless columns
df.drop(columns=['PassengerId','Name','Ticket','Cabin','SibSp','Parch'], errors='ignore', inplace=True)

# One-hot encoding
df = pd.get_dummies(df, columns=['Pclass','Embarked'], drop_first=True)

# -----------------------
# FEATURES
# -----------------------
X = df.drop('Survived', axis=1)
y = df['Survived']

# Save column order for Streamlit
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

# -----------------------
# TRAIN MODEL
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------
# SAVE MODEL
# -----------------------
pickle.dump(model, open("model.pkl", "wb"))