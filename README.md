# Titanic Survival Prediction

## Project Overview

This project predicts whether a passenger survived the Titanic disaster using machine learning techniques. The model is built using Python and deployed using Streamlit for interactive prediction.

---

## Dataset

The project uses the Titanic dataset:

* Source: Kaggle Titanic Dataset
* File: Titanic-Dataset.csv
* Description: Contains passenger information such as age, sex, ticket class, fare, and survival status.

---

## Objective

To build a machine learning model that predicts passenger survival based on given features.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Pickle

---

## Features Used

After preprocessing and feature engineering, the following features are used:

* Age
* Fare
* Sex (encoded)
* FamilySize
* IsAlone
* Pclass (one-hot encoded)
* Embarked (one-hot encoded)

---

## Data Preprocessing

* Missing values handled using median and mode
* Categorical variables encoded
* Feature engineering applied:

  * FamilySize = SibSp + Parch + 1
  * IsAlone feature created
* Unnecessary columns removed

---

## Model Building

* Algorithm: Random Forest Classifier
* Train-test split applied (80-20)
* Model trained on processed dataset
* Accuracy evaluated using accuracy score

---

## Model Performance

The model achieves an accuracy of approximately 83% to 87% depending on random state and parameters.

---

## Files in Repository

* Titanic-Dataset.csv
* train.py
* app.py
* model.pkl
* columns.pkl
* README.md

---

## How to Run the Project

### 1. Train the Model

```bash
python train.py
```

This will generate:

* model.pkl
* columns.pkl

---

### 2. Run Streamlit App

```bash
streamlit run app.py
```

---

## Project Workflow

1. Load dataset
2. Clean and preprocess data
3. Feature engineering
4. Train machine learning model
5. Save model using pickle
6. Load model in Streamlit app
7. Make predictions through user input

---

## Input Features for Prediction

* Age
* Fare
* Sex
* Passenger Class
* Embarked Port
* Family Size

---

## Output

* 1 → Passenger Survived
* 0 → Passenger Did Not Survive

---

## Author

Developed as a machine learning project for practice and learning purposes.

---
