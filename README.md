# 🚢 Titanic Survival Prediction - Machine Learning Project

A comprehensive machine learning project that predicts passenger survival on the Titanic using logistic regression and data analysis.

## 📊 Project Overview
This project analyzes the famous Titanic dataset to build a predictive model for passenger survival. The workflow includes data exploration, feature engineering, and machine learning modeling.

## 🛠 Technical Implementation

### Data Preprocessing
- **Missing Values**: Handled Age and Fare missing values with median imputation
- **Feature Engineering**: 
  - Created `FamilySize` from SibSp and Parch
  - Created `IsAlone` flag for solo passengers
  - Encoded categorical variables (Sex, Embarked)
- **Data Cleaning**: Removed unnecessary columns (Cabin, Name, Ticket)

### Machine Learning
- **Model**: Logistic Regression with 200 max iterations
- **Train-Test Split**: 80-20 split with random_state=42
- **Evaluation**: Accuracy score and confusion matrix

### Exploratory Data Analysis (EDA)
- Survival rates by gender and passenger class
- Age distribution analysis
- Passenger class vs gender visualization

## 📈 Results
The model achieves competitive accuracy in predicting passenger survival based on key features like gender, age, fare, and family size.

## 🚀 How to Run
1. Ensure you have the required libraries: `pandas`, `seaborn`, `matplotlib`, `scikit-learn`
2. Download the Titanic dataset (tested.csv)
3. Run the script:
```bash
python titanic2.py
