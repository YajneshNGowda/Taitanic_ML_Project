#Taitanic_ML_Project

##Problem Statement

Given passenger information, predict whether the passenger survived (1) or not (0).
This is a binary classification problem using supervised learning.

##Project Structure

titanic-survival-prediction/
│
├── titanic_ml_project.py
├── README.md
├── survival_distribution.png
├── survival_by_demographics.png
├── age_correlation.png
├── model_comparison.png
├── confusion_matrix.png
├── feature_importance.png


##Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

##Dataset

Dataset used: Titanic dataset
Source:
https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
Number of records: 891
Number of features: 12 (before preprocessing)

##Exploratory Data Analysis (EDA)

The following insights were found:
Around 38% of passengers survived
Females had much higher survival rates than males
First-class passengers survived more than others
Children had better survival chances
Passenger class and fare strongly influenced survival
Generated visualizations:
Survival distribution
Survival by gender and class
Age distribution
Correlation heatmap

##Data Preprocessing

Steps applied:
Dropped irrelevant columns (PassengerId, Name, Ticket, Cabin)
Filled missing values:
Age → Median
Embarked → Mode
Fare → Median
Encoded categorical variables
Feature Engineering:
FamilySize
IsAlone

