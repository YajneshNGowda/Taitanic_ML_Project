"""
End-to-End Machine Learning Project: Titanic Survival Prediction
Author: AI/ML Intern Application
Date: 2024
Fixed version - Compatible with Windows/Mac/Linux
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
import warnings
import os
warnings.filterwarnings('ignore')

# Configure visualization settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Create output directory for plots (current working directory)
OUTPUT_DIR = os.getcwd()
print(f"Output directory: {OUTPUT_DIR}")


def print_section_header(title):
    """Print formatted section headers"""
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70 + "\n")


def load_and_explore_data():
    """Load and perform initial exploration of the dataset"""
    print_section_header("1. DATA LOADING AND EXPLORATION")
    
    # Load dataset
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    # Display basic information
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nColumn Information:")
    df.info()
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    # Check missing values
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_values,
        'Percentage': missing_percentage
    })
    print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))
    
    return df


def perform_eda(df):
    """Perform Exploratory Data Analysis with visualizations"""
    print_section_header("2. EXPLORATORY DATA ANALYSIS")
    
    # Visualization 1: Survival Distribution
    print("Creating Visualization 1: Survival Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    df['Survived'].value_counts().plot(kind='bar', ax=axes[0], color=['salmon', 'lightblue'])
    axes[0].set_title('Survival Count', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Survived (0 = No, 1 = Yes)')
    axes[0].set_ylabel('Count')
    axes[0].set_xticklabels(['Not Survived', 'Survived'], rotation=0)
    
    df['Survived'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                                       colors=['salmon', 'lightblue'])
    axes[1].set_title('Survival Percentage', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'survival_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: survival_distribution.png")
    
    # Visualization 2: Survival by Gender and Class
    print("Creating Visualization 2: Survival by Gender and Class...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.countplot(data=df, x='Sex', hue='Survived', ax=axes[0], palette='Set2')
    axes[0].set_title('Survival by Gender', fontsize=14, fontweight='bold')
    axes[0].legend(title='Survived', labels=['No', 'Yes'])
    
    sns.countplot(data=df, x='Pclass', hue='Survived', ax=axes[1], palette='Set1')
    axes[1].set_title('Survival by Passenger Class', fontsize=14, fontweight='bold')
    axes[1].legend(title='Survived', labels=['No', 'Yes'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'survival_by_demographics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: survival_by_demographics.png")
    
    # Visualization 3: Age Distribution and Correlation
    print("Creating Visualization 3: Age Distribution and Correlation...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    df[df['Survived']==0]['Age'].hist(bins=30, alpha=0.5, label='Not Survived', 
                                      ax=axes[0], color='red')
    df[df['Survived']==1]['Age'].hist(bins=30, alpha=0.5, label='Survived', 
                                      ax=axes[0], color='green')
    axes[0].set_title('Age Distribution by Survival', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    correlation = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=axes[1])
    axes[1].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'age_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: age_correlation.png")
    
    print("\nKey Observations:")
    print("• Approximately 38% of passengers survived")
    print("• Females had significantly higher survival rates than males")
    print("• First class passengers had better survival rates")
    print("• Children had slightly better survival rates")
    print("• Strong negative correlation between Pclass and Survival")


def preprocess_data(df):
    """Preprocess the dataset"""
    print_section_header("3. DATA PREPROCESSING")
    
    # Create a copy
    data = df.copy()
    
    # Feature Selection - Drop unnecessary columns
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    data = data.drop(columns=columns_to_drop)
    print(f"Dropped columns: {columns_to_drop}")
    
    # Handle missing values
    print("\nHandling missing values...")
    age_median = data['Age'].median()
    data['Age'].fillna(age_median, inplace=True)
    print(f"• Filled missing Age with median: {age_median}")
    
    embarked_mode = data['Embarked'].mode()[0]
    data['Embarked'].fillna(embarked_mode, inplace=True)
    print(f"• Filled missing Embarked with mode: {embarked_mode}")
    
    fare_median = data['Fare'].median()
    data['Fare'].fillna(fare_median, inplace=True)
    print(f"• Filled missing Fare with median: {fare_median}")
    
    # Encode categorical variables
    print("\nEncoding categorical variables...")
    data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
    data = pd.get_dummies(data, columns=['Embarked'], prefix='Embarked', drop_first=True)
    print("• Encoded Sex and Embarked")
    
    # Feature Engineering
    print("\nFeature Engineering...")
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    print("• Created FamilySize and IsAlone features")
    
    print(f"\nFinal preprocessed shape: {data.shape}")
    
    return data


def train_models(X_train, X_test, y_train, y_test):
    """Train multiple machine learning models"""
    print_section_header("4. MODEL TRAINING")
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    print("Training models...\n")
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'model': model
        }
        
        print(f"{name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}\n")
    
    return results, models


def compare_models(results):
    """Compare model performance"""
    print_section_header("5. MODEL COMPARISON")
    
    # Create comparison DataFrame
    results_df = pd.DataFrame({k: {m: v for m, v in v.items() if m != 'model'} 
                               for k, v in results.items()}).T
    results_df = results_df.round(4)
    
    print("Model Performance Comparison:")
    print(results_df)
    
    # Visualize
    results_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved: model_comparison.png")
    
    # Best model
    best_model_name = results_df['Accuracy'].idxmax()
    best_accuracy = results_df['Accuracy'].max()
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Accuracy: {best_accuracy:.4f}")
    
    return results_df


def improve_model(X_train, X_test, y_train, y_test):
    """Apply feature scaling and hyperparameter tuning"""
    print_section_header("6. MODEL IMPROVEMENT")
    
    # Feature Scaling
    print("Applying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Feature scaling completed")
    
    # Hyperparameter Tuning for Random Forest
    print("\nPerforming hyperparameter tuning for Random Forest...")
    print("This may take a few moments...\n")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf_model, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    print("✓ Hyperparameter tuning completed")
    print(f"\nBest Parameters: {best_params}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    
    return best_model, scaler, X_train_scaled, X_test_scaled


def evaluate_final_model(model, X_test, y_test, feature_names):
    """Evaluate the final tuned model"""
    print_section_header("7. FINAL MODEL EVALUATION")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("Final Model: Tuned Random Forest Classifier")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.title('Confusion Matrix - Tuned Random Forest', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrix.png")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
    plt.title('Feature Importance - Tuned Random Forest', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: feature_importance.png")
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())
    
    return accuracy, precision, recall, f1


def print_final_summary(accuracy, precision, recall, f1, best_params):
    """Print final project summary"""
    print_section_header("8. FINAL PROJECT SUMMARY")
    
    print(" DATASET INFORMATION:")
    print("  • Dataset: Titanic Survival Prediction")
    print("  • Total samples: 891")
    print("  • Train/Test split: 80/20")
    
    print("\n FINAL MODEL:")
    print("  • Model: Tuned Random Forest Classifier")
    print(f"  • Best Parameters: {best_params}")
    
    print("\n FINAL PERFORMANCE METRICS:")
    print(f"  • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  • Precision: {precision:.4f}")
    print(f"  • Recall:    {recall:.4f}")
    print(f"  • F1-Score:  {f1:.4f}")
    
    print("\n KEY INSIGHTS:")
    print("  1. Gender (Sex) was the most important predictor of survival")
    print("  2. Passenger class (Pclass) and fare significantly influenced survival")
    print("  3. Females had much higher survival rates than males")
    print("  4. First-class passengers had better survival chances")
    print("  5. Age played a moderate role, with children having slightly better odds")
    
    print("\n IMPROVEMENTS APPLIED:")
    print("  1. Feature Engineering: Created FamilySize and IsAlone features")
    print("  2. Feature Scaling: Standardized all numerical features")
    print("  3. Hyperparameter Tuning: Optimized Random Forest using GridSearchCV")
    
    print("\n CONCLUSION:")
    print(f"  The tuned Random Forest model successfully predicts Titanic survival")
    print(f"  with {accuracy*100:.2f}% accuracy. The model reveals that socio-economic")
    print("  factors (class, fare) and demographics (sex, age) were the primary")
    print("  determinants of survival. This demonstrates the 'women and children first'")
    print("  evacuation protocol and the advantage of higher-class passengers.")
    
    print("\n" + "="*70)
    print("✨ PROJECT COMPLETED SUCCESSFULLY! ✨")
    print("="*70)


def main():
    """Main function to run the entire ML pipeline"""
    print("="*70)
    print("TITANIC SURVIVAL PREDICTION - ML PROJECT")
    print("="*70)
    
    # 1. Load and explore data
    df = load_and_explore_data()
    
    # 2. Perform EDA
    perform_eda(df)
    
    # 3. Preprocess data
    data = preprocess_data(df)
    
    # 4. Prepare train-test split
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print_section_header("DATA SPLIT")
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # 5. Train models
    results, models = train_models(X_train, X_test, y_train, y_test)
    
    # 6. Compare models
    results_df = compare_models(results)
    
    # 7. Improve model
    best_model, scaler, X_train_scaled, X_test_scaled = improve_model(
        X_train, X_test, y_train, y_test
    )
    
    # 8. Evaluate final model
    accuracy, precision, recall, f1 = evaluate_final_model(
        best_model, X_test_scaled, y_test, X.columns
    )
    
    # 9. Print summary
    print_final_summary(accuracy, precision, recall, f1, best_model.get_params())
    
    print("\n📁 Generated Files (in current directory):")
    print("  • survival_distribution.png")
    print("  • survival_by_demographics.png")
    print("  • age_correlation.png")
    print("  • model_comparison.png")
    print("  • confusion_matrix.png")
    print("  • feature_importance.png")


if __name__ == "__main__":
    main()
