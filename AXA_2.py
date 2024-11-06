#!/usr/bin/env python
# coding: utf-8

# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

df = pd.read_csv('depression_data.csv')

# Prepare the data
X = df.drop(['Name', 'History of Mental Illness'], axis=1)
y = df['History of Mental Illness']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = ['Age', 'Number of Children', 'Income']
categorical_features = ['Marital Status', 'Education Level', 'Smoking Status', 
                        'Physical Activity Level', 'Employment Status', 
                        'Alcohol Consumption', 'Dietary Habits', 'Sleep Patterns', 
                        'History of Substance Abuse', 'Family History of Depression', 
                        'Chronic Medical Conditions']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# Create pipelines
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit and evaluate models
models = [lr_pipeline, rf_pipeline]
model_names = ['Logistic Regression', 'Random Forest']

for name, model in zip(model_names, models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Feature importance for Random Forest
try:
    rf_feature_importance = rf_pipeline.named_steps['classifier'].feature_importances_
    feature_names = (numeric_features + 
                     preprocessor.named_transformers_['cat']
                     .get_feature_names_out(categorical_features).tolist())

    importance_df = pd.DataFrame({'feature': feature_names, 'importance': rf_feature_importance})
    importance_df = importance_df.sort_values('importance', ascending=False)
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
except Exception as e:
    print(f"An error occurred while getting feature importances: {str(e)}")

# Print class distribution
print("\nClass Distribution:")
print(y.value_counts(normalize=True))

