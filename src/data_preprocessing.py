"""
Data Preprocessing for Student Performance Prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(ROOT_DIR, "dataset", "student_data.csv")

def load_data(path=DEFAULT_DATA_PATH):
    """Load the dataset (path resolved from the project root, not the working directory)"""
    return pd.read_csv(path)

def clean_data(df):
    """Clean data: handle missing values, outliers"""
    df = df.copy()
    # Fill numeric missing with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical missing with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def feature_engineering(df):
    """Create new features"""
    df = df.copy()
    
    # Encode categorical
    le_extra = LabelEncoder()
    le_family = LabelEncoder()
    
    if 'Extracurricular' in df.columns:
        df['Extracurricular_Encoded'] = le_extra.fit_transform(df['Extracurricular'])
    if 'FamilyBackground' in df.columns:
        df['FamilyBackground_Encoded'] = le_family.fit_transform(df['FamilyBackground'])
    
    # New features
    df['Study_Attendance_Ratio'] = df['StudyHours'] * (df['Attendance'] / 100)
    df['Avg_Academic_Score'] = (df['PreviousScore'] + df['AssignmentScore'] + df['InternalScore']) / 3
    df['Sleep_Study_Balance'] = df['SleepHours'] / (df['StudyHours'] + 0.1)
    df['Productivity_Score'] = df['StudyHours'] * (1 - df['InternetUsageHours']/10)
    df['Consistency'] = 1 - (abs(df['PreviousScore'] - df['InternalScore']) / 100)
    
    return df, le_extra, le_family

def prepare_data(test_size=0.2, random_state=42):
    """Full pipeline: load, clean, engineer, split"""
    df = load_data()
    df = clean_data(df)
    df, le_extra, le_family = feature_engineering(df)
    
    # Features for regression
    feature_cols = [
        'StudyHours', 'Attendance', 'PreviousScore', 'AssignmentScore',
        'InternalScore', 'SleepHours', 'InternetUsageHours',
        'Extracurricular_Encoded', 'FamilyBackground_Encoded',
        'Study_Attendance_Ratio', 'Avg_Academic_Score',
        'Sleep_Study_Balance', 'Productivity_Score', 'Consistency'
    ]
    
    X = df[feature_cols]
    y_reg = df['FinalScore']
    y_class = df['PerformanceCategory']
    
    # Train test split
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
        X, y_reg, y_class, test_size=test_size, random_state=random_state, stratify=y_class
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for readability
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
        'y_reg_train': y_reg_train, 'y_reg_test': y_reg_test,
        'y_class_train': y_class_train, 'y_class_test': y_class_test,
        'scaler': scaler,
        'le_extra': le_extra,
        'le_family': le_family,
        'feature_cols': feature_cols,
        'df': df
    }

if __name__ == "__main__":
    data = prepare_data()
    print("Data Prepared Successfully!")
    print(f"Train Shape: {data['X_train'].shape}")
    print(f"Test Shape: {data['X_test'].shape}")
    print(f"Features: {data['feature_cols']}")
