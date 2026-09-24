"""
Prediction Module - Used by both CLI and Streamlit app
"""
import pickle
import os
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def load_artifacts(model_dir=MODEL_DIR):
    with open(f"{model_dir}/regression_model.pkl", 'rb') as f:
        reg_model = pickle.load(f)
    with open(f"{model_dir}/classification_model.pkl", 'rb') as f:
        clf_model = pickle.load(f)
    with open(f"{model_dir}/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open(f"{model_dir}/feature_cols.pkl", 'rb') as f:
        feature_cols = pickle.load(f)
    with open(f"{model_dir}/encoders.pkl", 'rb') as f:
        encoders = pickle.load(f)
    return reg_model, clf_model, scaler, feature_cols, encoders

def performance_category(score):
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 60:
        return "Average"
    else:
        return "Needs Improvement"

def risk_level(score, attendance, study_hours):
    if score >= 75 and attendance >= 75 and study_hours >= 4:
        return "Low"
    elif score >= 60 and attendance >= 60:
        return "Medium"
    else:
        return "High"

def predict_student(
    study_hours, attendance, previous_score,
    assignment_score, internal_score, sleep_hours,
    internet_usage=3.0, extracurricular='Yes', family_background='Graduate'
):
    reg_model, clf_model, scaler, feature_cols, encoders = load_artifacts()
    
    # Encode categoricals
    try:
        extra_encoded = encoders['extra'].transform([extracurricular])[0]
    except:
        extra_encoded = 1 if extracurricular == 'Yes' else 0
    
    try:
        family_encoded = encoders['family'].transform([family_background])[0]
    except:
        family_encoded = 0
    
    # Feature engineering same as training
    study_attendance_ratio = study_hours * (attendance / 100)
    avg_academic = (previous_score + assignment_score + internal_score) / 3
    sleep_study_balance = sleep_hours / (study_hours + 0.1)
    productivity = study_hours * (1 - internet_usage / 10)
    consistency = 1 - (abs(previous_score - internal_score) / 100)
    
    # Create DataFrame
    data = {
        'StudyHours': study_hours,
        'Attendance': attendance,
        'PreviousScore': previous_score,
        'AssignmentScore': assignment_score,
        'InternalScore': internal_score,
        'SleepHours': sleep_hours,
        'InternetUsageHours': internet_usage,
        'Extracurricular_Encoded': extra_encoded,
        'FamilyBackground_Encoded': family_encoded,
        'Study_Attendance_Ratio': study_attendance_ratio,
        'Avg_Academic_Score': avg_academic,
        'Sleep_Study_Balance': sleep_study_balance,
        'Productivity_Score': productivity,
        'Consistency': consistency
    }
    
    X = pd.DataFrame([data])[feature_cols]
    
    # Decide scaling
    reg_is_linear = 'Linear' in str(type(reg_model)) or 'Logistic' in str(type(reg_model))
    clf_is_linear = 'Linear' in str(type(clf_model)) or 'Logistic' in str(type(clf_model))
    
    if reg_is_linear:
        X_reg = pd.DataFrame(scaler.transform(X), columns=feature_cols)
    else:
        X_reg = X
        
    if clf_is_linear:
        X_clf = pd.DataFrame(scaler.transform(X), columns=feature_cols)
    else:
        X_clf = X
    
    final_score = reg_model.predict(X_reg)[0]
    final_score = float(np.clip(final_score, 0, 100))
    
    # Classification prediction with decoding support
    try:
        pred = clf_model.predict(X_clf)[0]
        # If model is XGBoost, decode label
        if 'label' in encoders and isinstance(pred, (int, np.integer)):
            try:
                perf_category = encoders['label'].inverse_transform([pred])[0]
            except:
                perf_category = str(pred)
        else:
            perf_category = pred
    except Exception as e:
        print(f"Classification fallback: {e}")
        perf_category = performance_category(final_score)
    
    # Ensure consistency
    auto_category = performance_category(final_score)
    
    risk = risk_level(final_score, attendance, study_hours)
    
    return {
        'predicted_score': round(final_score, 2),
        'performance_category': perf_category,
        'auto_category': auto_category,
        'risk_level': risk,
        'features_used': data
    }

if __name__ == "__main__":
    # Example from prompt
    result = predict_student(
        study_hours=6,
        attendance=88,
        previous_score=75,
        assignment_score=82,
        internal_score=78,
        sleep_hours=7
    )
    print(f"Predicted Final Score: {result['predicted_score']}")
    print(f"Performance: {result['performance_category']}")
    print(f"Risk Level: {result['risk_level']}")
