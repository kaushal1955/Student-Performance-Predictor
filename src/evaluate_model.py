"""
Model Evaluation
"""
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import os

from data_preprocessing import prepare_data

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def load_models(model_dir=MODEL_DIR):
    with open(f"{model_dir}/regression_model.pkl", 'rb') as f:
        reg_model = pickle.load(f)
    with open(f"{model_dir}/classification_model.pkl", 'rb') as f:
        clf_model = pickle.load(f)
    with open(f"{model_dir}/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open(f"{model_dir}/model_results.pkl", 'rb') as f:
        results = pickle.load(f)
    return reg_model, clf_model, scaler, results

def evaluate():
    data = prepare_data()
    reg_model, clf_model, scaler, results = load_models()
    
    # Check if model needs scaled input
    reg_is_linear = 'Linear' in str(type(reg_model))
    clf_is_logistic = 'Logistic' in str(type(clf_model))
    
    X_test_reg = data['X_test_scaled'] if reg_is_linear else data['X_test']
    X_test_clf = data['X_test_scaled'] if clf_is_logistic else data['X_test']
    
    # Regression evaluation
    y_reg_pred = reg_model.predict(X_test_reg)
    mae = mean_absolute_error(data['y_reg_test'], y_reg_pred)
    rmse = mean_squared_error(data['y_reg_test'], y_reg_pred) ** 0.5
    r2 = r2_score(data['y_reg_test'], y_reg_pred)
    
    print("=== REGRESSION EVALUATION ===")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    print("\nModel Comparison:")
    for name, metrics in results['regression'].items():
        print(f"  {name}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.2f}")
    
    # Classification evaluation
    y_clf_pred = clf_model.predict(X_test_clf)
    acc = accuracy_score(data['y_class_test'], y_clf_pred)
    
    print("\n=== CLASSIFICATION EVALUATION ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(data['y_class_test'], y_clf_pred))
    
    print("\nModel Comparison:")
    for name, metrics in results['classification'].items():
        print(f"  {name}: Accuracy={metrics['Accuracy']:.4f}")
    
    # Feature importance if available
    if hasattr(reg_model, 'feature_importances_'):
        print("\n=== FEATURE IMPORTANCE ===")
        importances = reg_model.feature_importances_
        feat_imp = pd.DataFrame({
            'Feature': data['feature_cols'],
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        print(feat_imp)
    
    return {
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'accuracy': acc,
        'reg_pred': y_reg_pred,
        'clf_pred': y_clf_pred
    }

if __name__ == "__main__":
    evaluate()
