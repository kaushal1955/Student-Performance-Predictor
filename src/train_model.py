"""
Model Training for Student Performance Prediction
Trains both Regression (Final Score) and Classification (Performance Category)
"""
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed, skipping XGB models")

from data_preprocessing import prepare_data

def train_regression_models(X_train, y_train, X_test, y_test):
    models = {
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(random_state=42, max_depth=8),
        'RandomForest': RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
    
    results = {}
    trained = {}
    
    for name, model in models.items():
        # Use scaled data for Linear, original for tree models
        if name == 'LinearRegression':
            model.fit(X_train['scaled'], y_train)
            y_pred = model.predict(X_test['scaled'])
        else:
            model.fit(X_train['original'], y_train)
            y_pred = model.predict(X_test['original'])
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        trained[name] = model
        print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
    
    return results, trained

def train_classification_models(X_train, y_train, X_test, y_test):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=500, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=8),
        'RandomForest': RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10),
    }
    
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
    
    results = {}
    trained = {}
    
    for name, model in models.items():
        # XGBoost needs encoded labels
        if name == 'XGBoost':
            y_tr = y_train_enc
            y_te = y_test_enc
        else:
            y_tr = y_train
            y_te = y_test
            
        if name == 'LogisticRegression':
            model.fit(X_train['scaled'], y_tr)
            y_pred = model.predict(X_test['scaled'])
        else:
            model.fit(X_train['original'], y_tr)
            y_pred = model.predict(X_test['original'])
        
        acc = accuracy_score(y_te, y_pred)
        results[name] = {'Accuracy': acc}
        trained[name] = model
        print(f"{name} - Accuracy: {acc:.4f}")
    
    return results, trained, le

def main():
    print("Preparing data...")
    data = prepare_data()
    
    X_train_dict = {
        'original': data['X_train'],
        'scaled': data['X_train_scaled']
    }
    X_test_dict = {
        'original': data['X_test'],
        'scaled': data['X_test_scaled']
    }
    
    print("\n--- Training Regression Models ---")
    reg_results, reg_models = train_regression_models(
        X_train_dict, data['y_reg_train'], X_test_dict, data['y_reg_test']
    )
    
    print("\n--- Training Classification Models ---")
    clf_results, clf_models, clf_label_encoder = train_classification_models(
        X_train_dict, data['y_class_train'], X_test_dict, data['y_class_test']
    )
    
    # Select best models
    best_reg_name = max(reg_results, key=lambda x: reg_results[x]['R2'])
    best_clf_name = max(clf_results, key=lambda x: clf_results[x]['Accuracy'])
    
    print(f"\nBest Regression Model: {best_reg_name} (R2={reg_results[best_reg_name]['R2']:.4f})")
    print(f"Best Classification Model: {best_clf_name} (Acc={clf_results[best_clf_name]['Accuracy']:.4f})")
    
    # Save models into <project root>/models (the folder the Streamlit app reads from)
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(model_dir, exist_ok=True)

    artifacts = {
        "regression_model.pkl": reg_models[best_reg_name],
        "classification_model.pkl": clf_models[best_clf_name],
        "scaler.pkl": data['scaler'],
        "feature_cols.pkl": data['feature_cols'],
        "encoders.pkl": {'extra': data['le_extra'], 'family': data['le_family'], 'label': clf_label_encoder},
        "model_results.pkl": {
            'regression': reg_results, 'classification': clf_results,
            'best_reg': best_reg_name, 'best_clf': best_clf_name
        },
    }
    for filename, obj in artifacts.items():
        with open(os.path.join(model_dir, filename), 'wb') as f:
            pickle.dump(obj, f)

    print(f"\nModels saved to {model_dir}/")
    return best_reg_name, best_clf_name

if __name__ == "__main__":
    main()
