# 🎓 Student Performance Predictor

A machine learning project that predicts a student's final score and performance category from study habits, attendance, past scores and background — with an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-red)
![ML](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost-orange)

## 🎯 What it does

Inputs: study hours, attendance, previous / assignment / internal scores, sleep hours, internet usage, extracurricular activity and family background.

Outputs: predicted final score (0–100), performance category, risk level and recommendations.

```
Input:  Study Hours = 6, Attendance = 88%, Previous = 75, Assignment = 82, Internal = 78, Sleep = 7
Output: Predicted Final Score ≈ 80.5  |  Performance: Good  |  Risk Level: Low
```

Performance categories: **90–100** Excellent · **75–89** Good · **60–74** Average · **below 60** Needs Improvement.

Risk level: **Low** = score ≥ 75, attendance ≥ 75% and study hours ≥ 4 · **Medium** = score ≥ 60 and attendance ≥ 60% · **High** = otherwise.

## 📊 Models and results

Trained on 500 synthetic student records (80/20 stratified split). The best model of each type is saved and used by the app.

**Regression (final score)**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** (used) | 4.04 | 5.09 | 0.679 |
| Random Forest | 4.34 | 5.64 | 0.605 |
| Gradient Boosting | 4.57 | 5.78 | 0.585 |
| XGBoost | 4.73 | 5.98 | 0.556 |
| Decision Tree | 5.79 | 7.33 | 0.333 |

**Classification (performance category)**

| Model | Accuracy |
|---|---|
| **Logistic Regression** (used) | 0.69 |
| Random Forest | 0.69 |
| XGBoost | 0.64 |
| Decision Tree | 0.62 |

> The dataset is synthetic and small, and the classes are imbalanced (only 12 "Excellent" students), so treat these numbers as a demo, not as real-world accuracy.

## 🖥️ Dashboard features

- Sliders for every input, with prediction gauge chart
- Performance and risk badges with recommendations
- Relative feature influence chart (based on model coefficients / importances)
- Dataset Insights tab (distributions, correlations)
- Model Performance tab (metrics for every model tried)

## 📁 Project structure

```
Student-Performance-Predictor/
├── app/
│   └── app.py                 # Streamlit dashboard (deployment entry point)
├── dataset/
│   └── student_data.csv       # 500 students, synthetic data
├── models/                    # Trained model files (committed so the app runs without training)
├── notebooks/
│   └── EDA.ipynb              # Exploratory data analysis
├── src/
│   ├── data_preprocessing.py  # Cleaning + feature engineering
│   ├── train_model.py         # Train and save all models
│   ├── evaluate_model.py      # Evaluation metrics
│   └── prediction.py          # Command-line prediction example
├── .streamlit/config.toml     # Theme
├── requirements.txt           # Runtime dependencies (used for deployment)
├── requirements-dev.txt       # Extra packages for training / notebook
├── run.sh / run.bat           # One-command local setup + launch
└── README.md
```

## 🚀 Run locally (VS Code)

```bash
# 1. Create and activate a virtual environment (Python 3.11 recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements-dev.txt

# 3. Run the app (the trained models are already included)
streamlit run app/app.py
```

The app opens at http://localhost:8501. Or just run `./run.sh` (Mac/Linux) / `run.bat` (Windows).

**Retraining (optional):**

```bash
python src/train_model.py      # writes new files into models/
python src/evaluate_model.py
```

> ⚠️ The `.pkl` files in `models/` must be loaded with the same scikit-learn version they were saved with. `requirements.txt` pins `scikit-learn==1.4.2` for this reason. If you retrain with a different version, commit the new `.pkl` files and update the pin.

## ☁️ Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** and pick the repository and branch.
3. Set **Main file path** to `app/app.py`.
4. Open **Advanced settings** and choose **Python 3.11** (the pinned package versions do not support Python 3.13, which is the default). Note that a `runtime.txt` file is *not* used by Community Cloud.
5. Click **Deploy**.

## 🛠️ Tech stack

Python · Pandas · NumPy · Scikit-learn · XGBoost (training only) · Plotly · Statsmodels · Streamlit · Jupyter
