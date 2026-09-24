import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page Config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    :root {
        --accent: #2952A3;
        --accent-dark: #1D3A75;
        --text-main: #2B2F36;
        --text-muted: #6B7280;
        --surface: #F4F6F9;
        --border: #E2E5EA;
        --good: #2F855A;
        --average: #B7791F;
        --excellent: #2952A3;
        --needs: #B23B3B;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--text-main);
    }

    .stApp {
        background-color: #FFFFFF;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-weight: 600;
        color: var(--text-main);
    }

    .main-header {
        font-size: 2.25rem;
        font-weight: 700;
        text-align: center;
        color: var(--accent-dark);
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }

    .sub-header {
        text-align: center;
        color: var(--text-muted);
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 20px;
        border-radius: 10px;
        color: var(--text-main);
        text-align: center;
    }

    .good { background: #EAF6EF; border: 1px solid #BFE3CE; color: var(--good); }
    .average { background: #FBF3E3; border: 1px solid #EBD8AE; color: var(--average); }
    .excellent { background: #EAF0FA; border: 1px solid #C6D6F0; color: var(--excellent); }
    .needs { background: #FBEAEA; border: 1px solid #EBBDBD; color: var(--needs); }

    /* Buttons */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border-radius: 8px;
        background-color: var(--accent);
        color: #FFFFFF;
        border: none;
    }
    .stButton > button:hover {
        background-color: var(--accent-dark);
        color: #FFFFFF;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }

    /* Metric widget */
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: var(--text-main);
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-muted);
    }
</style>
""", unsafe_allow_html=True)

# Paths are resolved from this file so the app works from any working directory
# (locally and on Streamlit Community Cloud).
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
DATA_PATH = os.path.join(ROOT_DIR, 'dataset', 'student_data.csv')

@st.cache_resource
def load_artifacts():
    with open(f"{MODEL_DIR}/regression_model.pkl", 'rb') as f:
        reg_model = pickle.load(f)
    with open(f"{MODEL_DIR}/classification_model.pkl", 'rb') as f:
        clf_model = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/feature_cols.pkl", 'rb') as f:
        feature_cols = pickle.load(f)
    with open(f"{MODEL_DIR}/encoders.pkl", 'rb') as f:
        encoders = pickle.load(f)
    with open(f"{MODEL_DIR}/model_results.pkl", 'rb') as f:
        results = pickle.load(f)
    return reg_model, clf_model, scaler, feature_cols, encoders, results

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

def performance_category(score):
    if score >= 90: return "Excellent"
    elif score >= 75: return "Good"
    elif score >= 60: return "Average"
    else: return "Needs Improvement"

def risk_level(score, attendance, study_hours):
    if score >= 75 and attendance >= 75 and study_hours >= 4:
        return "Low"
    elif score >= 60 and attendance >= 60:
        return "Medium"
    else:
        return "High"

def predict(student_data):
    reg_model, clf_model, scaler, feature_cols, encoders, _ = load_artifacts()
    
    # Encode
    try:
        extra_encoded = encoders['extra'].transform([student_data['Extracurricular']])[0]
    except Exception:
        extra_encoded = 1 if student_data['Extracurricular'] == 'Yes' else 0
    try:
        family_encoded = encoders['family'].transform([student_data['FamilyBackground']])[0]
    except Exception:
        family_encoded = 0
    
    # Features
    study_attendance_ratio = student_data['StudyHours'] * (student_data['Attendance'] / 100)
    avg_academic = (student_data['PreviousScore'] + student_data['AssignmentScore'] + student_data['InternalScore']) / 3
    sleep_study_balance = student_data['SleepHours'] / (student_data['StudyHours'] + 0.1)
    productivity = student_data['StudyHours'] * (1 - student_data['InternetUsageHours']/10)
    consistency = 1 - (abs(student_data['PreviousScore'] - student_data['InternalScore']) / 100)
    
    data = {
        'StudyHours': student_data['StudyHours'],
        'Attendance': student_data['Attendance'],
        'PreviousScore': student_data['PreviousScore'],
        'AssignmentScore': student_data['AssignmentScore'],
        'InternalScore': student_data['InternalScore'],
        'SleepHours': student_data['SleepHours'],
        'InternetUsageHours': student_data['InternetUsageHours'],
        'Extracurricular_Encoded': extra_encoded,
        'FamilyBackground_Encoded': family_encoded,
        'Study_Attendance_Ratio': study_attendance_ratio,
        'Avg_Academic_Score': avg_academic,
        'Sleep_Study_Balance': sleep_study_balance,
        'Productivity_Score': productivity,
        'Consistency': consistency
    }
    
    X = pd.DataFrame([data])[feature_cols]
    
    reg_is_linear = 'Linear' in str(type(reg_model))
    clf_is_linear = 'Logistic' in str(type(reg_model)) or 'Logistic' in str(type(clf_model))
    
    if reg_is_linear:
        X_reg = pd.DataFrame(scaler.transform(X), columns=feature_cols)
    else:
        X_reg = X
    if clf_is_linear:
        X_clf = pd.DataFrame(scaler.transform(X), columns=feature_cols)
    else:
        X_clf = X
    
    final_score = float(reg_model.predict(X_reg)[0])
    final_score = float(np.clip(final_score, 0, 100))
    
    try:
        pred = clf_model.predict(X_clf)[0]
        if 'label' in encoders and isinstance(pred, (int, np.integer)):
            try:
                perf_cat = encoders['label'].inverse_transform([pred])[0]
            except Exception:
                perf_cat = str(pred)
        else:
            perf_cat = pred
    except Exception:
        perf_cat = performance_category(final_score)
    
    risk = risk_level(final_score, student_data['Attendance'], student_data['StudyHours'])
    
    # Feature importance (tree models expose feature_importances_; linear models are
    # trained on standardized inputs, so |coefficient| is a comparable influence measure)
    importance = None
    if hasattr(reg_model, 'feature_importances_'):
        raw_importance = np.asarray(reg_model.feature_importances_, dtype=float)
    elif hasattr(reg_model, 'coef_'):
        raw_importance = np.abs(np.ravel(reg_model.coef_))
    else:
        raw_importance = None
    if raw_importance is not None and raw_importance.sum() > 0:
        importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': raw_importance / raw_importance.sum()
        }).sort_values('Importance', ascending=False)
    
    return final_score, perf_cat, risk, importance, data

# --- UI ---
st.markdown('<div class="main-header">Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict final academic performance based on study habits & background</p>', unsafe_allow_html=True)

# Load artifacts check
try:
    reg_model, clf_model, scaler, feature_cols, encoders, results = load_artifacts()
    models_loaded = True
except Exception as e:
    st.error(f"Models not found. Please train first: {e}")
    models_loaded = False
    results = None

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Dataset Insights", "📈 Model Performance"])

with tab1:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("📝 Enter Student Details")
        
        study_hours = st.slider("Study Hours per Day", 1.0, 10.0, 6.0, 0.5, help="Daily study hours")
        attendance = st.slider("Attendance (%)", 40, 100, 88, help="Overall attendance percentage")
        previous_score = st.slider("Previous Exam Score", 20, 100, 75, help="Last exam score")
        assignment_score = st.slider("Assignment Score", 30, 100, 82)
        internal_score = st.slider("Internal Score", 25, 100, 78)
        sleep_hours = st.slider("Sleep Hours", 4.0, 10.0, 7.0, 0.5)
        
        st.markdown("**Additional Factors**")
        c1, c2 = st.columns(2)
        with c1:
            internet_usage = st.slider("Internet Usage (hrs/day)", 0.5, 8.0, 3.0, 0.5)
            extracurricular = st.selectbox("Extracurricular", ["Yes", "No"], index=0)
        with c2:
            family_bg = st.selectbox("Family Background", ["Graduate", "High School", "Postgraduate"], index=0)
        
        predict_btn = st.button("🚀 Predict Performance", type="primary", use_container_width=True)
    
    with col2:
        if predict_btn and models_loaded:
            student_input = {
                'StudyHours': study_hours,
                'Attendance': attendance,
                'PreviousScore': previous_score,
                'AssignmentScore': assignment_score,
                'InternalScore': internal_score,
                'SleepHours': sleep_hours,
                'InternetUsageHours': internet_usage,
                'Extracurricular': extracurricular,
                'FamilyBackground': family_bg
            }
            
            score, category, risk, importance, engineered = predict(student_input)
            
            st.subheader("🎯 Prediction Result")
            
            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Predicted Final Score", f"{score:.1f}/100")
            with m2:
                color_map = {"Excellent":"🟣", "Good":"🟢", "Average":"🟡", "Needs Improvement":"🔴"}
                st.metric("Performance", f"{color_map.get(category, '')} {category}")
            with m3:
                risk_emoji = {"Low":"✅ Low", "Medium":"⚠️ Medium", "High":"🚨 High"}
                st.metric("Risk Level", risk_emoji.get(risk, risk))
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Final Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2952A3"},
                    'steps': [
                        {'range': [0, 60], 'color': "#FBEAEA"},
                        {'range': [60, 75], 'color': "#FBF3E3"},
                        {'range': [75, 90], 'color': "#EAF6EF"},
                        {'range': [90, 100], 'color': "#EAF0FA"}],
                    'threshold': {'line': {'color': "#B23B3B", 'width': 4}, 'thickness': 0.75, 'value': 90}
                }
            ))
            fig.update_layout(height=300, font_family="Inter, sans-serif", paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if risk == "High":
                st.error("**High Risk Detected!** - Needs immediate attention.")
                st.write("- Increase study hours to at least 5-6 hrs/day")
                st.write("- Improve attendance above 75%")
                st.write("- Reduce internet distraction")
            elif risk == "Medium":
                st.warning("**Medium Risk** - Can improve with consistent effort.")
                st.write("- Maintain study schedule")
                st.write("- Focus on internal & assignment scores")
            else:
                st.success("**Low Risk - On Track!** Keep up the good work.")
                if score < 90:
                    st.write("- Push for Excellent: increase productivity score")
                    st.write("- Optimize sleep-study balance")
            
            # Feature importance
            if importance is not None:
                st.subheader("📊 What Drives Your Score?")
                fig2 = px.bar(importance.head(8), x='Importance', y='Feature', orientation='h', 
                              title="Relative Feature Influence", color='Importance', color_continuous_scale=['#C6D6F0', '#2952A3'])
                fig2.update_layout(height=350, yaxis={'categoryorder':'total ascending'},
                                   font_family="Inter, sans-serif", paper_bgcolor="white", plot_bgcolor="white")
                st.plotly_chart(fig2, use_container_width=True)
        
        elif not models_loaded:
            st.warning("Train the models first by running `python src/train_model.py`")
        else:
            st.info("👈 Enter details and click **Predict** to see results")
            example_input = {
                'StudyHours': 6, 'Attendance': 88, 'PreviousScore': 75,
                'AssignmentScore': 82, 'InternalScore': 78, 'SleepHours': 7,
                'InternetUsageHours': 3.0, 'Extracurricular': 'Yes', 'FamilyBackground': 'Graduate'
            }
            ex_score, ex_cat, ex_risk, _, _ = predict(example_input)
            st.markdown(f"""
            **Example input:**
            - Study Hours: 6
            - Attendance: 88%
            - Previous Score: 75
            - Assignment Score: 82
            - Internal Score: 78
            - Sleep Hours: 7
            - **Output:** Predicted Score ~{ex_score:.1f} ({ex_cat}, {ex_risk} Risk)
            """)
            # Show sample chart
            df_sample = load_dataset()
            try:
                fig = px.scatter(df_sample, x='StudyHours', y='FinalScore', color='PerformanceCategory',
                                 title="Study Hours vs Final Score (Sample Data)", trendline="ols",
                                 color_discrete_sequence=['#2952A3', '#2F855A', '#B7791F', '#B23B3B'])
            except Exception:
                fig = px.scatter(df_sample, x='StudyHours', y='FinalScore', color='PerformanceCategory',
                                 title="Study Hours vs Final Score (Sample Data)",
                                 color_discrete_sequence=['#2952A3', '#2F855A', '#B7791F', '#B23B3B'])
            fig.update_layout(font_family="Inter, sans-serif", paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📂 Dataset Overview")
    try:
        df = load_dataset()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Students", len(df))
        c2.metric("Avg Final Score", f"{df['FinalScore'].mean():.1f}")
        c3.metric("Avg Attendance", f"{df['Attendance'].mean():.1f}%")
        c4.metric("Avg Study Hours", f"{df['StudyHours'].mean():.1f}")
        
        st.dataframe(df.head(20), use_container_width=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(df, x='FinalScore', nbins=20, title="Final Score Distribution", color_discrete_sequence=['#2952A3'])
            fig.update_layout(font_family="Inter, sans-serif", paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.box(df, x='PerformanceCategory', y='StudyHours', title="Study Hours by Performance", color_discrete_sequence=['#2952A3'])
            fig2.update_layout(font_family="Inter, sans-serif", paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)
        with col_b:
            fig3 = px.scatter(df, x='Attendance', y='FinalScore', color='StudyHours', title="Attendance vs Final Score",
                               color_continuous_scale=['#C6D6F0', '#2952A3'])
            fig3.update_layout(font_family="Inter, sans-serif", paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig3, use_container_width=True)
            corr = df[['StudyHours','Attendance','PreviousScore','AssignmentScore','InternalScore','SleepHours','FinalScore']].corr()
            fig4 = px.imshow(corr, text_auto=True, title="Correlation Heatmap", color_continuous_scale=['#FBEAEA', '#FFFFFF', '#2952A3'])
            fig4.update_layout(font_family="Inter, sans-serif", paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig4, use_container_width=True)
            
    except Exception as e:
        st.error(f"Could not load dataset: {e}")

with tab3:
    st.subheader("🤖 Model Performance")
    if results:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Regression (Final Score Prediction)**")
            reg_df = pd.DataFrame(results['regression']).T
            st.dataframe(reg_df.style.highlight_min(subset=['MAE', 'RMSE'], color='#EAF6EF').highlight_max(subset=['R2'], color='#EAF6EF'), use_container_width=True)
            
            fig = px.bar(reg_df, x=reg_df.index, y='R2', title="R² Score Comparison", color='R2', color_continuous_scale=['#C6D6F0', '#2952A3'])
            fig.update_layout(font_family="Inter, sans-serif", paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Classification (Performance Category)**")
            clf_df = pd.DataFrame(results['classification']).T
            st.dataframe(clf_df.style.highlight_max(subset=['Accuracy'], color='#EAF6EF'), use_container_width=True)
            
            fig2 = px.bar(clf_df, x=clf_df.index, y='Accuracy', title="Accuracy Comparison", color='Accuracy', color_continuous_scale=['#C6D6F0', '#2952A3'])
            fig2.update_layout(font_family="Inter, sans-serif", paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)
        
        best_reg = results.get('best_reg') or max(results['regression'], key=lambda k: results['regression'][k]['R2'])
        best_clf = results.get('best_clf') or max(results['classification'], key=lambda k: results['classification'][k]['Accuracy'])
        st.info(f"🏆 Best Regression: **{best_reg}** | Best Classification: **{best_clf}**")
    else:
        st.warning("No results found. Train models first.")

