
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config (must be first Streamlit command) ────────
st.set_page_config(
    page_title="Burnout Predictor",
    page_icon="🧠",
    layout="centered"
)

# ── Load the saved model, scaler, encoders ───────────────
@st.cache_resource   # cache so it doesn't reload on every interaction
def load_model():
    model    = joblib.load("model/best_model.pkl")
    scaler   = joblib.load("model/scaler.pkl")
    encoders = joblib.load("model/label_encoders.pkl")
    features = joblib.load("model/feature_names.pkl")
    return model, scaler, encoders, features

model, scaler, encoders, feature_names = load_model()

# ── Header ───────────────────────────────────────────────
st.title("🧠 Tech Employee Burnout Predictor")
st.markdown("""
This app predicts your **burnout level** based on your work habits, mental health scores, and demographics.
Fill in the details below and click **Predict**.
""")
st.markdown("---")

# ── Sidebar info ─────────────────────────────────────────
st.sidebar.title("ℹ️ About this App")
st.sidebar.write("**Dataset:** Depression & Anxiety in Tech (100,000 employees)")
st.sidebar.write("**Model:** Gaussian Naive Bayes (best accuracy: 69.4%)")
st.sidebar.write("**Target:** Burnout Level — Low / Moderate / High / Severe")
st.sidebar.markdown("---")
st.sidebar.write("**Algorithms compared:**")
st.sidebar.write("- KNN → 59.15%")
st.sidebar.write("- Naive Bayes → 69.37% ✅")
st.sidebar.write("- Decision Tree → 65.75%")


# ── Input form ───────────────────────────────────────────
st.subheader("📋 Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 22, 55, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
    years_experience = st.slider("Years of Experience", 0, 30, 5)
    years_at_company = st.slider("Years at Current Company", 0.0, 20.0, 2.0, step=0.5)
    salary_usd = st.number_input("Salary (USD/year)", 40000, 270000, 90000, step=5000)
    work_hours_per_week = st.slider("Work Hours per Week", 35, 72, 47)
    meetings_per_day = st.slider("Meetings per Day", 1.0, 15.0, 5.0, step=0.5)
    sleep_hours_per_night = st.slider("Sleep Hours per Night", 3.0, 10.0, 6.5, step=0.5)
    exercise_days_per_week = st.slider("Exercise Days per Week", 0, 7, 2)
with col2:
    job_role = st.selectbox("Job Role", [
        "Software Engineer", "Full Stack Developer", "Data Scientist",
        "DevOps Engineer", "Product Manager", "UX Designer", "QA Engineer"])
    seniority_level = st.selectbox("Seniority Level", [
        "Junior", "Mid", "Senior", "Lead", "Manager", "Director"])
    company_size = st.selectbox("Company Size", [
        "Startup (1-50)", "Small (51-200)", "Mid (201-1000)", "Large (1000+)"])
    work_mode = st.selectbox("Work Mode", ["Remote", "Hybrid", "On-site"])
    industry = st.selectbox("Industry", [
        "Fintech", "Healthcare Tech", "SaaS / Cloud",
        "E-commerce", "AI / ML", "Cybersecurity"])
    stress_score = st.slider("Stress Score (1–10)", 1.0, 10.0, 6.0, step=0.1)
    phq9_score = st.slider("PHQ-9 Depression Score (0–27)", 0, 27, 8)
    gad7_score = st.slider("GAD-7 Anxiety Score (0–21)", 0, 21, 6)
    deadline_pressure_score = st.slider("Deadline Pressure (1–10)", 1.0, 10.0, 6.0, step=0.1)
st.markdown("---")
st.subheader("🔧 More Details")

col3, col4 = st.columns(2)
with col3:
    team_size = st.slider("Team Size", 2, 50, 10)
    vacation_days_taken = st.slider("Vacation Days Taken (this year)", 0, 30, 10)
    manager_support_score = st.slider("Manager Support Score (1–10)", 1.0, 10.0, 6.0, step=0.1)
    work_life_balance_score = st.slider("Work-Life Balance Score (1–10)", 1.0, 10.0, 5.0, step=0.1)
    job_satisfaction_score = st.slider("Job Satisfaction Score (1–10)", 1.0, 10.0, 5.0, step=0.1)
with col4:
    social_support_score = st.slider("Social Support Score (1–10)", 1.0, 10.0, 5.0, step=0.1)
    autonomy_score = st.slider("Autonomy Score (1–10)", 1.0, 10.0, 6.0, step=0.1)
    country = st.selectbox("Country", ["India", "USA", "UK", "Canada", "Germany", "Australia"])
    therapy_access = st.radio("Have access to therapy?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    uses_therapy = st.radio("Currently using therapy?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    ai_tools_daily = st.radio("Use AI tools daily?", [0, 1], format_func=lambda x: "Yes" if x else "No")
# ── Predict button ───────────────────────────────────────
st.markdown("---")
if st.button("🔮 Predict My Burnout Level", use_container_width=True):
    # Build raw input dictionary (all features the model needs)
    raw_input = {
        'age': age,
        'gender': gender,
        'country': country,
        'job_role': job_role,
        'seniority_level': seniority_level,
        'years_experience': years_experience,
        'years_at_company': years_at_company,
        'company_size': company_size,
        'industry': industry,
        'work_mode': work_mode,
        'salary_usd': salary_usd,
        'work_hours_per_week': work_hours_per_week,
        'meetings_per_day': meetings_per_day,
        'team_size': team_size,
        'sleep_hours_per_night': sleep_hours_per_night,
        'exercise_days_per_week': exercise_days_per_week,
        'vacation_days_taken': vacation_days_taken,
        'therapy_access': therapy_access,
        'uses_therapy': uses_therapy,
        'ai_tools_daily': ai_tools_daily,
        'manager_support_score': manager_support_score,
        'work_life_balance_score': work_life_balance_score,
        'job_satisfaction_score': job_satisfaction_score,
        'social_support_score': social_support_score,
        'deadline_pressure_score': deadline_pressure_score,
        'autonomy_score': autonomy_score,
        'stress_score': stress_score,
        'phq9_score': phq9_score,
        'gad7_score': gad7_score,
        'seeks_mental_health_support': 0,   # default
        'job_change_intention': 0            # default
    }

    # Encode text columns (same way we did in training)
    for col, le in encoders.items():
        if col == 'target':
            continue
        if col in raw_input:
            try:
                raw_input[col] = le.transform([str(raw_input[col])])[0]
            except:
                raw_input[col] = 0   # handle unseen labels
# Convert to DataFrame with correct column order
    input_df = pd.DataFrame([raw_input])[feature_names]

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict
    pred_encoded = model.predict(input_scaled)[0]
    pred_label   = encoders['target'].inverse_transform([pred_encoded])[0]

    # Get probabilities
    pred_proba = model.predict_proba(input_scaled)[0]
    class_labels = encoders['target'].classes_

# Show result
    color_map = {
        "Low": "🟢", "Moderate": "🟡",
        "High": "🟠", "Severe": "🔴"
    }
    emoji = color_map.get(pred_label, "⚪")

    st.success(f"### {emoji} Predicted Burnout Level: **{pred_label}**")

    # Show probability bars for each class
    st.write("**Confidence per class:**")
    for label, prob in zip(class_labels, pred_proba):
        st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")

    # Simple advice
    advice = {
        "Low":      "✅ Great! Keep maintaining your healthy work habits.",
        "Moderate": "⚠️ Watch your stress levels. Take regular breaks and talk to someone.",
        "High":     "🚨 Consider reducing workload and seeking support from your manager or therapist.",
        "Severe":   "🆘 Please seek professional mental health support. Your wellbeing matters most."
    }
    st.info(advice.get(pred_label, ""))

st.markdown("---")
st.caption("Built with ❤️ using Python, scikit-learn, and Streamlit")

