import streamlit as st

# -----------------------------
# Student Academic Performance Predictor
# Created by Chakragiri Karthik
# -----------------------------

st.set_page_config(page_title="📊 Student Academic Performance Predictor", page_icon="🎓")

st.title("📊 Student Academic Performance Predictor")
st.markdown("**Created by:** Chakragiri Karthik")
st.markdown("---")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title of the app
st.title("🎓 Student Performance Prediction")
st.write("This app predicts if a student will Pass or Fail based on given details.")

# Sidebar for user input
st.sidebar.header("Enter Student Details")

def user_input_features():
    hours_study = st.sidebar.slider("Hours of Study per Day", 0, 12, 4)
    attendance = st.sidebar.slider("Attendance (%)", 0, 100, 75)
    past_score = st.sidebar.slider("Past Exam Score (%)", 0, 100, 70)
    extra_activities = st.sidebar.selectbox("Extra-Curricular Activities", ("Yes", "No"))
    health_status = st.sidebar.selectbox("Health Status", ("Good", "Average", "Poor"))
    
    data = {
        "Hours_Study": hours_study,
        "Attendance": attendance,
        "Past_Score": past_score,
        "Extra_Activities": 1 if extra_activities == "Yes" else 0,
        "Health_Status": 2 if health_status == "Good" else (1 if health_status == "Average" else 0)
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Load or create dataset
@st.cache_data
def load_data():
    np.random.seed(42)
    data = pd.DataFrame({
        "Hours_Study": np.random.randint(1, 10, 200),
        "Attendance": np.random.randint(50, 100, 200),
        "Past_Score": np.random.randint(40, 100, 200),
        "Extra_Activities": np.random.randint(0, 2, 200),
        "Health_Status": np.random.randint(0, 3, 200),
    })
    data["Result"] = np.where(
        (data["Hours_Study"] > 4) & (data["Attendance"] > 70) & (data["Past_Score"] > 60),
        "Pass", "Fail"
    )
    return data

df = load_data()

# Train model
X = df.drop("Result", axis=1)
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
prediction = model.predict(input_df)
pred_proba = model.predict_proba(input_df)

# Show results
st.subheader("Prediction probability")
st.markdown("**Note:** The table below shows the prediction confidence for each outcome. For example, if 'Pass' is 0.77 and 'Fail' is 0.23, the model predicts 'Pass' with 77% confidence.")
# Get class labels from the model
class_labels = model.classes_



st.write(proba_df)
st.write(f"**The student will likely:** {prediction[0]}")

st.write(pred_proba)

# Model accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {acc*100:.2f}%")

st.markdown("---")
st.markdown("Developed using **Python & Streamlit** by Chakragiri Karthik")


