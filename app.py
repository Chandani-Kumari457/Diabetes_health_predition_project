# ===============================
# app.py - Colorful & Attractive GUI
# ===============================

import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load trained model
# -------------------------------
model = pickle.load(open(r"C:\Users\Chandani\Desktop\DMDW_PROJECT\diabetes_model.pkl", "rb"))

# Load dataset for feature names
data = pd.read_csv(r"C:\Users\Chandani\Desktop\DMDW_PROJECT\diabetes.csv")
features = data.drop("Outcome", axis=1).columns.tolist()

# -------------------------------
# Step 2: Page Config
# -------------------------------
st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")
st.markdown("<h1 style='text-align:center; color: darkblue;'>💉 Early Diabetes Risk Prediction System</h1>", unsafe_allow_html=True)

# -------------------------------
# Sidebar for Model Info / Tips
# -------------------------------
st.sidebar.markdown("## 🧠 Model Information")
st.sidebar.markdown("""
- **Model:** Random Forest Classifier  
- **Accuracy:** ~85%  
- **Dataset:** Public Diabetes Dataset  
- Input realistic values for better prediction
""")
st.sidebar.markdown("## 📝 Tips for Inputs")
st.sidebar.markdown("""
- Glucose > 140 → HIGH RISK  
- BMI > 25 → Overweight / Obese  
- Age > 50 → Higher risk  
""") 


# -------------------------------
# Step 3: Example Input Section
# -------------------------------
st.markdown("<h3 style='color:purple;'>Example Input Values</h3>", unsafe_allow_html=True)
st.markdown("""
- Pregnancies: 2  
- Glucose Level: 120  
- Blood Pressure: 70  
- Skin Thickness: 20  
- Insulin Level: 79  
- BMI: 28.5  
- Diabetes Pedigree Function: 0.5  
- Age: 33
""")

# -------------------------------
# Step 4: User Inputs in 2 Columns with colored box
# -------------------------------
st.markdown("<h3 style='color:darkgreen;'>Enter Your Health Details</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    skin = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
    age = st.number_input("Age", min_value=0, step=1)

# -------------------------------
# Step 5: Tabs for Prediction & Feature Importance
# -------------------------------
tab1, tab2 = st.tabs(["Prediction", "Feature Importance"])

with tab1:
    if st.button("Predict"):
        # Make prediction
        result = model.predict([[preg, glucose, bp, skin, insulin, bmi, pedigree, age]])
        prob = model.predict_proba([[preg, glucose, bp, skin, insulin, bmi, pedigree, age]])[0][1]
        
        # Color-coded result box
        if result[0]==1:
            st.markdown(f"<h2 style='color:white; background-color:red; padding:10px;'>Diabetes Risk: HIGH ⚠️ ({prob*100:.1f}%)</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:white; background-color:green; padding:10px;'>Diabetes Risk: LOW ✅ ({prob*100:.1f}%)</h2>", unsafe_allow_html=True)
            st.balloons()
        
        # Additional color-coded warnings
        if bmi > 25:
            st.warning("⚠️ BMI indicates Overweight / Obese")
        if glucose > 140:
            st.error("⚠️ Glucose level HIGH")
        if age > 50:
            st.info("ℹ️ Age > 50, monitor regularly")
        
        st.info("Model Accuracy: ~85% (Random Forest)")

with tab2:
    st.markdown("<h3 style='color:blue;'>Feature Importance (Random Forest)</h3>", unsafe_allow_html=True)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=True)
    st.bar_chart(importance_df.set_index('Feature'))

# -------------------------------
# Step 6: Footer / Notes
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='color:gray;'>Note: Enter realistic values for accurate prediction. This system uses a Random Forest Classifier trained on a public diabetes dataset.</p>", unsafe_allow_html=True)
