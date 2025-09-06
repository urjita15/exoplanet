import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Exoplanet Mass Prediction 🚀")

# --------- Interactive sliders ---------
st.header("Enter Planet and Star Features")
pl_orbper = st.slider("Orbital Period", 0.0, 500.0, 50.0)
pl_rade = st.slider("Planet Radius", 0.0, 20.0, 1.0)
st_teff = st.slider("Star Temperature (K)", 2000, 10000, 5778)
st_rad = st.slider("Star Radius", 0.0, 10.0, 1.0)
st_mass = st.slider("Star Mass", 0.0, 10.0, 1.0)

# Feature engineering
pl_density = pl_rade
st_luminosity = st_rad**2 * (st_teff/5778)**4
pl_to_star_ratio = pl_rade / st_rad

# Prepare input
X_input = pd.DataFrame([[pl_orbper, pl_rade, st_teff, st_rad, st_mass, pl_density, st_luminosity, pl_to_star_ratio]],
                       columns=["pl_orbper","pl_rade","st_teff","st_rad","st_mass","pl_density","st_luminosity","pl_to_star_ratio"])

X_input_scaled = scaler.transform(X_input)

# Prediction
prediction = model.predict(X_input_scaled)
st.write(f"Predicted Planet Mass: {prediction[0]:.2f} Earth masses")

# --------- SHAP feature importance ---------
st.header("Feature Importance")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_input_scaled)
shap.summary_plot(shap_values, X_input_scaled, feature_names=X_input.columns)
st.pyplot(plt.gcf())

# --------- CSV upload ---------
st.header("Upload CSV for Batch Predictions")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_scaled = scaler.transform(df)
    predictions = model.predict(df_scaled)
    df['Predicted Mass'] = predictions
    st.write(df)
