import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load your saved AI stuff
@st.cache_resource
def load_stuff():
    model = joblib.load('exoplanet_model.pkl')
    le = joblib.load('label_encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, le, scaler

# Title and intro
st.title("🪐 Exoplanet Hunter AI")
st.write("Enter numbers or upload data to classify exoplanets! (Trained on KOI data)")

# Load the AI
model, le, scaler = load_stuff()

# Define the clues (same as training)
features = ['koi_period', 'koi_duration', 'koi_prad', 'koi_teq']

# Easy input: Type numbers
st.header("Quick Hunt: Enter Planet Data")
period = st.number_input("Orbital Period (days)", value=10.0, help="Time for one orbit")
duration = st.number_input("Transit Duration (hours)", value=3.0, help="How long it blocks light")
radius = st.number_input("Planet Radius (Earth sizes)", value=1.5, help="Size compared to Earth")
temp = st.number_input("Equilibrium Temp (K)", value=300.0, help="Planet's average temp")

if st.button("🚀 Hunt for Exoplanet!"):
    # Make input data
    input_data = np.array([[period, duration, radius, temp]])
    input_scaled = scaler.transform(input_data)
    pred_num = model.predict(input_scaled)[0]
    pred_word = le.inverse_transform([pred_num])[0]
    prob = model.predict_proba(input_scaled)[0]
    
    st.success(f"**Result: {pred_word}!**")
    st.write(f"Confidence: {max(prob)*100:.0f}%")
    st.write(f"Full chances: Confirmed {prob[le.classes_.tolist().index('CONFIRMED')]*100:.0f}%, Candidate {prob[le.classes_.tolist().index('CANDIDATE')]*100:.0f}%, False Positive {prob[le.classes_.tolist().index('FALSE POSITIVE')]*100:.0f}%")

# Bonus: Upload CSV
st.header("Advanced: Upload CSV File")
uploaded = st.file_uploader("Choose a CSV file (with columns: koi_period, etc.)", type='csv')
if uploaded is not None:
    # Smart read for NASA files (ignore # comments, skip bad lines)
    df_up = pd.read_csv(uploaded, comment='#', low_memory=False, on_bad_lines='skip')
    if all(col in df_up.columns for col in features):
        X_up = scaler.transform(df_up[features])
        preds = model.predict(X_up)
        df_up['Prediction'] = le.inverse_transform(preds)
        st.dataframe(df_up[['koi_period', 'koi_prad', 'Prediction']].head())  # Show key columns
        st.write("Predictions added! Scroll to see.")
    else:
        st.error("CSV missing required columns. Needs: " + ", ".join(features))

# Model stats
st.header("AI Stats")
st.write("Trained on 7k+ KOI samples. Expected accuracy: 90%+")
st.balloons()  # Fun confetti!