import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load combined AI
@st.cache_resource
def load_stuff():
    model = joblib.load('combined_model.pkl')
    le = joblib.load('combined_le.pkl')
    scaler = joblib.load('combined_scaler.pkl')
    return model, le, scaler

# Title and intro
st.title("🪐 Combined Exoplanet Hunter AI (KOI + TOI + K2!)")
st.write("Enter numbers or upload data to classify exoplanets! Trained on 20k+ samples from 3 NASA missions.")

# Load the AI
model, le, scaler = load_stuff()

# Define the clues (same as training)
features = ['koi_period', 'koi_prad', 'koi_teq']  # 3 features now

# Easy input: Type numbers (no duration)
st.header("Quick Hunt: Enter Planet Data")
period = st.number_input("Orbital Period (days)", value=10.0, help="Time for one orbit")
radius = st.number_input("Planet Radius (Earth sizes)", value=1.5, help="Size compared to Earth")
temp = st.number_input("Equilibrium Temp (K)", value=300.0, help="Planet's average temp")

if st.button("🚀 Hunt for Exoplanet!"):
    # Make input data (3 features)
    input_data = np.array([[period, radius, temp]])
    input_scaled = scaler.transform(input_data)
    pred_num = model.predict(input_scaled)[0]
    pred_word = le.inverse_transform([pred_num])[0]
    prob = model.predict_proba(input_scaled)[0]
    
    st.success(f"**Result: {pred_word}!**")
    st.write(f"Confidence: {max(prob)*100:.0f}%")
    
    # Safe prob display (handles class order)
    class_names = le.classes_
    probs_dict = {name: prob[i] for i, name in enumerate(class_names)}
    st.write(f"Full chances: Confirmed {probs_dict['CONFIRMED']*100:.0f}%, Candidate {probs_dict['CANDIDATE']*100:.0f}%, False Positive {probs_dict['FALSE POSITIVE']*100:.0f}%")
    
    # Bonus: Bar chart for probs
    st.bar_chart(probs_dict)

# Bonus: Upload CSV
st.header("Advanced: Upload CSV File")
uploaded = st.file_uploader("Choose a CSV file (with columns: koi_period, koi_prad, koi_teq, etc.)", type='csv')
if uploaded is not None:
    # Smart read for NASA files (ignore # comments, skip bad lines)
    df_up = pd.read_csv(uploaded, comment='#', low_memory=False, on_bad_lines='skip')
    if all(col in df_up.columns for col in features):
        X_up = scaler.transform(df_up[features])
        preds = model.predict(X_up)
        df_up['Prediction'] = le.inverse_transform(preds)
        st.dataframe(df_up[features + ['Prediction']].head())  # Show all features + prediction
        st.write("Predictions added! Scroll to see.")
    else:
        st.error("CSV missing required columns. Needs: " + ", ".join(features))

# Model stats
st.header("AI Stats")
st.write("Trained on 20k+ samples from KOI, TOI, K2. Expected accuracy: 95%+ (Random Forest)")
st.balloons()  # Fun confetti!