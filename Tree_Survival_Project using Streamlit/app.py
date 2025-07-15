# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and preprocessors
model = joblib.load("bagged_trees_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# App title and description
st.title("🌳 Tree Survival Prediction")
st.markdown("""
This app predicts whether a tree survives using the **Bagged Trees ML model**.

You can either:
- 📝 Enter tree features manually for a single prediction.
- 📂 Upload a CSV file with multiple trees for batch prediction.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choose Prediction Mode:", ["Single Prediction", "Batch Prediction (CSV)"])

if mode == "Single Prediction":
    st.header("📝 Enter Tree Features")

    # User input for each feature
    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.number_input(f"Enter {feature}:", value=0.0)

    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])

    # Scale input
    input_scaled = scaler.transform(input_df)

    if st.button("Predict"):
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        # Fix: convert prediction to integer before inverse_transform
        predicted_class = label_encoder.inverse_transform([int(round(prediction[0]))])[0]
        confidence = np.max(probability) * 100

        st.success(f"🎯 Predicted Tree Survival Event: **{predicted_class}**")
        st.info(f"📊 Confidence: {confidence:.2f}%")

elif mode == "Batch Prediction (CSV)":
    st.header("📂 Upload CSV for Batch Prediction")
    st.markdown("""
    **Instructions:**
    - Your CSV file must have the following columns:  
      `No, Plot, Subplot, Species, Light_ISF, Light_Cat, Core, Soil, Conspecific, Myco, SoilMyco, AMF, Phenolics, Lignin, NSC, Census, Time`
    - Upload your file and get predictions for all rows.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read CSV
        batch_data = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded Data:", batch_data)

        # Scale and predict
        try:
            scaled_batch = scaler.transform(batch_data[feature_names])
            batch_predictions = model.predict(scaled_batch)
            decoded_preds = label_encoder.inverse_transform([int(round(p)) for p in batch_predictions])

            # Add predictions to DataFrame
            batch_data['Predicted_Event'] = decoded_preds

            st.success("🎯 Predictions Completed!")
            st.write(batch_data)

            # Option to download results
            csv_download = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_download,
                file_name='tree_survival_predictions.csv',
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
