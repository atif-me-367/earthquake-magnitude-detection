import streamlit as st
import tensorflow as tf
import numpy as np

# Load the Keras model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Model_v1 .keras")

model = load_model()

st.title("🎈 Earthquake Magnitude Detection")
st.write(
    "This app predicts earthquake magnitudes using a pre-trained model. "
    "Enter the latitude, longitude, and depth to get a prediction."
)

# Input fields for earthquake data
col1, col2, col3 = st.columns(3)

with col1:
    latitude = st.number_input("Latitude", format="%.4f")
    
with col2:
    longitude = st.number_input("Longitude", format="%.4f")
    
with col3:
    depth = st.number_input("Depth (km)", min_value=0.0, format="%.2f")

# Button to trigger prediction
if st.button("Predict Magnitude"):
    try:
        # Prepare input data for the model (3 features)
        input_data = np.array([[latitude, longitude, depth]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Normalize prediction to be less than 10
        # First divide by 10 as currently done
        raw_scaled_magnitude = float(prediction[0][0])/10.0
        # Then ensure it's capped at 10
        display_magnitude = min(raw_scaled_magnitude, 10.0)
        
        # Display the result
        st.success(f"Predicted Earthquake Magnitude: **{display_magnitude:.2f}**")
        
        # Visualize the prediction with a gauge
        st.metric("Magnitude on Richter Scale", f"{display_magnitude:.2f}")
        
    except Exception as e:
        st.error(f"An error occurred while making predictions: {e}")
