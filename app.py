import streamlit as st
import pickle
import numpy as np

# Load your models/scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))

# Crop dictionary for label mapping
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 
    7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 
    12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 
    17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans", 
    21: "Chickpea", 22: "Coffee"
}

# Streamlit app layout and form
st.title("Crop Recommendation System 🌱")

st.subheader("Input the Agricultural Parameters:")

# Input fields in Streamlit
N = st.number_input('Nitrogen', min_value=0, max_value=140, value=90)
P = st.number_input('Phosphorus', min_value=0, max_value=145, value=42)
K = st.number_input('Potassium', min_value=0, max_value=205, value=43)
temp = st.number_input('Temperature (°C)', value=25.0)
humidity = st.number_input('Humidity (%)', value=80.0)
ph = st.number_input('Soil pH', min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input('Rainfall (mm)', value=200.0)

# Predict button
if st.button("Get Recommendation"):
    # Prepare the input for the model
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Apply the scalers
    sc_features = sc.transform(single_pred)

    # Make prediction
    prediction = model.predict(sc_features)
    
    # Display the result
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        st.success(f"Recommended crop for cultivation: {crop}")
    else:
        st.error("Sorry, we could not determine the best crop for the provided data.")

# Optionally display an image
st.image("crop.png", width=100)  
