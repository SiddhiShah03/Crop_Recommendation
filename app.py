import streamlit as st
import pickle
import numpy as np

# Load the model and scalers
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

# Custom CSS for styling
st.markdown("""
    <style>
        .main-title {
            font-size: 50px;
            color: #2E8B57;
            text-align: center;
            font-weight: bold;
        }
        .stButton button {
            width: 100%;
            background-color: #007bff;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            margin-top: 20px;
            border-radius: 10px;
        }
        .result-box {
            background-color: #000;
            color: #fff;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">Crop Recommendation System 🌱</h1>', unsafe_allow_html=True)

# Input fields layout
col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input('Nitrogen', min_value=0, max_value=140, value=90)
with col2:
    P = st.number_input('Phosphorus', min_value=0, max_value=145, value=42)
with col3:
    K = st.number_input('Potassium', min_value=0, max_value=205, value=43)

col4, col5, col6 = st.columns(3)

with col4:
    temp = st.number_input('Temperature (°C)', value=25.0)
with col5:
    humidity = st.number_input('Humidity (%)',value=80.0)
with col6:
    ph = st.number_input('Soil pH', min_value=0.0, max_value=14.0,value=6.5)

# Single row for rainfall
rainfall = st.number_input('Rainfall (mm)',value=200.0)

# Predict button
if st.button("Get Recommendation"):
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    
    # Preprocessing the input data
    sc_features = sc.transform(single_pred)
    
    # Prediction
    prediction = model.predict(sc_features)
    
    # Display result
    # Replace this with your actual raw GitHub URL
    image_url = "https://raw.githubusercontent.com/SiddhiShah03/Crop_Recommendation/main/crop.png"
    
    # Display result
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        st.markdown(f"""
        <div class="result-box" style="display: flex; align-items: center; flex-direction: row-reverse;">
            <div style="flex: 1; padding-left: 20px;">
                <h5 style="color: white;">Recommended Crop for Cultivation is:</h5>
                <h2 style="color: white;">{crop}</h2>
                <p style="color: white;">{crop} is the best crop to be cultivated right there</p>
            </div>
            <img src="{image_url}" width="100" style="border-radius: 10px;" />
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Sorry, we could not determine the best crop for the provided data.")
