import streamlit as st
import pandas as pd
import joblib

# Load models and preprocessor
reg_model = joblib.load('rf_regressor.pkl')
clf_model = joblib.load('rf_classifier.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.title("🏠 Nigerian House Price Prediction Dashboard")

# Sidebar Inputs
st.sidebar.header("Property Features")
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 3)
toilets = st.sidebar.slider("Toilets", 1, 10, 3)
parking_space = st.sidebar.slider("Parking Space", 0, 10, 2)
title = st.sidebar.selectbox("House Type", ["Detached Duplex", "Semi Detached Duplex", "Terraced Duplexes",
                                             "Block of Flats", "Detached Bungalow", "Terraced Bungalow","Semi Detached Bungalow"])
town = st.sidebar.selectbox("Town", ["Lekki", "Ajah", "Mabushi", "Katampe","Surulere", "Guzape District", "Gwarinpa", "Ikoyi",
                                     "Magodo", "Ibeju Lekki", "Ogudu", "Lokogoma District","Oredo", "Victoria Island (VI)", "Mowe Ofada", "Epe",
                                     "Arepo", "Simawa", "Life Camp", "Port Harcourt","Ifako-Ijaiye", "Isolo", "Asokoro District", "Jabi",
                                     "Karmo", "Maitama District", "Ojo", "Ibadan","Gudu", "Kukwaba", "Enugu", "Owerri Muncipal", 
                                     "Mbora (Nbora)", "Lugbe", "Dakwo", "Isheri North", "Karu", "Ikorodu", "Wuye", 
                                     "Wuse","Galadimawa", "Alimosho", "Yaba", "Ifo", "Maryland", "Kaduna South", "Ikotun", "Sango Ota",
                                     "Garki", "Mowe Town", "Wuse 2", "Magboro","Ipaja", "Aba", "Ojodu", "Ogijo",
                                     "Owerri West", "Apo", "Kaura", "Jabi","Agege", "Kurudu", "Gbagada", "Asaba"])
state = st.sidebar.selectbox("State", ["Lagos", "Abuja","Delta","Imo","Abia","Nasawara","Kaduna","Oyo","Rivers",
                                       "Ogun","Adamawa","Anambra","Bauchi","Benue","Borno","Cross River","Jigawa",
                                       "Ebonyi","Enugu","Gombe","Jigawa","Kano","Katsina","Kwara","Kebbi","Kogi","Edo",
                                       "Plateau","Niger","Osun","Ekiti","Taraba","Zamfara","Yobe","Akwa Ibom",
                                       "Bayelsa","Ondo"])

# Make prediction
input_data = pd.DataFrame({
    'bedrooms': [int(bedrooms)],
    'bathrooms': [int(bathrooms)],
    'toilets': [int(toilets)],
    'parking_space': [int(parking_space)],
    'title': [title],
    'town': [town],
    'state': [state]
})


# Predictions
price_pred = reg_model.predict(input_data)[0]
category_pred = clf_model.predict(input_data)[0]

# Results
st.subheader("🏷️ Prediction")
st.write(f"Estimated Price: ₦{int(price_pred):,}")
st.write(f"Price Category: {'Expensive' if category_pred == 1 else 'Affordable'}")
