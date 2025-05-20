import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
reg_model = joblib.load('regression_model.pkl')
clf_model = joblib.load('classification_model.pkl')

# Page title
st.title("🏠 Nigerian House Price Prediction Dashboard")
st.markdown("Use this app to predict house price and affordability category based on selected features.")

# Sidebar Inputs
st.sidebar.header("Enter House Features")

def user_input_features():
    bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 3)
    toilets = st.sidebar.slider("Toilets", 1, 10, 3)
    parking_space = st.sidebar.slider("Parking Space", 0, 6, 1)
    title = st.sidebar.selectbox("House Type", [
        "Detached Duplex", "Semi Detached Duplex", "Terraced Duplexes", "Block of Flats",
        "Detached Bungalow", "Terraced Bungalow", "Semi Detached Bungalow"
    ])
    town = st.sidebar.selectbox("Town", [
        "Lekki", "Ajah", "Mabushi", "Katampe", "Surulere", "Guzape District", "Gwarinpa", "Ikoyi",
        "Magodo", "Ibeju Lekki", "Ogudu", "Lokogoma District", "Oredo", "Victoria Island (VI)",
        "Mowe Ofada", "Epe", "Arepo", "Simawa", "Life Camp", "Port Harcourt", "Ifako-Ijaiye", "Isolo",
        "Asokoro District", "Jabi", "Karmo", "Maitama District", "Ojo", "Ibadan", "Gudu", "Kukwaba",
        "Enugu", "Owerri Muncipal", "Mbora (Nbora)", "Lugbe", "Dakwo", "Isheri North", "Karu",
        "Ikorodu", "Wuye", "Wuse", "Galadimawa", "Alimosho", "Yaba", "Ifo", "Maryland", "Kaduna South",
        "Ikotun", "Sango Ota", "Garki", "Mowe Town", "Wuse 2", "Magboro", "Ipaja", "Aba", "Ojodu",
        "Ogijo", "Owerri West", "Apo", "Kaura", "Agege", "Kurudu", "Gbagada", "Asaba"
    ])
    state = st.sidebar.selectbox("State", [
        "Lagos", "Abuja", "Delta", "Imo", "Abia", "Nasawara", "Kaduna", "Oyo", "Rivers", "Ogun",
        "Adamawa", "Anambra", "Bauchi", "Benue", "Borno", "Cross River", "Jigawa", "Ebonyi",
        "Enugu", "Gombe", "Kano", "Katsina", "Kwara", "Kebbi", "Kogi", "Edo", "Plateau", "Niger",
        "Osun", "Ekiti", "Taraba", "Zamfara", "Yobe", "Akwa Ibom", "Bayelsa", "Ondo"
    ])

    return pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'toilets': toilets,
        'parking_space': parking_space,
        'title': title,
        'town': town,
        'state': state
    }])

input_df = user_input_features()

# Prediction Section
st.subheader("🏷️ Predictions")
if st.button("Predict"):
    try:
        price_pred = reg_model.predict(input_df)[0]
        category_pred = clf_model.predict(input_df)[0]
        category_text = "Affordable" if category_pred == 0 else "Expensive"

        st.success(f"🏷️ Predicted Price: ₦{int(price_pred):,}")
        st.info(f"💡 Price Category: {category_text}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# =======================
# Upload and Visualization Section
# =======================
st.subheader("📊 Price Category Distribution")

uploaded_file = st.file_uploader("Upload your CSV dataset to visualize category distributions", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'state' in df.columns and 'price Category' in df.columns:
        # Geopolitical zone mapping
        geopolitical_zones = {
            'North Central': ['Benue', 'Kogi', 'Kwara', 'Nasarawa', 'Niger', 'Plateau', 'Abuja'],
            'North East': ['Adamawa', 'Bauchi', 'Borno', 'Gombe', 'Taraba', 'Yobe'],
            'North West': ['Jigawa', 'Kaduna', 'Kano', 'Katsina', 'Kebbi', 'Sokoto', 'Zamfara'],
            'South East': ['Abia', 'Anambra', 'Ebonyi', 'Enugu', 'Imo'],
            'South South': ['Akwa Ibom', 'Bayelsa', 'Cross River', 'Delta', 'Edo', 'Rivers'],
            'South West': ['Ekiti', 'Lagos', 'Ogun', 'Ondo', 'Osun', 'Oyo']
        }

        def get_zone(state):
            for zone, states in geopolitical_zones.items():
                if state in states:
                    return zone
            return "Unknown"

        df['zone'] = df['state'].apply(get_zone)

        # Plot by zone
        zone_plot_data = df.groupby(['zone', 'price Category']).size().reset_index(name='count')
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=zone_plot_data, x='zone', y='count', hue='price Category', ax=ax1)
        ax1.set_title('Price Category Distribution Across Geopolitical Zones')
        ax1.set_ylabel("Number of Properties")
        ax1.set_xlabel("Geopolitical Zone")
        plt.xticks(rotation=45)
        st.pyplot(fig1)
    else:
        st.warning("Dataset must contain 'state' and 'price Category' columns.")


# streamlit run app.py
