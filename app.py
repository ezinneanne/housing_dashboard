import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load models and preprocessor
reg_model = joblib.load('regression_model.pkl')
clf_model = joblib.load('classification_model.pkl')
# preprocessor = joblib.load('preprocessor.pkl')

st.title("🏠 Nigerian House Price Prediction Dashboard")
st.markdown("Predict house price and affordability category using trained ML models.")
st.sidebar.header("Enter House Features")

# Sidebar Inputs
def user_input_features():
    bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 3)
    toilets = st.sidebar.slider("Toilets", 1, 10, 3)
    parking_space = st.sidebar.slider("Parking Space", 0, 6, 1)
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
    data = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'toilets': toilets,
        'parking_space': parking_space,
        'title': title,
        'town': town,
        'state': state
    }
    return pd.DataFrame([data])


input_df = user_input_features()


# Prediction Section
st.subheader("🏷️ Predictions")
if st.button("Predict"):
    # Predict Price
    price_pred = reg_model.predict(input_df)[0]
    category_pred = clf_model.predict(input_df)[0]
    category_text = "Affordable" if category_pred == 0 else "Expensive"

    st.success(f"🏷️ Predicted Price: ₦{int(price_pred):,}")
    st.info(f"💡 Price Category: {category_text}")

# Upload Dataset
st.subheader("📊 Price Category Distribution by Town")
uploaded_data = st.file_uploader("Upload your dataset to explore", type=["csv"])
if uploaded_data is not None:
    df = pd.read_csv(uploaded_data)

    if 'town' in df.columns and 'price Category' in df.columns:
        st.write("Data Preview:", df.head())

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x='town', hue='price Category', palette='Set2', ax=ax)
        ax.set_title("Distribution of Price Categories Across Towns")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.warning("Uploaded CSV must contain `town` and `price Category` columns.")





# streamlit run app.py
