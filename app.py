import streamlit as st
import pandas as pd
from model import load_data, preprocess_data, train_model, safe_transform

# Load data and model
df = load_data('car_prediction_data.csv')
X, y, fuel_type_encoder, seller_type_encoder, transmission_encoder, scaler = preprocess_data(df)
model, X_train, X_test, y_train, y_test = train_model(X, y)

# Create dictionaries for text to integer mappings
fuel_type_mapping = {label: index for index, label in enumerate(fuel_type_encoder.classes_)}
seller_type_mapping = {label: index for index, label in enumerate(seller_type_encoder.classes_)}
transmission_mapping = {label: index for index, label in enumerate(transmission_encoder.classes_)}

# Function to preprocess input data
def preprocess_input(df):
    df['number_of_year'] = 2024 - df['Year']
    df['Fuel_Type'] = df['Fuel_Type'].map(fuel_type_mapping)
    df['Seller_Type'] = df['Seller_Type'].map(seller_type_mapping)
    df['Transmission'] = df['Transmission'].map(transmission_mapping)
    numerical_features = ['Present_Price', 'Kms_Driven', 'number_of_year']
    df[numerical_features] = scaler.transform(df[numerical_features])
    return df

# Function to predict car price
def predict_price(model, input_data):
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit web app
def main():
    st.title('Car Price Prediction')

    # Sidebar - User inputs
    st.sidebar.header('User Input Features')

    # Dropdowns for categorical variables
    car_names = df['Car_Name'].unique()
    selected_car = st.sidebar.selectbox('Car Name', car_names)

    fuel_types = list(fuel_type_mapping.keys())
    selected_fuel_type = st.sidebar.selectbox('Fuel Type', fuel_types)

    seller_types = list(seller_type_mapping.keys())
    selected_seller_type = st.sidebar.selectbox('Seller Type', seller_types)

    transmissions = list(transmission_mapping.keys())
    selected_transmission = st.sidebar.selectbox('Transmission', transmissions)

    owners = df['Owner'].unique()
    selected_owner = st.sidebar.selectbox('Owner', owners)

    # Slider for numerical variables
    year = st.sidebar.slider('Year', min_value=2000, max_value=2024, value=2015)
    kms_driven = st.sidebar.number_input('Kilometers Driven', min_value=0, value=10000)
    present_price = st.sidebar.number_input('Present Price (in Lakhs)', min_value=0.0, value=5.0)

    # Process user inputs
    user_inputs = {
        'Car_Name': selected_car,
        'Fuel_Type': selected_fuel_type,
        'Seller_Type': selected_seller_type,
        'Transmission': selected_transmission,
        'Owner': selected_owner,
        'Year': year,
        'Kms_Driven': kms_driven,
        'Present_Price': present_price
    }

    # Convert user inputs into DataFrame
    input_data = pd.DataFrame([user_inputs])

    # Preprocess input data
    if st.sidebar.button('Predict'):
        input_data = preprocess_input(input_data.copy())
        # Ensure the order of features matches the training data
        input_data = input_data[['Present_Price', 'Kms_Driven', 'number_of_year', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
        prediction = predict_price(model, input_data)

        # Display prediction on the main page
        st.markdown(f"<h2 style='text-align: center; color: green;'>Predicted Selling Price: {prediction:.2f} Lakhs</h2>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
