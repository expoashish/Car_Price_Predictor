import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess the data
def preprocess_data(df):
    # Add number_of_year feature
    current_year = 2024
    df['number_of_year'] = current_year - df['Year']

    # Initialize label encoders and scaler
    fuel_type_encoder = LabelEncoder()
    seller_type_encoder = LabelEncoder()
    transmission_encoder = LabelEncoder()

    # Fit label encoders
    df['Fuel_Type'] = fuel_type_encoder.fit_transform(df['Fuel_Type'])
    df['Seller_Type'] = seller_type_encoder.fit_transform(df['Seller_Type'])
    df['Transmission'] = transmission_encoder.fit_transform(df['Transmission'])

    # Select numerical columns to scale
    numerical_features = ['Present_Price', 'Kms_Driven', 'number_of_year']

    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Define features and target variable
    X = df[['Present_Price', 'Kms_Driven', 'number_of_year', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
    y = df['Selling_Price']

    return X, y, fuel_type_encoder, seller_type_encoder, transmission_encoder, scaler

# Train the model
def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = DecisionTreeRegressor(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

# Function to handle unseen labels
def safe_transform(encoder, values):
    return [encoder.transform([v])[0] if v in encoder.classes_ else -1 for v in values]
