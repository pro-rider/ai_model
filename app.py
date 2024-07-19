import joblib
import pandas as pd
import streamlit as st

# Load the model and feature names
model = joblib.load('iris_linear.pkl')
features = joblib.load('iris_cols.pkl')

# Function to make predictions
def predict(new_data):
    pred = model.predict(new_data)
    return pred

# Streamlit UI components
st.title("Flower Classification Model")

# Input fields for user to enter data
sepal_length = st.number_input("Enter sepal length:")
sepal_width = st.number_input("Enter sepal width:")
petal_length = st.number_input("Enter petal length:")
petal_width = st.number_input("Enter petal width:")

# Predict button
if st.button("Predict"):
    # Create a dictionary from input data
    input_data = {
        'sepal_length': [sepal_length],  # Wrap in list to match DataFrame structure
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    }
    
    # Convert dictionary to DataFrame
    new_data = pd.DataFrame(input_data)
    
    # Make prediction
    prediction = predict(new_data[features])  # Assuming 'features' contains the column names in the right order
    
    # Display prediction result
    st.write(f"The predicted flower species is: {prediction[0]}")
