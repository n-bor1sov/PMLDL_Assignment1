import streamlit as st
import requests

# Set the page title
st.title("Sentiment Analysis App")

# Create a text input field
user_input = st.text_input("Enter some text:")

# Create a "Predict" button
if st.button("Predict"):
    # Check if the text input is empty
    if not user_input:
        st.warning("Please enter some text before clicking the 'Predict' button.")
    else:
        # Create the JSON payload
        payload = {
            'text': user_input
        }
        
        # URL of the prediction endpoint
        url = 'http://api:3010/sentiment-analisys'
        
        # Send the POST request
        response = requests.post(url, json=payload)
        
        # Parse the sentiment from the response
        if response.status_code == 200:
            response_data = response.json()
            sentiment = response_data.get('sentiment', 'Unknown')
            
            # Display the sentiment
            st.write("Prediction Result:")
            st.write(f"Sentiment: {sentiment}")
        else:
            st.write("Error: Failed to get a valid response from the API.")