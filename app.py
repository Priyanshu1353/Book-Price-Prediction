import streamlit as st
import pandas as pd
import pickle


# Set the background image and text color
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://images.unsplash.com/photo-1495446815901-a7297e633e8d?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: 100vw 100vh;
    background-position: center;  
    background-repeat: no-repeat;
    color: #000000; /* Set text color to blue */
}

</style>
"""

st.markdown(background_image, unsafe_allow_html=True)



# Load the trained model from the pickle file
with open('model2.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Define the main function to create the Streamlit app
def main():
    st.title('Book Price Prediction')

    # Add empty lines for center alignment
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Collect user input for prediction
    user_input = {
        'User Rating': st.slider('User Rating', min_value=3.3, max_value=4.9, step=0.1, value=4.6),
        'Reviews': st.number_input('Reviews', min_value=37, max_value=87841, step=1, value=37),
        'Year': st.slider('Year', min_value=2009, max_value=2019, step=1, value=2011),
        'Genre': st.selectbox('Genre', ['Fiction', 'Non Fiction']),
        'Positive Sentiment': st.slider('Positive Sentiment', min_value=0.0, max_value=0.73, step=0.01, value=0.15),
        'Compound Sentiment': st.slider('Compound Sentiment', min_value=-0.9552, max_value=0.9153, step=0.01, value=0.11)
    }

    if user_input['Genre'] == 'Fiction':
        user_input['Genre'] = 0
    else:
        user_input['Genre'] = 1
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Make prediction
    if st.button('Predict'):
        prediction = rf_model.predict(input_df)
        st.success(f'Predicted Price: {prediction[0]}')

# Run the main function to start the Streamlit app
if __name__ == '__main__':
    main()