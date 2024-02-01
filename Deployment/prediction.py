import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the GRU model
model = load_model('model_gru_2')

def run():

    image = Image.open('twittersentiment.jpg')
    st.image(image, caption = 'Twitter Sentiment')
    
    with st.form('sentiment_prediction'):
        # Field Input Text
        input_text = st.text_area('Input Text', '', help='Enter the text for sentiment prediction')
        
        # Create a submit button
        submitted = st.form_submit_button('Predict')

    # Inference
    if submitted:
        # Make a prediction using the model
        # Convert the input text to lowercase (optional)
        input_text = input_text.lower()

        # Make a prediction using the model
        predictions = model.predict(np.array([input_text]))

        # Map predicted class to labels
        predicted_class = np.argmax(predictions[0])
        class_labels = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
        predicted_label = class_labels[predicted_class]

        # Display the results
        st.write('## Sentiment Prediction:')
        st.write('Input Text:', input_text)
        st.write('Predicted Class:', predicted_class)
        st.write('Predicted Label:', predicted_label)
        st.write('Prediction Probabilities:', predictions[0])

if __name__ == '__main__':
    run()
