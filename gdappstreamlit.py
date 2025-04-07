import streamlit as st
from tensorflow.keras.models import load_model
import gdown
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Function to download the model from Google Drive if not already downloaded
def download_model():
    file_id = '1J2-y7b73HhO7dIp8zhgWXjtzpEBFVQAB'  # Replace with your actual file ID
    output = 'casting_defect_model.h5'
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)

# Load the model
if not os.path.exists('casting_defect_model.h5'):
    st.write("Downloading the model...")
    download_model()
model = load_model('casting_defect_model.h5')

# Display a message when the model is ready
st.write("Model is ready to use!")

# Let the user upload an image for prediction
uploaded_image = st.file_uploader("Upload a Casting Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the image using PIL
    image = Image.open(uploaded_image)
    
    # Display the uploaded image in the app
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image for the model (resize, normalize, etc.)
    image = image.resize((224, 224))  # Adjust the size depending on the model's input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension for the model
    
    # Button to trigger the prediction
    if st.button("Predict"):
        # Make prediction using the model
        prediction = model.predict(image)
        
        # Get the confidence score (assuming binary classification)
        confidence = prediction[0][0]  # Model's output confidence for being defective
        defect = "Defective" if confidence >= 0.5 else "Non-Defective"  # Threshold at 0.5
        
        # Display the result and confidence
        st.write(f"Prediction: {defect}")
        st.write(f"Confidence: {confidence:.2f}")

        # Optional: Display a bar chart of the confidence
        confidence_values = [confidence, 1 - confidence]  # Confidence for defective and non-defective
        labels = ['Defective', 'Non-Defective']
        
        # Plotting the bar chart
        plt.bar(labels, confidence_values, color=['red', 'green'])
        plt.ylim(0, 1)
        plt.ylabel('Confidence')
        st.pyplot(plt)

        # Display additional information based on confidence
        if confidence >= 0.5:
            st.write("The casting is predicted to be defective.")
        else:
            st.write("The casting is predicted to be non-defective.")
