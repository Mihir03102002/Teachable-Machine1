import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Load the model
model = tf.keras.models.load_model('path_to_your_model_directory')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the input size of the model
    image = np.array(image) / 255.0  # Normalize the image
    return np.expand_dims(image, axis=0)

# Streamlit app layout
st.title("Anomaly Detection in Manufactured Products")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    class_index = np.argmax(prediction, axis=1)
    
    # Display result
    if class_index == 0:
        st.write("This product is Normal.")
    else:
        st.write("Anomaly detected in this product.")
