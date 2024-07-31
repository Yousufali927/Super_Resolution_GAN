import streamlit as st
from PIL import Image
import numpy as np
from generator_model import generator

st.title("Super-Resolution GAN")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Enhancing...")

    # Preprocess the image
    img = np.array(image)
    img = img / 255.0  # normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # add batch dimension

    # Generate high-resolution image
    hr_image = generator.predict(img)[0]

    # Convert to image format
    hr_image = (hr_image * 255.0).clip(0, 255).astype(np.uint8)
    hr_image = Image.fromarray(hr_image)

    st.image(hr_image, caption='Enhanced Image', use_column_width=True)
