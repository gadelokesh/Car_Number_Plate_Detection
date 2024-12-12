import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Title of the app
st.title("Number Plate Detection App 🚗")

# Sidebar information
st.sidebar.title("Settings")
upload_option = st.sidebar.selectbox("Choose Input Type:", ["Upload Image"])

# Load Haar Cascade for number plate detection
@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

plate_cascade = load_cascade()

# Function for detecting number plates
def detect_number_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in plates:
        # Draw rectangle around the detected number plate
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Crop number plate region (optional)
        plate_region = image[y:y + h, x:x + w]
        st.sidebar.image(plate_region, caption="Detected Plate Region", use_container_width=True)

    return image, plates

# Upload image option
if upload_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Perform detection
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Detecting number plates...")
        detected_image, plates = detect_number_plate(image_np)

        # Display result
        st.image(detected_image, caption="Detected Number Plates", use_container_width=True)
        st.write(f"Number of plates detected: {len(plates)}")

st.sidebar.write("Developed with ❤️ using Pytesseract and OpenCV")
