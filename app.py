import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

model = load_model("paddy_disease_model.h5")

classes = [
    "bacterial_leaf_blight",
    "brown_spot",
    "leaf_scald",
    "healthy"
]
disease_info = {
    "brown_spot": {
        "description": "Brown Spot is a fungal disease that causes brown lesions on leaves.",
        "control": "Apply fungicides and maintain proper field drainage."
    },
    "leaf_scald": {
        "description": "Leaf Scald causes drying and discoloration of leaf margins.",
        "control": "Use resistant varieties and balanced fertilization."
    },
    "bacterial_leaf_blight": {
        "description": "A bacterial disease causing yellowing and drying of leaves.",
        "control": "Use disease-free seeds and avoid excessive nitrogen."
    },
    "healthy": {
        "description": "The paddy leaf is healthy.",
        "control": "No control measures required."
    }
}

st.title("Paddy Leaf Disease Predictor")

st.write("""
This project predicts paddy leaf diseases using CNN.
Upload a leaf image for instant prediction.
""")

uploaded_file = st.file_uploader(
    "Upload Paddy Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=250)

    img = img.convert("L")
    img = img.resize((400, 400))

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 400, 400, 1)

    prediction = model.predict(img_array)
    idx = np.argmax(prediction)

    st.success(f"Prediction: {classes[idx]}")
    info = disease_info.get(classes[idx], {})

    st.write("### Description")
    st.write(info.get("description", "No information available"))

    st.write("### Control Measures")
    st.write(info.get("control", "No control measures available"))