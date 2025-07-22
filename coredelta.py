import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import io
import requests
import os
import urllib.request

# Model download URL (Google Drive direct link format)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1XCxew13_vNiF-dczi39DErRyu1ftqhvS"
MODEL_PATH = "fasterrcnn_model.pth"

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Load the model with weights_only=False
@st.cache_resource
def load_model():
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
    model.eval()
    return model

model = load_model()

# Define transform
transform = T.Compose([
    T.ToTensor()
])

# Streamlit UI
st.title("Fingerprint Detection with Faster R-CNN")
st.write("Upload an image to detect fingerprints.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

CONFIDENCE_THRESHOLD = 0.8

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)[0]

    draw = ImageDraw.Draw(image)
    for box, score in zip(output["boxes"], output["scores"]):
        if score >= CONFIDENCE_THRESHOLD:
            draw.rectangle(box.tolist(), outline="red", width=2)
            draw.text((box[0], box[1]), f"{score:.2f}", fill="red")

    st.image(image, caption="Detected fingerprints")
