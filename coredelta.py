import streamlit as st
import torch
import torchvision
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import os
import urllib.request

# Path to save the model locally
model_path = "fasterrcnn_model.pth"

# Only download if not already present
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        url = "https://drive.google.com/uc?export=download&id=1XCxew13_vNiF-dczi39DErRyu1ftqhvS"
        urllib.request.urlretrieve(url, model_path)


# Set page config
st.set_page_config(page_title="Fingerprint Detection", layout="centered")

# Load model function
@st.cache_resource
def load_model(model_path, num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Prediction function with threshold = 0.7
def get_prediction(model, image, threshold=0.7):
    transform = T.ToTensor()
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']

    selected = [i for i, score in enumerate(scores) if score >= threshold]
    return boxes[selected], labels[selected], scores[selected]

# Draw boxes and labels
def draw_boxes(image, boxes, labels, scores, label_map):
    image_np = np.array(image).copy()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.int().tolist()
        label_name = label_map.get(label.item(), f"Class {label.item()}")
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label_name}: {score:.2f}"
        cv2.putText(image_np, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image_np

# ---- Streamlit Interface ----

st.title("Fingerprint Core/Delta Detector")

uploaded_file = st.file_uploader("Upload a fingerprint image", type=['jpg', 'png', 'jpeg'])

# Load model (adjust path and class count)
model_path = "fasterrcnn_model.pth"
num_classes = 3  # Background + Core + Delta (example)
label_map = {1: "Core", 2: "Delta"}  # Adjust based on your dataset

model = load_model(model_path, num_classes)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting..."):
        boxes, labels, scores = get_prediction(model, image, threshold=0.8)

        if len(boxes) == 0:
            st.warning("No fingerprint features detected above threshold 0.7.")
        else:
            result_img = draw_boxes(image, boxes, labels, scores, label_map)
            st.image(result_img, caption="Detection Result", use_column_width=True)
