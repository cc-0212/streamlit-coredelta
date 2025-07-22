import streamlit as st
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image, ImageDraw
import gdown
import os

# Google Drive file ID and local model path
GDRIVE_FILE_ID = '1XCxew13_vNiF-dczi39DErRyu1ftqhvS'  # Replace with your actual file ID
MODEL_PATH = 'fasterrcnn_model.pth'

# Download model weights from Google Drive
@st.cache_resource
def download_model_weights():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return MODEL_PATH

# Load Faster R-CNN model
@st.cache_resource
def load_model():
    path = download_model_weights()
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=3)  # Adjust num_classes
    # Explicitly set weights_only=False for PyTorch 2.6+
    state_dict = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Image transform
transform = T.Compose([
    T.ToTensor()
])

# Draw bounding boxes and labels
def draw_boxes(image, boxes, labels, scores, threshold=0.7):
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            draw.rectangle(box.tolist(), outline="red", width=3)
            draw.text((box[0], box[1]), f"{label}: {score:.2f}", fill="red")
    return image

# Streamlit UI
st.title("Fingerprint Detection using Faster R-CNN")

uploaded_file = st.file_uploader("Upload a fingerprint image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with st.spinner("Running detection..."):
        with torch.no_grad():
            outputs = model(img_tensor)[0]

    boxes = outputs['boxes']
    labels = outputs['labels']
    scores = outputs['scores']

    # Draw results on the image
    result_image = image.copy()
    result_image = draw_boxes(result_image, boxes, labels, scores, threshold=0.7)

    st.image(result_image, caption="Detection Results", use_column_width=True)
