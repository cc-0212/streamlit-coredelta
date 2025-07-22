import streamlit as st
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import gdown
import os

# Google Drive file ID and model path
GDRIVE_FILE_ID = '1XCxew13_vNiF-dczi39DErRyu1ftqhvS'
MODEL_PATH = 'fasterrcnn_model.pth'

# Download model from Google Drive
@st.cache_resource
def download_model_weights():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return MODEL_PATH

# Load the model
@st.cache_resource
def load_model():
    path = download_model_weights()
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=3)
    state_dict = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Image transform
transform = T.Compose([
    T.ToTensor()
])

# Draw bounding boxes and labels
from PIL import ImageDraw, ImageFont

def draw_boxes(image, boxes, labels, scores, threshold=0.7):
    draw = ImageDraw.Draw(image)
    try:
        # Load a larger TrueType font
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        # Fallback to default if arial not available
        font = ImageFont.load_default()

    highlight_green = (0, 255, 0)  # Bright highlight green

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            label_text = "core" if label == 1 else "delta" if label == 2 else str(label.item())
            box = box.tolist()

            # Draw rectangle
            draw.rectangle(box, outline=highlight_green, width=3)

            # Draw label text above the box
            text_position = (box[0], max(0, box[1] - 25))  # Raise text a bit more for larger font
            draw.text(text_position, f"{label_text} ({score:.2f})", fill=highlight_green, font=font)

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

    result_image = image.copy()
    result_image = draw_boxes(result_image, boxes, labels, scores, threshold=0.7)

    st.image(result_image, caption="Detection Results", use_column_width=True)
