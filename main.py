import streamlit as st
import time
import av
import numpy as np
import torch
import os
import cv2
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
from OCR import extract_text

# Workaround for torch.classes error in some environments
torch.classes.__path__ = []

# Load YOLO model
model = YOLO("best.pt")

# Ensure directories exist
IMAGE_DIR = "image"
CROP_DIR = "croped"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

IMAGE_PATH = os.path.join(IMAGE_DIR, "frame.jpg")
stop_image = cv2.imread("/content/ChatGPT Image May 16, 2025, 04_13_56 AM.png")

# Sidebar menu
st.sidebar.markdown("## 🧭 Navigation")
menu_choice = st.sidebar.selectbox("Choose Action", ["Camera Mode", "Upload Image"])

def process_image(image_path, image):
    with st.spinner("🔍 Detecting ID..."):
        results = model(image_path)
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="Detected ID", use_container_width=True)

    # Save crops
    for i, box in enumerate(results[0].boxes.xyxy):
        class_id = int(results[0].boxes.cls[i])
        label = model.names[class_id]
        x1, y1, x2, y2 = map(int, box)
        cropped = image[y1:y2, x1:x2]
        crop_path = f"{label}.jpg"
        cv2.imwrite(os.path.join(CROP_DIR, crop_path), cropped)

    with st.spinner("🔍 Processing the image..."):
        try:
            data = extract_text(CROP_DIR)
            st.success("✅ Extraction complete!")
            df = pd.DataFrame([data])
            st.table(df)
        except Exception as e:
            st.error(f"❌ Failed to process image: {e}")
            raise e

    # Clean up
    try:
        os.remove(image_path)
        for f in os.listdir(CROP_DIR):
            os.remove(os.path.join(CROP_DIR, f))
        os.rmdir(CROP_DIR)
    except Exception:
        pass

# ----------------- CAMERA OPTION ----------------- #
if menu_choice == "Camera Mode":

    st.subheader("📸 Live Camera Capture")

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.stop = False
            self.last_frame = None

        def recv(self, frame):
            if self.stop:
                resized_stop_image = cv2.resize(stop_image, (640, 360))  # Resize for smaller overlay
                return av.VideoFrame.from_ndarray(resized_stop_image, format="bgr24")

            img = frame.to_ndarray(format="bgr24")
            results = model(img, conf=0.6)[0]
            boxes = results.boxes

            if len(boxes) == 8 and all(float(box.conf[0]) >= 0.6 for box in boxes):
                self.last_frame = img
                self.stop = True
                cv2.imwrite(IMAGE_PATH, img)

            return av.VideoFrame.from_ndarray(results.plot(), format="bgr24")

    ctx = webrtc_streamer(
        key="camera",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if os.path.exists(IMAGE_PATH):
        st.image(IMAGE_PATH, caption="Captured Frame", use_container_width=True)
        image = cv2.imread(IMAGE_PATH)
        process_image(IMAGE_PATH, image)

# ----------------- UPLOAD OPTION ----------------- #
elif menu_choice == "Upload Image":
    st.subheader("📤 Upload an Image")
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(IMAGE_PATH, image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        process_image(IMAGE_PATH, image)
