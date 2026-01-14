import streamlit as st
from PIL import Image
import os
import sys

# Add src to path to allow imports if running directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictor import FracturePredictor

# Page Config
st.set_page_config(
    page_title="Bone Fracture Detection AI",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("Configuration")
st.sidebar.markdown("Adjust detection sensitivity.")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Instructions:**\n"
    "1. Upload an X-ray image.\n"
    "2. View detected fractures and bounding boxes.\n"
    "3. Check the summary statistics."
)

# Main Content
st.title("🦴 Bone Fracture Detection AI")
st.markdown("### Intelligent X-ray Analysis System")

# Model Loading
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runs_dir = os.path.join(project_root, "bone_fracture_project")

model_path = None
best_pt_files = []

# Search for all 'best.pt' files in the runs directory
if os.path.exists(runs_dir):
    for root, dirs, files in os.walk(runs_dir):
        if "best.pt" in files:
            full_path = os.path.join(root, "best.pt")
            best_pt_files.append(full_path)

# Sort by modification time (newest first)
if best_pt_files:
    best_pt_files.sort(key=os.path.getmtime, reverse=True)
    model_path = best_pt_files[0]


if not model_path:
    st.error("⚠️ Model weights not found. Please run `python src/train.py` first.")
    st.stop()

@st.cache_resource
def get_predictor(path):
    return FracturePredictor(path)

try:
    predictor = get_predictor(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Original X-Ray", use_container_width=True)

    with st.spinner("Analyzing image for fractures..."):
        results = predictor.predict(image, conf_threshold, iou_threshold)
        
        # Visualize
        for result in results:
            im_array = result.plot()
            result_image = Image.fromarray(im_array[..., ::-1])
            
            with col2:
                st.image(result_image, caption="AI Analysis Result", use_container_width=True)
            
            # Statistics
            counts = predictor.get_fracture_counts(results)
            
            st.markdown("### 📊 Detection Summary")
            if counts:
                for fracture_type, count in counts.items():
                    st.success(f"**Detected:** {count}x {fracture_type}")
                
                with st.expander("View Detailed Confidence Scores"):
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = result.names[cls_id]
                        conf = float(box.conf[0])
                        st.write(f"- {cls_name}: `{conf:.2%}`")
            else:
                st.info("No fractures detected with current threshold settings.")