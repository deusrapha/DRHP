import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io

# Set page config
st.set_page_config(
    page_title="Herbal Plant Detection",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Herbal Plant Detection and Recognition (DRHP)")
st.write("Upload an image or use your camera to detect herbal plants.")

# Load the model
@st.cache_resource
def load_model():
    # Cache buster comment: load Colab Instance Segmentation model
    model = YOLO("best.pt")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Upload Image", "Camera Input"])

def process_and_display(image):
    # Perform inference with a 50% confidence threshold to filter out low-confidence mistakes
    results = model(image, conf=0.5)
    
    # Render the results on the image
    res_plotted = results[0].plot()
    
    # Convert from BGR to RGB using numpy slicing instead of cv2 (bypasses Streamlit Cloud import errors)
    res_plotted_rgb = res_plotted[:, :, ::-1]
    
    st.image(res_plotted_rgb, caption="Detection Results", use_column_width=True)
    
    # Display predictions details
    with st.expander("Show Prediction Details", expanded=True):
        boxes = results[0].boxes
        if len(boxes) == 0:
            st.write("No herbal plants detected in this image.")
        else:
            for box in boxes:
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                conf = box.conf[0].item()
                st.write(f"- **{class_name}**: {conf:.2%} confidence")

    st.markdown("---")
    st.subheader("👨‍🔬 Active Learning: Expert Feedback")
    st.write("Did the model make a mistake? Help it learn by providing the correct label!")
    
    # Get list of classes from the model
    class_names = list(model.names.values())
    
    # Dropdown for expert correction
    correct_label = st.selectbox("Select the true plant species:", ["-- Select True Label --"] + class_names)
    
    if correct_label != "-- Select True Label --":
        # Convert original numpy image to bytes for download
        img_pil = Image.fromarray(image)
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        
        # Create a safe filename using the true label
        safe_label = correct_label.replace(" ", "_").replace("/", "_")
        
        st.download_button(
            label=f"💾 Download Image as '{safe_label}'",
            data=byte_im,
            file_name=f"{safe_label}_expert_correction.jpg",
            mime="image/jpeg",
            help="Download this image with the correct label. You can then upload it to Roboflow to retrain and improve the model!"
        )

with tab1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file).convert('RGB')
        # Convert to numpy array
        image_np = np.array(image)
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("---")
        if st.button("Detect Plants", key="detect_upload"):
            with st.spinner("Processing..."):
                process_and_display(image_np)

with tab2:
    st.header("Camera Input")
    st.write("Allow camera access to take a picture and detect plants.")
    camera_image = st.camera_input("Take a picture")
    
    if camera_image is not None:
        # Read the image
        image = Image.open(camera_image).convert('RGB')
        # Convert to numpy array
        image_np = np.array(image)
        
        st.write("---")
        with st.spinner("Processing..."):
            process_and_display(image_np)

st.markdown("---")
st.markdown("Developed for the Detection and Recognition of Herbal Plants from Video Frames (DRHP) project.")
