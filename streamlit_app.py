import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io
from fpdf import FPDF
from herb_dict import HERB_DICT
# Set page config
st.set_page_config(
    page_title="Herbal Plant Detection",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Herbal Plant Detection and Recognition (DRHP)")
st.write("Upload an image or use your camera to detect herbal plants.")

# Class-specific confidence thresholds to reduce false positives for rare/harder classes
CLASS_THRESHOLDS = {
    "Mugavu -Albizia coriaria": 0.40,
    "Olweza - Euphorbia-hirta": 0.35,
    "Nnabbbugira - Mentha aquatica": 0.35,
    "Omumbejja - Artemisia annua - Sweet Wormwood": 0.40,
    "Omwetango - Lambs-Quarters": 0.40,
    "Omwolola -Entada abyssinica": 0.40,
}
DEFAULT_THRESHOLD = 0.25

# Load the model
@st.cache_resource
def load_model():
    # Cache buster comment: load Colab Instance Segmentation model
    model = YOLO("best_v3.pt")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Upload Image", "Camera Input"])

def process_and_display(image):
    # Convert RGB numpy array to BGR format for YOLO model using numpy slicing
    image_bgr = image[:, :, ::-1]
    
    # Perform inference with standard YOLO default threshold (e.g. 0.15) to get all candidates, then filter
    results = model(image_bgr, conf=0.15)
    
    # Filter boxes based on class-specific thresholds
    keep_indices = []
    if results[0].boxes is not None:
        for idx, box in enumerate(results[0].boxes):
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            conf = box.conf[0].item()
            threshold = CLASS_THRESHOLDS.get(class_name, DEFAULT_THRESHOLD)
            if conf >= threshold:
                keep_indices.append(idx)
                
    # Keep only filtered detections
    if len(keep_indices) > 0:
        results[0] = results[0][keep_indices]
    else:
        # If no boxes passed the thresholds, create an empty results object
        if results[0].boxes is not None:
            results[0] = results[0][[]]
            
    # Render the results on the image
    res_plotted = results[0].plot()
    
    # Convert from BGR to RGB using numpy slicing instead of cv2 (bypasses Streamlit Cloud import errors)
    res_plotted_rgb = res_plotted[:, :, ::-1]
    
    st.image(res_plotted_rgb, caption="Detection Results", use_column_width=True)
    
    # Display Doctor's Prescription Form
    with st.expander("🩺 Doctor's Prescription Details", expanded=True):
        boxes = results[0].boxes
        if len(boxes) == 0:
            st.write("No herbal plants detected in this image.")
        else:
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                conf = box.conf[0].item()
                
                # Fetch details from dictionary
                herb_info = HERB_DICT.get(class_name, {})
                local_name = herb_info.get('local_name', class_name)
                sci_name = herb_info.get('scientific_name', 'Unknown')
                treats = herb_info.get('treats', 'Consult herbalist')
                prep = herb_info.get('preparation', 'Consult herbalist')
                
                st.markdown(f"### 🌿 Prescription: {local_name}")
                st.markdown(f"**Scientific Name:** _{sci_name}_")
                st.markdown(f"**DRHP Confidence:** {conf:.2%}")
                st.markdown(f"**What it Heals:** {treats}")
                st.markdown(f"**Administration & Preparation:** {prep}")
                
                if conf < 0.60:
                    st.warning("Low confidence result. Please confirm with a trained herbal medicine expert before use.")
                
                # Generate PDF Prescription
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, txt="DRHP - Herbal Medicine Prescription", ln=True, align='C')
                pdf.set_font("Arial", '', 12)
                pdf.ln(10)
                pdf.cell(200, 10, txt=f"Local Name: {local_name}", ln=True)
                pdf.cell(200, 10, txt=f"Scientific Name: {sci_name}", ln=True)
                pdf.cell(200, 10, txt=f"DRHP Confidence: {conf:.2%}", ln=True)
                pdf.ln(10)
                pdf.multi_cell(0, 10, txt=f"What it Treats:\n{treats}")
                pdf.ln(5)
                pdf.multi_cell(0, 10, txt=f"Preparation & Administration:\n{prep}")
                
                # Save PDF to bytes for Streamlit Download (fpdf2 returns a bytearray, Streamlit strictly needs bytes)
                pdf_output = bytes(pdf.output())
                
                st.download_button(
                    label=f"📄 Download PDF Prescription for {local_name}",
                    data=pdf_output,
                    file_name=f"{local_name.replace(' ', '_').replace('/', '_')}_Prescription.pdf",
                    mime="application/pdf",
                    key=f"download_pdf_{i}_{local_name}"
                )

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
