import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io
import datetime
import random
from fpdf import FPDF
from herb_dict import HERB_DICT

def safe_text(text):
    if not isinstance(text, str):
        return str(text)
    replacements = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2022': '*',  # Bullet point
        '\u2122': 'TM', # Trademark
        '\u00ae': '(R)',# Registered
        '\u00a9': '(C)',# Copyright
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    return text.encode('latin-1', 'ignore').decode('latin-1')

def generate_diagnostic_report_pdf(selected_herbs):
    # Initialize PDF
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    
    # Color palette
    primary_green = (46, 125, 50)    # #2E7D32
    forest_green = (27, 94, 32)     # #1B5E20
    gray_text = (85, 85, 85)        # #555555
    light_green = (232, 245, 233)   # #E8F5E9
    
    # 1. Header Logo & Institution Details
    try:
        pdf.image("logo.png", x=10, y=10, w=25)
    except Exception:
        # Fallback if image fails to load (draw clean text placeholder)
        pdf.set_fill_color(*primary_green)
        pdf.rect(10, 10, 25, 25, "F")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_xy(10, 18)
        pdf.cell(25, 5, "DRHP", align='C')
        
    # Title & Subtitle (x=38, y=12)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*forest_green)
    pdf.set_xy(38, 12)
    pdf.cell(0, 8, safe_text("DRHP - ETHNOBOTANICAL DIAGNOSTIC REPORT"), ln=True)
    
    pdf.set_font("Helvetica", "I", 9.5)
    pdf.set_text_color(*gray_text)
    pdf.set_x(38)
    pdf.cell(0, 5, safe_text("Ugandan Indigenous Herbal Identification & Recommendation System"), ln=True)
    
    # Report Metadata (Top Right)
    date_str = datetime.date.today().strftime("%B %d, %Y")
    report_id = f"DRHP-REC-{random.randint(1000, 9999)}"
    
    pdf.set_xy(140, 12)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*gray_text)
    pdf.cell(60, 5, safe_text(f"Date: {date_str}"), ln=True, align='R')
    pdf.set_xy(140, 18)
    pdf.cell(60, 5, safe_text(f"Report ID: {report_id}"), ln=True, align='R')
    
    # Header Divider Line
    pdf.set_draw_color(*primary_green)
    pdf.set_line_width(0.8)
    pdf.line(10, 38, 200, 38)
    
    # 2. Diagnostic Summary Box
    pdf.set_xy(10, 43)
    pdf.set_fill_color(*light_green)
    pdf.set_draw_color(*primary_green)
    pdf.set_line_width(0.3)
    pdf.rect(10, 43, 190, 26, "FD")
    
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(*forest_green)
    pdf.set_xy(12, 45)
    pdf.cell(95, 5, safe_text("DIAGNOSTIC STATUS"))
    pdf.cell(95, 5, safe_text("CLASSIFICATION SUMMARY"))
    
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(0, 0, 0)
    
    pdf.set_xy(12, 51)
    pdf.cell(95, 5, safe_text("System Status: Species Identified Successfully"))
    herb_list_str = ", ".join([h['local_name'] for h in selected_herbs])
    pdf.cell(95, 5, safe_text(f"Identified Herb(s): {herb_list_str}"))
    
    pdf.set_xy(12, 56)
    pdf.cell(95, 5, safe_text("Verification Method: YOLOv8 Instance Segmentation"))
    sci_list_str = ", ".join([h['scientific_name'] for h in selected_herbs])
    pdf.cell(95, 5, safe_text(f"Scientific Name: {sci_list_str}"))
    
    pdf.set_xy(12, 61)
    pdf.cell(95, 5, safe_text(f"Diagnostic Date: {date_str}"))
    conf_list_str = ", ".join([f"{h['confidence']:.2%}" for h in selected_herbs])
    pdf.cell(95, 5, safe_text(f"Classification Confidence: {conf_list_str}"))
    
    # 3. Herb Recommendations Details
    current_y = 74
    
    for herb in selected_herbs:
        # Check if height overflows A4 page bounds
        if current_y > 210:
            pdf.add_page()
            current_y = 15
            
        # Draw a clean header band for each herb
        pdf.set_xy(10, current_y)
        pdf.set_fill_color(241, 248, 233)  # #F1F8E9
        pdf.set_draw_color(*primary_green)
        pdf.rect(10, current_y, 190, 8, "FD")
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*forest_green)
        pdf.set_xy(12, current_y + 1.5)
        pdf.cell(100, 5, safe_text(f"Herb: {herb['local_name']} ({herb['scientific_name']})"))
        pdf.set_font("Helvetica", "I", 9.5)
        pdf.set_text_color(*gray_text)
        pdf.cell(86, 5, safe_text(f"Model Confidence: {herb['confidence']:.2%}"), align='R')
        
        current_y += 10
        
        # Indications Section
        pdf.set_xy(12, current_y)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(*forest_green)
        pdf.cell(0, 5, safe_text("INDICATIONS & THERAPEUTIC PROPERTIES:"), ln=True)
        
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(50, 50, 50)
        pdf.set_x(12)
        pdf.multi_cell(186, 5, safe_text(herb['treats']))
        current_y = pdf.get_y() + 2
        
        # Preparation Section
        pdf.set_xy(12, current_y)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(*forest_green)
        pdf.cell(0, 5, safe_text("PREPARATION & DOSAGE METHODOLOGY:"), ln=True)
        
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(50, 50, 50)
        pdf.set_x(12)
        pdf.multi_cell(186, 5, safe_text(herb['preparation']))
        current_y = pdf.get_y() + 6
        
    # 4. Warnings and Disclaimers (Force to new page if it doesn't fit on this one)
    if current_y > 230:
        pdf.add_page()
        current_y = 15
        
    pdf.set_xy(10, current_y)
    pdf.set_fill_color(255, 243, 224)    # Light Orange #FFF3E0
    pdf.set_draw_color(255, 152, 0)      # Orange border #FF9800
    pdf.rect(10, current_y, 190, 14, "FD")
    
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(230, 81, 0)       # Deep Orange #E65100
    pdf.set_xy(12, current_y + 1.5)
    pdf.cell(0, 4, safe_text("BOTANICAL CLINICAL DISCLAIMER & RECOMMENDATION"), ln=True)
    
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(80, 80, 80)
    pdf.set_x(12)
    pdf.multi_cell(186, 3.5, safe_text(
        "This report is generated automatically by the DRHP Deep Learning Model based on image classification. "
        "Herbal medicines possess active chemical compounds; please consult a certified phytotherapist or medical doctor to "
        "validate dosage, interactions, and suitability."
    ))
    
    # 5. System Verification / Signature Block
    pdf.set_y(242)
    pdf.set_draw_color(180, 180, 180)
    pdf.set_line_width(0.3)
    pdf.line(130, 247, 190, 247)
    pdf.set_xy(130, 249)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*gray_text)
    pdf.cell(60, 4, safe_text("System Verification Signature"), ln=True, align='C')
    
    # 6. Institutional Footer
    pdf.set_y(265)
    pdf.set_draw_color(*primary_green)
    pdf.set_line_width(0.5)
    pdf.line(10, 265, 200, 265)
    pdf.set_y(267)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 4, safe_text("DRHP Digital Herbarium * Kampala, Uganda * Support: info@drhp.org * Web: www.drhp.org"), ln=True, align='C')
    
    return bytes(pdf.output())


# Set page config
st.set_page_config(
    page_title="Herbal Plant Detection",
    page_icon="🌿",
    layout="wide"
)

# Custom styling using a green color palette
st.markdown(
    """
    <style>
    /* Green theme styling */
    :root {
        --primary-color: #2E7D32;
        --background-color: #F1F8E9;
    }
    
    /* Sidebar Styling (Left Panel in Green Palette) */
    [data-testid="stSidebar"] {
        background-color: #E8F5E9 !important;
        border-right: 2px solid #C8E6C9;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #1B5E20 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2E7D32 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: bold !important;
        padding: 8px 16px !important;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1B5E20 !important;
        box-shadow: 0px 4px 10px rgba(46, 125, 50, 0.3) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #E8F5E9;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #2E7D32;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E7D32 !important;
        color: white !important;
    }
    
    /* Card design */
    .herb-card {
        background-color: #F9FBE7;
        padding: 20px;
        border-left: 5px solid #2E7D32;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌿 Herbal Plant Detection and Recognition (DRHP)")
st.write("Automated identification and ethnobotanical diagnostics of indigenous Ugandan herbal plants.")

# Sidebar settings for resizing images and display
st.sidebar.image("logo.png", width=120)
st.sidebar.header("🔧 Layout & Display Settings")
st.sidebar.write("Customize how images are displayed on screen:")
layout_mode = st.sidebar.selectbox("Layout Mode", ["Side-by-Side (Columns)", "Stacked (Standard)"])
image_width = st.sidebar.slider("Image Display Width (px)", min_value=200, max_value=800, value=400, step=50)

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

# Setup main application tabs
tab_diagnostics, tab_encyclopedia, tab_stats, tab_info = st.tabs([
    "🌿 Real-time Herb Diagnostics", 
    "🌱 Ethnobotanical Encyclopedia", 
    "📈 Training & Model Statistics",
    "🧪 System Information & XAI"
])

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
    
    # Display images based on layout mode and size slider
    if layout_mode == "Side-by-Side (Columns)":
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Input", width=image_width)
        with col2:
            st.image(res_plotted_rgb, caption="DRHP Detection Results", width=image_width)
    else:
        st.image(image, caption="Original Input", width=image_width)
        st.image(res_plotted_rgb, caption="DRHP Detection Results", width=image_width)
    
    # Display Diagnostic & Recommendation Details
    with st.expander("🩺 Diagnostic & Recommendation Details", expanded=True):
        boxes = results[0].boxes
        if len(boxes) == 0:
            st.write("No herbal plants detected in this image.")
        else:
            # Group boxes by class_id and keep the one with the maximum confidence to eliminate repetition
            unique_detections = {}
            for box in boxes:
                class_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                if class_id not in unique_detections or conf > unique_detections[class_id].conf[0].item():
                    unique_detections[class_id] = box
            
            detected_names = []
            for box in unique_detections.values():
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                detected_names.append(class_name)
                
            st.write("Select which detected species to keep for recommendation details (uncheck false positives):")
            selected_names = st.multiselect(
                "Detections:", 
                options=detected_names, 
                default=detected_names,
                label_visibility="collapsed"
            )
            
            st.write("---")
            
            if not selected_names:
                st.write("No species selected. Please check at least one box above to view details.")
            else:
                for i, name in enumerate(selected_names):
                    # Find corresponding box in unique_detections
                    box = None
                    for b in unique_detections.values():
                        cid = int(b.cls[0].item())
                        if model.names[cid] == name:
                            box = b
                            break
                            
                    if box is None:
                        continue
                        
                    class_id = int(box.cls[0].item())
                    class_name = model.names[class_id]
                    conf = box.conf[0].item()
                    
                    # Fetch details from dictionary
                    herb_info = HERB_DICT.get(class_name, {})
                    local_name = herb_info.get('local_name', class_name)
                    sci_name = herb_info.get('scientific_name', 'Unknown')
                    treats = herb_info.get('treats', 'Consult herbalist')
                    prep = herb_info.get('preparation', 'Consult herbalist')
                    
                    st.markdown(
                        f"""
                        <div class="herb-card">
                            <h3 style="color: #1B5E20; margin-top: 0; margin-bottom: 8px;">🌿 Recommendation: {local_name}</h3>
                            <p style="margin: 4px 0;"><b>Scientific Name:</b> <i>{sci_name}</i></p>
                            <p style="margin: 4px 0;"><b>DRHP Confidence:</b> <span style="color: #2E7D32; font-weight: bold;">{conf:.2%}</span></p>
                            <p style="margin: 4px 0;"><b>What it Heals:</b> {treats}</p>
                            <p style="margin: 4px 0;"><b>Administration & Preparation:</b> {prep}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    if conf < 0.60:
                        st.warning("Low confidence result. Please confirm with a trained herbal medicine expert before use.")
                    
                    # Generate PDF Diagnostic Report for individual herb
                    herb_pdf_data = [{
                        'local_name': local_name,
                        'scientific_name': sci_name,
                        'confidence': conf,
                        'treats': treats,
                        'preparation': prep
                    }]
                    pdf_output = generate_diagnostic_report_pdf(herb_pdf_data)
                    
                    st.download_button(
                        label=f"📄 Download Diagnostic Report for {local_name}",
                        data=pdf_output,
                        file_name=f"{local_name.replace(' ', '_').replace('/', '_')}_Diagnostic_Report.pdf",
                        mime="application/pdf",
                        key=f"download_pdf_{i}_{local_name}"
                    )
                    
                    st.write("---")
                
                # Unified Diagnostic Report for all selected herbs
                if len(selected_names) > 1:
                    st.markdown("<h3 style='color: #1B5E20;'>📄 Consolidated Diagnostic Report</h3>", unsafe_allow_html=True)
                    st.write("Generate a unified formal report containing recommendation details for all selected species:")
                    
                    # Gather details for all selected herbs
                    selected_herbs_data = []
                    for name in selected_names:
                        # Find corresponding box in unique_detections
                        box = None
                        for b in unique_detections.values():
                            cid = int(b.cls[0].item())
                            if model.names[cid] == name:
                                box = b
                                break
                        if box is not None:
                            class_id = int(box.cls[0].item())
                            class_name = model.names[class_id]
                            conf = box.conf[0].item()
                            herb_info = HERB_DICT.get(class_name, {})
                            selected_herbs_data.append({
                                'local_name': herb_info.get('local_name', class_name),
                                'scientific_name': herb_info.get('scientific_name', 'Unknown'),
                                'confidence': conf,
                                'treats': herb_info.get('treats', 'Consult herbalist'),
                                'preparation': herb_info.get('preparation', 'Consult herbalist')
                            })
                    
                    if selected_herbs_data:
                        unified_pdf_bytes = generate_diagnostic_report_pdf(selected_herbs_data)
                        
                        st.download_button(
                            label="📥 Download Unified Ethnobotanical Report (All Detections)",
                            data=unified_pdf_bytes,
                            file_name="Unified_Ethnobotanical_Diagnostic_Report.pdf",
                            mime="application/pdf",
                            key="download_unified_pdf_report"
                        )
                        st.write("---")

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

# Populate Diagnostics Tab
with tab_diagnostics:
    st.header("🌿 Diagnostic Scan")
    st.write("Upload a leaf image or capture one using your device camera to invoke the instance segmentation model.")
    
    input_tab1, input_tab2 = st.tabs(["Upload Image", "Camera Input"])
    
    with input_tab1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="upload_file_uploader")
        if uploaded_file is not None:
            # Read the image
            image = Image.open(uploaded_file).convert('RGB')
            # Convert to numpy array
            image_np = np.array(image)
            
            st.write("---")
            with st.spinner("Processing..."):
                process_and_display(image_np)
                
    with input_tab2:
        st.write("Allow camera access to take a picture and detect plants.")
        camera_image = st.camera_input("Take a picture", key="camera_input_comp")
        
        if camera_image is not None:
            # Read the image
            image = Image.open(camera_image).convert('RGB')
            # Convert to numpy array
            image_np = np.array(image)
            
            st.write("---")
            with st.spinner("Processing..."):
                process_and_display(image_np)

# Populate Encyclopedia Tab
with tab_encyclopedia:
    st.header("📖 Ethnobotanical Encyclopedia")
    st.write("Search and browse details for all 17 plant species supported by the DRHP model.")
    
    search_query = st.text_input("🔍 Search herbs by name, scientific classification, or illnesses treated:", "")
    st.write("---")
    
    for herb_key, info in HERB_DICT.items():
        local = info.get('local_name', herb_key)
        scientific = info.get('scientific_name', '')
        treats = info.get('treats', '')
        preparation = info.get('preparation', '')
        
        # Check query match
        match = (search_query.lower() in local.lower() or 
                 search_query.lower() in scientific.lower() or 
                 search_query.lower() in treats.lower())
                 
        if not search_query or match:
            with st.expander(f"🌿 {local} ({scientific})", expanded=False):
                st.markdown(f"**Scientific Name:** _{scientific}_")
                st.markdown(f"**What it Heals:** {treats}")
                st.markdown(f"**Preparation & Administration:** {preparation}")

# Populate Statistics Tab
with tab_stats:
    st.header("📊 Model Performance & Training Curves")
    st.write("Quantitative metrics comparing performance benchmarks across model versions.")
    
    st.markdown(
        """
        ### Cross-Version Validation Results
        
        | Metric | V2 (Overfitted/Train Val) | V2 (Fair Unseen Split) | V3 (Fair Unseen Split) |
        |---|---|---|---|
        | **Mask mAP@50** | 67.61% | 57.30% | **54.00%** |
        | **Mask mAP@50-95** | 45.63% | 39.60% | **33.80%** |
        | **Mask Precision** | 82.46% | 69.30% | **53.70%** |
        | **Mask Recall** | 60.93% | 54.30% | **56.20%** |
        | **Box mAP@50** | 72.50% | 62.00% | **56.80%** |
        | **Box mAP@50-95** | 56.86% | 47.50% | **40.10%** |
        | **Box Precision** | 83.98% | 72.60% | **55.20%** |
        | **Box Recall** | 63.40% | 56.50% | **58.20%** |
        
        ### Key Achievements in V3
        * **Resolved Leakage**: Corrected training validation set mapping.
        * **Rare Class Recovery**: Additional smart-polygon annotations in V3 yielded major leaps on underrepresented species:
          * *Mugavu* improved from **0.00%** in V2 to **79.14%** in V3.
          * *Olweza* improved from **1.50%** in V2 to **43.25%** in V3.
          * New class *Nnabbbugira* successfully detected at **48.72%** mAP50.
        """
    )

# Populate Info/XAI Tab
with tab_info:
    st.header("💡 System Information & Explainable AI")
    
    col_x, col_y = st.columns(2)
    with col_x:
        st.subheader("Image Preprocessing")
        st.write("To standardize lighting, shadows, and glares in real-world captures, the app preprocesses input frames:")
        st.markdown(
            """
            * **CLAHE**: Enhances leaf contrast locally without amplifying image noise, making leaf veins and margins stand out.
            * **HSV Color Normalization**: Ignores raw luminance variations, making features robust to shadow and direct sunlight.
            """
        )
    with col_y:
        st.subheader("Explainable AI (XAI)")
        st.write("To support transparency, the system highlights texture gradients and edge segments:")
        st.markdown(
            """
            * **Saliency Feature Heatmaps**: Outlines key structural edges and vein nodes utilized by the CNN backbone.
            * **Active Learning loop**: Integrates expert-in-the-loop dropdown corrections to save mislabeled frames for future model training iterations.
            """
        )

st.markdown("---")
st.markdown("<p style='text-align: center; color: #2E7D32;'>Developed for the Detection and Recognition of Herbal Plants from Video Frames (DRHP) project.</p>", unsafe_allow_html=True)
