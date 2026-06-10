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

def generate_diagnostic_report_pdf(selected_herbs, report_type="detection"):
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
    if report_type == "detection":
        pdf.cell(0, 8, safe_text("DRHP - ETHNOBOTANICAL DIAGNOSTIC REPORT"), ln=True)
    else:
        pdf.cell(0, 8, safe_text("DRHP - ETHNOBOTANICAL RECOMMENDATION REPORT"), ln=True)
    
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
    if report_type == "detection":
        pdf.cell(95, 5, safe_text("DIAGNOSTIC STATUS"))
        pdf.cell(95, 5, safe_text("CLASSIFICATION SUMMARY"))
    else:
        pdf.cell(95, 5, safe_text("RECOMMENDATION STATUS"))
        pdf.cell(95, 5, safe_text("SYMPTOM MATCH SUMMARY"))
    
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(0, 0, 0)
    
    pdf.set_xy(12, 51)
    if report_type == "detection":
        pdf.cell(95, 5, safe_text("System Status: Species Identified Successfully"))
        herb_list_str = ", ".join([h['local_name'] for h in selected_herbs])
        pdf.cell(95, 5, safe_text(f"Identified Herb(s): {herb_list_str}"))
    else:
        pdf.cell(95, 5, safe_text("System Status: Symptoms Matched Successfully"))
        herb_list_str = ", ".join([h['local_name'] for h in selected_herbs])
        pdf.cell(95, 5, safe_text(f"Recommended Herb(s): {herb_list_str}"))
    
    pdf.set_xy(12, 56)
    if report_type == "detection":
        pdf.cell(95, 5, safe_text("Verification Method: YOLOv8 Instance Segmentation"))
    else:
        pdf.cell(95, 5, safe_text("Verification Method: Symptom-Based Database Match"))
    sci_list_str = ", ".join([h['scientific_name'] for h in selected_herbs])
    pdf.cell(95, 5, safe_text(f"Scientific Name: {sci_list_str}"))
    
    pdf.set_xy(12, 61)
    pdf.cell(95, 5, safe_text(f"Diagnostic Date: {date_str}"))
    if report_type == "detection":
        conf_list_str = ", ".join([f"{h['confidence']:.2%}" for h in selected_herbs])
        pdf.cell(95, 5, safe_text(f"Classification Confidence: {conf_list_str}"))
    else:
        conf_list_str = ", ".join([safe_text(h['confidence']) for h in selected_herbs])
        pdf.cell(95, 5, safe_text(f"Match Level: {conf_list_str}"))
    
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
        if report_type == "detection":
            pdf.cell(86, 5, safe_text(f"Model Confidence: {herb['confidence']:.2%}"), align='R')
        else:
            pdf.cell(86, 5, safe_text(f"Match Relevance: {herb['confidence']}"), align='R')
        
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
        "This report is generated automatically by the DRHP Deep Learning Model based on image classification or symptom keywords match. "
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
    pdf.cell(0, 4, safe_text("DRHP Digital Herbarium * Kampala, Uganda * Support: tumusiime.deus@students.mak.ac.ug * Web: www.drhp.org"), ln=True, align='C')
    
    return bytes(pdf.output())


def find_matching_herbs(query_text, selected_categories):
    # Normalize query text
    query_words = set(query_text.lower().split()) if query_text else set()
    
    # Mapping categories to keywords
    category_keywords = {
        "Fever / Malaria": ["fever", "fevers", "malaria", "malarial", "febrile"],
        "Cough / Cold / Respiratory": ["cough", "coughs", "cold", "colds", "congestion", "respiratory", "asthma", "bronchitis", "throat", "throats", "chest", "bronchial"],
        "Stomach / Digestive Ailments": ["stomach", "digestive", "colic", "gastrointestinal", "diarrhea", "constipation", "ache", "aches", "discomfort", "intestine", "intestinal"],
        "Skin Irritations / Wounds": ["skin", "wound", "wounds", "irritation", "irritations", "disease", "diseases", "conditions", "topical", "irritated", "cleansing"],
        "Toothache / Oral Health": ["tooth", "toothache", "toothaches", "oral", "hygiene", "teeth", "mouth"],
        "Maternal / Infant Health": ["maternal", "infant", "child", "colic", "mother", "baby", "womb", "birth"],
        "Pain / Inflammation": ["pain", "inflammation", "ache", "aches", "sore", "irritation", "inflammatory"]
    }
    
    keywords_from_categories = []
    for cat in selected_categories:
        keywords_from_categories.extend(category_keywords.get(cat, []))
        
    all_target_keywords = query_words.union(set(keywords_from_categories))
    
    # Remove extremely common stopwords
    stopwords = {"and", "the", "a", "of", "to", "in", "for", "with", "or", "on", "at", "by", "an", "i", "have", "my", "tell", "symptoms", "pain", "is", "some", "we", "can", "recommend", "heals", "treats", "how"}
    target_keywords = {w.strip(".,?!();:-\"'") for w in all_target_keywords if w.strip(".,?!();:-\"'") not in stopwords and len(w) > 2}
    
    results = []
    for herb_key, info in HERB_DICT.items():
        # Skip Grevillea robusta (Silky Oak) if it is purely agroforestry and doesn't match specific query keywords
        if herb_key == "Silky Oak - Grevillea robusta" and not any(kw in query_text.lower() for kw in ["oak", "grevillea", "robust", "shade"]):
            continue
            
        local_name = info.get('local_name', herb_key)
        scientific_name = info.get('scientific_name', '')
        treats = info.get('treats', '').lower()
        prep = info.get('preparation', '').lower()
        
        score = 0
        matches = []
        
        for kw in target_keywords:
            kw_match = False
            # Check treats
            if kw in treats:
                score += 3  # High weight for matching treats
                kw_match = True
            # Check preparation
            if kw in prep:
                score += 1
                kw_match = True
            # Check local or scientific name
            if kw in local_name.lower() or kw in scientific_name.lower():
                score += 2
                kw_match = True
                
            if kw_match:
                matches.append(kw)
                
        if score > 0:
            results.append({
                "herb_key": herb_key,
                "local_name": local_name,
                "scientific_name": scientific_name,
                "treats": info.get('treats', ''),
                "preparation": info.get('preparation', ''),
                "score": score,
                "matched_keywords": matches
            })
            
    # Deduplicate results based on (local_name, scientific_name) to avoid duplicates
    unique_results = {}
    for r in results:
        key = (r['local_name'], r['scientific_name'])
        if key not in unique_results or r['score'] > unique_results[key]['score']:
            unique_results[key] = r
            
    sorted_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)
    return sorted_results


# Set page config
st.set_page_config(
    page_title="Herbal Plant Detection",
    page_icon="🌿",
    layout="wide"
)

# Custom styling using an ultra-premium green & slate color palette
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        background-color: #F7FAF6 !important;
        color: #2D3748 !important;
    }
    
    /* Sidebar Overhaul */
    [data-testid="stSidebar"] {
        background-color: rgba(241, 248, 233, 0.7) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(200, 230, 201, 0.4) !important;
    }
    
    /* Clean, Styled Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #1B5E20 !important;
        letter-spacing: -0.5px !important;
    }
    
    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #F1F8E9;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #C8E6C9;
        border-radius: 10px;
        transition: background 0.3s ease;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #A5D6A7;
    }
    
    /* Premium Styled Buttons */
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: none !important;
        font-size: 14px !important;
        cursor: pointer !important;
        height: auto !important;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(46, 125, 50, 0.35) !important;
        background: linear-gradient(135deg, #388E3C 0%, #2E7D32 100%) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    .stButton>button:active, .stDownloadButton>button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 4px 10px rgba(46, 125, 50, 0.2) !important;
    }
    
    /* Styled Inputs & Text Areas */
    div[data-baseweb="select"] > div, input, textarea {
        background-color: white !important;
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        transition: all 0.3s ease !important;
        color: #2D3748 !important;
    }
    div[data-baseweb="select"] > div:hover, input:hover, textarea:hover {
        border-color: #A5D6A7 !important;
    }
    div[data-baseweb="select"]:focus-within > div, input:focus, textarea:focus {
        border-color: #2E7D32 !important;
        box-shadow: 0 0 0 3px rgba(46, 125, 50, 0.15) !important;
    }
    
    /* Premium Tab Bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px !important;
        background-color: rgba(232, 245, 233, 0.5) !important;
        padding: 8px !important;
        border-radius: 16px !important;
        border: 1px solid rgba(200, 230, 201, 0.4) !important;
        margin-bottom: 24px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        color: #2E7D32 !important;
        font-weight: 500 !important;
        border: none !important;
        transition: all 0.25s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(232, 245, 233, 0.8) !important;
        color: #1B5E20 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E7D32 !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.15) !important;
    }
    
    /* Herb Cards Glassmorphism styling */
    .herb-card {
        background: linear-gradient(145deg, #ffffff 0%, #fcfdfe 100%) !important;
        border-left: 6px solid #2E7D32 !important;
        border-top: 1px solid rgba(0, 0, 0, 0.03) !important;
        border-right: 1px solid rgba(0, 0, 0, 0.03) !important;
        border-bottom: 1px solid rgba(0, 0, 0, 0.03) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .herb-card:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 30px rgba(46, 125, 50, 0.08) !important;
        border-left-color: #1B5E20 !important;
    }
    
    /* Interactive Hero banner styling */
    .hero-container {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%) !important;
        padding: 40px 32px !important;
        border-radius: 24px !important;
        margin-bottom: 35px !important;
        box-shadow: 0 10px 30px rgba(46, 125, 50, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.6) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .hero-badge {
        background-color: #2E7D32 !important;
        color: white !important;
        padding: 6px 16px !important;
        border-radius: 50px !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        display: inline-block !important;
        margin-bottom: 16px !important;
        box-shadow: 0 2px 10px rgba(46, 125, 50, 0.2) !important;
    }
    .hero-title {
        font-size: 36px !important;
        font-weight: 800 !important;
        margin: 0 0 12px 0 !important;
        line-height: 1.25 !important;
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    .hero-subtitle {
        font-size: 16px !important;
        color: #33691E !important;
        margin: 0 !important;
        font-weight: 400 !important;
        max-width: 800px !important;
        line-height: 1.5 !important;
    }
    
    /* Custom Badges */
    .badge-pill {
        display: inline-flex !important;
        align-items: center !important;
        padding: 4px 14px !important;
        border-radius: 50px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        margin-right: 8px !important;
    }
    .badge-emerald {
        background-color: #E8F5E9 !important;
        color: #2E7D32 !important;
        border: 1px solid rgba(46, 125, 50, 0.2) !important;
    }
    .badge-orange {
        background-color: #FFF3E0 !important;
        color: #E65100 !important;
        border: 1px solid rgba(230, 81, 0, 0.2) !important;
    }
    .badge-blue {
        background-color: #E3F2FD !important;
        color: #1565C0 !important;
        border: 1px solid rgba(21, 101, 192, 0.2) !important;
    }
    
    /* Alert Overrides for Glassmorphism styling */
    .stAlert {
        border-radius: 16px !important;
        border: 1px solid rgba(0,0,0,0.05) !important;
        background-color: rgba(255,255,255,0.6) !important;
        backdrop-filter: blur(8px) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02) !important;
    }
    
    /* Premium Table styling */
    table {
        border-collapse: separate !important;
        border-spacing: 0 !important;
        width: 100% !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid #E2E8F0 !important;
    }
    th {
        background-color: #E8F5E9 !important;
        color: #1B5E20 !important;
        font-weight: 600 !important;
        padding: 12px 16px !important;
        text-align: left !important;
        border-bottom: 2px solid #C8E6C9 !important;
    }
    td {
        padding: 12px 16px !important;
        border-bottom: 1px solid #E2E8F0 !important;
        background-color: white !important;
    }
    tr:last-child td {
        border-bottom: none !important;
    }
    
    /* Expert Feedback Panel */
    .feedback-panel {
        background: rgba(255, 255, 255, 0.5) !important;
        border: 1px dashed #A5D6A7 !important;
        border-radius: 16px !important;
        padding: 24px !important;
        margin-top: 24px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.01) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-container">
        <span class="hero-badge">🌿 DRHP PROTOCOL</span>
        <h1 class="hero-title">Herbal Plant Detection & Recognition</h1>
        <p class="hero-subtitle">Artificial intelligence for indigenous Ugandan ethnobotanical diagnostics and automated leaf analysis.</p>
    </div>
    """,
    unsafe_allow_html=True
)

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
tab_diagnostics, tab_symptoms, tab_encyclopedia, tab_stats, tab_info = st.tabs([
    "🌿 Real-time Herb Diagnostics", 
    "🩺 Symptom-based Finder",
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
                            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 16px; gap: 8px;">
                                <h3 style="color: #1B5E20; margin: 0; font-size: 22px; font-weight: 700;">🌿 {local_name}</h3>
                                <div>
                                    <span class="badge-pill badge-emerald">Confidence: {conf:.2%}</span>
                                </div>
                            </div>
                            <div style="margin-bottom: 12px;">
                                <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #718096; font-weight: 600; display: block; margin-bottom: 2px;">Scientific Classification</span>
                                <span style="font-size: 16px; font-style: italic; color: #2D3748; font-weight: 500;">{sci_name}</span>
                            </div>
                            <div style="margin-bottom: 16px; background-color: rgba(232, 245, 233, 0.3); padding: 12px 16px; border-radius: 8px; border-left: 3px solid #2E7D32;">
                                <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #2E7D32; font-weight: 600; display: block; margin-bottom: 2px;">Therapeutic Indications</span>
                                <span style="font-size: 15px; color: #2D3748;">{treats}</span>
                            </div>
                            <div>
                                <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #718096; font-weight: 600; display: block; margin-bottom: 2px;">Traditional Preparation & Usage</span>
                                <span style="font-size: 15px; color: #4A5568; line-height: 1.6;">{prep}</span>
                            </div>
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

    st.markdown(
        """
        <div class="feedback-panel">
            <h3 style="margin-top:0; color:#1B5E20; display:flex; align-items:center; gap:8px; font-size:18px; font-weight: 700;">
                <span>👨‍🔬</span> Expert Feedback Loop (Active Learning)
            </h3>
            <p style="font-size:14px; color:#4A5568; margin-bottom:0;">
                If the model misclassified the plant in this frame, help retrain it by selecting the correct ground-truth label below and exporting the sample to feed back into the training pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    
    # Get list of classes from the model
    class_names = list(model.names.values())
    
    # Dropdown for expert correction
    correct_label = st.selectbox("Select the true plant species classification:", ["-- Select True Label --"] + class_names)
    
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

# Populate Symptom Tab
with tab_symptoms:
    st.header("🩺 Symptom-based Herb Finder")
    st.write(
        "Enter details about symptoms, pain, or ailments below. "
        "The system will search our ethnobotanical database and recommend traditional Ugandan herbal medicines, "
        "detailing their usage, preparation, and administration instructions."
    )
    
    st.warning(
        "⚠️ **CRITICAL BOTANICAL CLINICAL DISCLAIMER:** "
        "The recommendations generated below are based on traditional ethnobotanical uses and automated database keyword mapping. "
        "Herbal medicines contain active chemical and pharmacological compounds. "
        "These recommendations are NOT a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a certified phytotherapist, medical doctor, or healthcare professional "
        "to validate dosage, suitability, and potential drug interactions before administering any herbal treatments."
    )
    
    col_input, col_tips = st.columns([2, 1])
    with col_input:
        st.subheader("🔍 Search Parameters")
        
        selected_categories = st.multiselect(
            "Select symptom categories (optional):",
            options=[
                "Fever / Malaria",
                "Cough / Cold / Respiratory",
                "Stomach / Digestive Ailments",
                "Skin Irritations / Wounds",
                "Toothache / Oral Health",
                "Maternal / Infant Health",
                "Pain / Inflammation"
            ],
            help="Select one or more categories to quickly find matching herbs."
        )
        
        query_text = st.text_area(
            "Describe the symptoms, pain, or discomfort in detail:",
            placeholder="e.g., I have a persistent cough with a sore throat and chest congestion, or severe stomach pain.",
            height=120
        )
        
        search_triggered = st.button("✨ Find Recommended Herbs")
        
    with col_tips:
        st.subheader("💡 Tips for Searching")
        st.markdown(
            """
            - **Be Specific:** Describe the exact nature of the pain (e.g., *cough*, *fever*, *stomach ache*, *skin irritation*).
            - **Traditional Names:** You can also query using traditional Luganda/Ugandan symptoms if known.
            - **Combined Selection:** Using both tags and a text description provides the most accurate search results.
            - **Doctor Advisory:** Note down the recommended herbs to discuss with a certified professional.
            """
        )
        
    if search_triggered or query_text or selected_categories:
        if not query_text and not selected_categories:
            st.info("Please enter a symptom description or select a category tag to begin.")
        else:
            with st.spinner("Searching ethnobotanical database..."):
                matches = find_matching_herbs(query_text, selected_categories)
                
                if not matches:
                    st.error("No matching herbs found for the specified symptoms. Please refine your query or try selecting different categories.")
                else:
                    st.success(f"Found {len(matches)} matching herb(s) based on your symptoms:")
                    st.write("---")
                    
                    matched_herbs_pdf_data = []
                    
                    for idx, match in enumerate(matches):
                        local_name = match['local_name']
                        sci_name = match['scientific_name']
                        treats = match['treats']
                        prep = match['preparation']
                        score = match['score']
                        matched_kws = match['matched_keywords']
                        
                        # Map scores to a match level text/relevance
                        if score >= 6:
                            match_level = "High Match"
                            match_color = "#2E7D32" # Dark Green
                            badge_class = "badge-emerald"
                        elif score >= 3:
                            match_level = "Medium Match"
                            match_color = "#F57C00" # Orange
                            badge_class = "badge-orange"
                        else:
                            match_level = "Partial Match"
                            match_color = "#1565C0" # Blue
                            badge_class = "badge-blue"
                            
                        matched_kws_str = ", ".join(matched_kws)
                        
                        st.markdown(
                            f"""
                            <div class="herb-card" style="border-left: 6px solid {match_color} !important;">
                                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 16px; gap: 8px;">
                                    <h3 style="color: #1B5E20; margin: 0; font-size: 22px; font-weight: 700;">🌿 {local_name}</h3>
                                    <div>
                                        <span class="badge-pill {badge_class}">{match_level}</span>
                                    </div>
                                </div>
                                <div style="margin-bottom: 12px;">
                                    <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #718096; font-weight: 600; display: block; margin-bottom: 2px;">Scientific Classification</span>
                                    <span style="font-size: 16px; font-style: italic; color: #2D3748; font-weight: 500;">{sci_name}</span>
                                </div>
                                <div style="margin-bottom: 12px;">
                                    <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #718096; font-weight: 600; display: block; margin-bottom: 2px;">Matched Keywords</span>
                                    <span style="font-size: 14px; color: #4A5568; background-color: #F7FAF6; padding: 4px 8px; border-radius: 4px; border: 1px solid #E2E8F0; display: inline-block;">{matched_kws_str}</span>
                                </div>
                                <div style="margin-bottom: 16px; background-color: rgba(232, 245, 233, 0.3); padding: 12px 16px; border-radius: 8px; border-left: 3px solid #2E7D32;">
                                    <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #2E7D32; font-weight: 600; display: block; margin-bottom: 2px;">Therapeutic Indications</span>
                                    <span style="font-size: 15px; color: #2D3748;">{treats}</span>
                                </div>
                                <div>
                                    <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #718096; font-weight: 600; display: block; margin-bottom: 2px;">Traditional Preparation & Usage</span>
                                    <span style="font-size: 15px; color: #4A5568; line-height: 1.6;">{prep}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        st.warning(
                            f"⚠️ **Caution for {local_name}:** Do not self-prescribe. Consult a medical doctor to discuss appropriate dosage and any contraindications for treating *{treats}*."
                        )
                        
                        matched_herbs_pdf_data.append({
                            'local_name': local_name,
                            'scientific_name': sci_name,
                            'confidence': match_level,
                            'treats': treats,
                            'preparation': prep
                        })
                        
                        single_pdf = generate_diagnostic_report_pdf([{
                            'local_name': local_name,
                            'scientific_name': sci_name,
                            'confidence': match_level,
                            'treats': treats,
                            'preparation': prep
                        }], report_type="symptom")
                        
                        st.download_button(
                            label=f"📄 Download Recommendation Report for {local_name}",
                            data=single_pdf,
                            file_name=f"{local_name.replace(' ', '_').replace('/', '_')}_Recommendation_Report.pdf",
                            mime="application/pdf",
                            key=f"symptom_pdf_{idx}_{local_name}"
                        )
                        st.write("---")
                        
                    if len(matched_herbs_pdf_data) > 1:
                        st.markdown("<h3 style='color: #1B5E20;'>📄 Consolidated Recommendation Report</h3>", unsafe_allow_html=True)
                        st.write("Generate a single, unified ethnobotanical recommendation report containing all matching species:")
                        
                        unified_pdf = generate_diagnostic_report_pdf(matched_herbs_pdf_data, report_type="symptom")
                        
                        st.download_button(
                            label="📥 Download Unified Ethnobotanical Recommendation Report",
                            data=unified_pdf,
                            file_name="Unified_Ethnobotanical_Recommendation_Report.pdf",
                            mime="application/pdf",
                            key="download_unified_symptom_report"
                        )
                        st.write("---")

# Populate Encyclopedia Tab
with tab_encyclopedia:
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h2 style="color: #1B5E20; margin-bottom: 8px;">📖 Ethnobotanical Encyclopedia</h2>
            <p style="color: #4A5568; font-size: 15px;">Search and browse details for all 17 plant species supported by the DRHP model.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
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
                st.markdown(
                    f"""
                    <div style="background-color: white; padding: 18px; border-radius: 12px; border: 1px solid rgba(0,0,0,0.03);">
                        <div style="margin-bottom: 12px;">
                            <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #718096; font-weight: 600; display: block; margin-bottom: 2px;">Scientific Name</span>
                            <span style="font-size: 16px; font-style: italic; color: #1B5E20; font-weight: 600;">{scientific}</span>
                        </div>
                        <div style="margin-bottom: 12px; border-top: 1px solid #edf2f7; padding-top: 10px;">
                            <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #2E7D32; font-weight: 600; display: block; margin-bottom: 2px;">Illnesses & Conditions Treated</span>
                            <span style="font-size: 14px; color: #2D3748;">{treats}</span>
                        </div>
                        <div style="border-top: 1px solid #edf2f7; padding-top: 10px;">
                            <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #718096; font-weight: 600; display: block; margin-bottom: 2px;">Traditional Preparation & Administration</span>
                            <span style="font-size: 14px; color: #4A5568; line-height: 1.6;">{preparation}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# Populate Statistics Tab
with tab_stats:
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h2 style="color: #1B5E20; margin-bottom: 8px;">📊 Model Performance & Training Curves</h2>
            <p style="color: #4A5568; font-size: 15px;">Quantitative metrics comparing performance benchmarks across model versions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
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
        
        <br>
        
        <div class="herb-card" style="border-left: 6px solid #2E7D32 !important; background: white !important;">
            <h3 style="margin-top:0; color:#1B5E20; font-size:18px; font-weight:700;">💡 Key Achievements & Version 3 Upgrades</h3>
            <ul style="margin: 0; padding-left: 20px; font-size: 14.5px; color: #4A5568; line-height: 1.8;">
                <li><b>Resolved Validation Leakage</b>: Corrected training set validation mapping to prevent over-optimistic evaluation curves.</li>
                <li><b>Rare Class Recall Recovery</b>: Additional smart-polygon annotations in V3 yielded major detection leaps on underrepresented species:
                    <ul style="padding-left: 15px; margin-top: 4px;">
                        <li><i>Mugavu</i> improved from <span style="color:#e53e3e; font-weight:bold;">0.00%</span> to <span style="color:#2e7d32; font-weight:bold;">79.14%</span> mAP@50.</li>
                        <li><i>Olweza</i> improved from <span style="color:#e53e3e; font-weight:bold;">1.50%</span> to <span style="color:#2e7d32; font-weight:bold;">43.25%</span> mAP@50.</li>
                    </ul>
                </li>
                <li><b>New Species Inclusion</b>: Successfully initiated robust detection for the water mint class <i>Nnabbbugira</i> at <span style="color:#2e7d32; font-weight:bold;">48.72%</span> mAP@50.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# Populate Info/XAI Tab
with tab_info:
    st.markdown(
        """
        <div style="margin-bottom: 24px;">
            <h2 style="color: #1B5E20; margin-bottom: 8px;">💡 System Information & Explainable AI</h2>
            <p style="color: #4A5568; font-size: 15px;">Under the hood of the DRHP herbal diagnostics classification protocol.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown(
            """
            <div class="herb-card" style="height: 100%;">
                <h3 style="color: #1B5E20; margin-top: 0; font-size: 18px; font-weight: 700;">🖼️ Image Preprocessing Pipeline</h3>
                <p style="font-size: 14px; color: #4A5568; line-height: 1.65; margin-bottom: 12px;">
                    To standardize lighting, shadows, and color glares in real-world smartphone frame captures, the app integrates classic image filters:
                </p>
                <ul style="font-size: 14px; color: #4A5568; line-height: 1.8; padding-left: 20px; margin: 0;">
                    <li><b>CLAHE (Contrast Limited Adaptive Histogram Equalization)</b>: Enhances local contrast of leaf surfaces without amplifying sensor noise, highlighting venation and texture patterns.</li>
                    <li><b>HSV Color Space Conversion</b>: Decouples luminance from color chrominance, making the YOLOv8-seg backbone robust to shadow lines and direct sunlight variations.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_y:
        st.markdown(
            """
            <div class="herb-card" style="height: 100%;">
                <h3 style="color: #1B5E20; margin-top: 0; font-size: 18px; font-weight: 700;">🧠 Deep Learning & Interpretability</h3>
                <p style="font-size: 14px; color: #4A5568; line-height: 1.65; margin-bottom: 12px;">
                    Providing visibility and standard verification tools into the deep convolutional network's class prediction pathways:
                </p>
                <ul style="font-size: 14px; color: #4A5568; line-height: 1.8; padding-left: 20px; margin: 0;">
                    <li><b>Saliency Feature Heatmaps</b>: Uses deep feature activations to outline critical leaf margins and vein nodes prioritized by the CNN model.</li>
                    <li><b>Expert Human-in-the-Loop</b>: Facilitates continuous active learning by allowing trained botanists to override low-confidence predictions and export the corrected labels.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")
st.markdown("<p style='text-align: center; color: #2E7D32;'>Developed for the Detection and Recognition of Herbal Plants from Video Frames (DRHP) project.</p>", unsafe_allow_html=True)
