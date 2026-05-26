from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import streamlit as st
import torch

st.set_page_config(page_title="AI Image Caption Generator", page_icon="🖼️", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    * { font-family: 'Poppins', sans-serif; }
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #1a0a2e, #0a1a2e, #0f0c29);
        background-size: 400% 400%;
        animation: gradientBG 10s ease infinite;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero {
        text-align: center;
        padding: 40px 20px 20px 20px;
        animation: fadeInDown 1s ease;
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .hero h1 {
        font-size: 3em;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        background-size: 200% auto;
        animation: shimmer 3s infinite;
    }
    @keyframes shimmer {
        0% { background-position: 0% center; }
        50% { background-position: 100% center; }
        100% { background-position: 0% center; }
    }
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        color: white;
        padding: 5px 18px;
        border-radius: 50px;
        font-size: 0.85em;
        font-weight: 600;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(96,165,250,0.5); }
        70% { box-shadow: 0 0 0 12px rgba(96,165,250,0); }
        100% { box-shadow: 0 0 0 0 rgba(96,165,250,0); }
    }
    .card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 30px;
        margin: 15px 0;
        animation: fadeIn 0.8s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton button {
        background: linear-gradient(135deg, #60a5fa, #a78bfa) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 15px 50px !important;
        font-size: 1.1em !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        letter-spacing: 1px !important;
    }
    .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 35px rgba(96,165,250,0.4) !important;
    }
    .caption-box {
        background: rgba(96,165,250,0.1);
        border: 2px solid rgba(96,165,250,0.4);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        animation: slideUp 0.6s ease;
        margin-top: 20px;
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .caption-text {
        color: white;
        font-size: 1.3em;
        font-weight: 600;
        line-height: 1.6;
        margin-top: 10px;
    }
    .stats-row {
        display: flex;
        gap: 15px;
        margin: 20px 0;
    }
    .stat-box {
        flex: 1;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8em;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-label {
        color: #94a3b8;
        font-size: 0.8em;
        margin-top: 5px;
    }
    .footer {
        text-align: center;
        padding: 30px;
        color: #475569;
        font-size: 0.85em;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <div class="badge">🤖 Powered by BLIP AI</div>
    <h1>🖼️ AI Image Caption Generator</h1>
    <p style="color:#94a3b8; font-size:1.1em;">Upload any image and AI will describe it instantly!</p>
    <p style="color:#60a5fa; font-size:0.9em; font-weight:600;">
        Akshat Anand &nbsp;|&nbsp; Building intelligent systems that solve real problems
    </p>
</div>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

with st.spinner("⏳ Loading AI model... (first time takes 1-2 minutes)"):
    processor, model = load_model()

# Upload
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📤 Upload any image:",
    type=['jpg', 'jpeg', 'png', 'webp'],
    help="Upload any image and AI will generate a caption!"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_column_width=True, caption="Your uploaded image")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_btn = st.button("✨ Generate Caption")

    if generate_btn:
        with st.spinner("🤖 AI is analyzing your image..."):
            inputs = processor(image, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(output[0], skip_special_tokens=True)

            words = len(caption.split())
            chars = len(caption)

            st.markdown(f"""
            <div class="stats-row">
                <div class="stat-box">
                    <div class="stat-number">{words}</div>
                    <div class="stat-label">Words Generated</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{chars}</div>
                    <div class="stat-label">Characters</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">AI</div>
                    <div class="stat-label">BLIP Model</div>
                </div>
            </div>
            <div class="caption-box">
                <div style="color:#60a5fa; font-size:0.85em; font-weight:600;
                letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;">
                    🤖 AI Generated Caption
                </div>
                <div class="caption-text">"{caption}"</div>
            </div>
            """, unsafe_allow_html=True)

            st.text_area("📋 Copy Caption:", value=caption, height=80)

# Footer
st.markdown("""
<div class="footer">
    Built with 🐍 Python & 🤖 BLIP AI by Salesforce<br>
    <span style="color:#60a5fa; font-weight:600;">
        Akshat Anand — Building intelligent systems that solve real problems
    </span>
</div>
""", unsafe_allow_html=True)