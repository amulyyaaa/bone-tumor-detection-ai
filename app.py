"""
Bone Tumor Detection — Streamlit Web App
Upload an X-ray image and get an AI-powered prediction.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bone Tumor Detection AI",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=Space+Grotesk:wght@600;700&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main { background-color: #0f172a; }
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }

    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 50%, #5eead4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-tumor {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 1px solid #ef4444;
    }
    .result-no-tumor {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 1px solid #22c55e;
    }
    .result-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .confidence-text {
        font-size: 1.2rem;
        opacity: 0.85;
    }
    .metric-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #14b8a6;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }
    .disclaimer {
        background: rgba(234,179,8,0.1);
        border: 1px solid rgba(234,179,8,0.3);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        color: #fde68a;
        font-size: 0.88rem;
        margin-top: 1.5rem;
    }
    .stFileUploader > div > div {
        border: 2px dashed #0d9488 !important;
        border-radius: 12px !important;
        background: rgba(13,148,136,0.05) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0d9488, #0891b2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100%;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(13,148,136,0.4) !important;
    }
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #0d9488, #5eead4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ──────────────────────────────────────────────────────────────
IMG_SIZE   = 224
MODEL_PATH = "model/bone_tumor_model.keras"
CLASS_PATH = "model/class_indices.json"

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_indices():
    if not os.path.exists(CLASS_PATH):
        return {"no_tumor": 0, "tumor": 1}
    with open(CLASS_PATH) as f:
        return json.load(f)

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block_16_project_BN"):
    """Generate Grad-CAM heatmap for explainability."""
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)[0]
        pooled = tf.reduce_mean(grads, axis=(0, 1))
        heatmap = conv_outputs[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
        return heatmap
    except Exception:
        return None

def overlay_heatmap(original_img: Image.Image, heatmap: np.ndarray) -> Image.Image:
    img_array = np.array(original_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    heatmap_resized = np.uint8(255 * plt.cm.jet(
        np.array(Image.fromarray(np.uint8(heatmap * 255)).resize((IMG_SIZE, IMG_SIZE)))
    ))[:, :, :3]
    superimposed = (img_array * 0.6 + heatmap_resized * 0.4).astype(np.uint8)
    return Image.fromarray(superimposed)

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦴 About This App")
    st.markdown("""
This tool uses **deep learning** (MobileNetV2 Transfer Learning)
to classify bone X-ray images as:
- 🔴 **Tumor Detected**
- 🟢 **No Tumor**

**Model Architecture**
- Base: MobileNetV2 (ImageNet weights)
- Fine-tuned on bone X-ray dataset
- Input: 224×224 RGB images

**Output**
- Binary prediction
- Confidence score (%)
- Grad-CAM heatmap overlay
""")
    st.divider()
    st.markdown("**Built with**")
    st.markdown("TensorFlow · Streamlit · OpenCV · Scikit-learn")
    st.divider()
    st.caption("⚠️ For educational/research use only. Not a substitute for medical diagnosis.")

# ─── MAIN LAYOUT ────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🦴 Bone Tumor Detection AI</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload a bone X-ray image — our AI model analyzes it for signs of tumor in seconds.</p>', unsafe_allow_html=True)

model = load_model()
class_indices = load_class_indices()

# Invert: {label: idx} → {idx: label}
idx_to_label = {v: k for k, v in class_indices.items()}

if model is None:
    st.warning("⚠️ Trained model not found at `model/bone_tumor_model.keras`.")
    st.info("Run `python train.py` first to train the model, then relaunch this app.")
    st.stop()

# ─── UPLOAD ─────────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📤 Upload X-Ray Image")
    uploaded = st.file_uploader(
        "Drag & drop or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)
        st.markdown(f"""
        <div style="display:flex; gap:1rem; margin-top:1rem;">
            <div class="metric-box">
                <div class="metric-value">{image.size[0]}px</div>
                <div class="metric-label">Width</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{image.size[1]}px</div>
                <div class="metric-label">Height</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{image.mode}</div>
                <div class="metric-label">Mode</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        predict_btn = st.button("🔍 Analyze X-Ray", use_container_width=True)
    else:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03);border-radius:12px;
                    padding:3rem;text-align:center;border:1px dashed #334155;margin-top:1rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">🩻</div>
            <div style="color:#64748b;font-size:0.95rem;">Upload an X-ray to get started</div>
        </div>
        """, unsafe_allow_html=True)
        predict_btn = False

# ─── RESULT ─────────────────────────────────────────────────────────────────
with col_result:
    st.markdown("### 📊 Analysis Result")

    if uploaded and predict_btn:
        with st.spinner("Analyzing image..."):
            img_array   = preprocess_image(image)
            raw_pred    = float(model.predict(img_array, verbose=0)[0][0])

            # class_indices maps folder names → ints. "tumor" folder = positive class.
            tumor_idx = class_indices.get("tumor", 1)
            if tumor_idx == 1:
                tumor_prob = raw_pred
            else:
                tumor_prob = 1 - raw_pred

            is_tumor    = tumor_prob >= 0.5
            confidence  = tumor_prob if is_tumor else (1 - tumor_prob)
            label       = "Tumor Detected" if is_tumor else "No Tumor Detected"
            emoji       = "🔴" if is_tumor else "🟢"
            card_class  = "result-tumor" if is_tumor else "result-no-tumor"
            color       = "#fca5a5" if is_tumor else "#86efac"

        st.markdown(f"""
        <div class="result-card {card_class}">
            <div style="font-size:3.5rem;margin-bottom:0.5rem;">{emoji}</div>
            <div class="result-title" style="color:{color};">{label}</div>
            <div class="confidence-text" style="color:{color};">
                Confidence: <strong>{confidence*100:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        st.markdown("**Confidence Breakdown**")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Tumor Probability",    f"{tumor_prob*100:.1f}%")
        with c2:
            st.metric("No-Tumor Probability", f"{(1-tumor_prob)*100:.1f}%")

        st.progress(float(tumor_prob))

        # Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, model)
        if heatmap is not None:
            st.markdown("**🌡️ Grad-CAM Attention Map**")
            st.caption("Highlighted regions show where the model focused.")
            overlay = overlay_heatmap(image, heatmap)
            st.image(overlay, use_column_width=True)

        # Disclaimer
        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Medical Disclaimer:</strong>
            This AI prediction is for <em>research and educational purposes only</em>.
            It is <strong>not</strong> a substitute for professional medical advice,
            diagnosis, or treatment. Always consult a qualified radiologist or physician.
        </div>
        """, unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03);border-radius:12px;
                    padding:3rem;text-align:center;border:1px dashed #334155;margin-top:1rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">📈</div>
            <div style="color:#64748b;font-size:0.95rem;">
                Results will appear here after analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#475569;font-size:0.85rem;padding:1rem 0;">
    Built with TensorFlow + Streamlit · MobileNetV2 Transfer Learning · 
    Grad-CAM Explainability
</div>
""", unsafe_allow_html=True)
