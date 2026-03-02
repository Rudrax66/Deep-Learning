import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Brain Tumor AI", page_icon="🧠", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');
* { font-family: 'Rajdhani', sans-serif; }
.stApp { background: #050510; color: #c0c0e0; }
[data-testid="stSidebar"] { background: #080818 !important; border-right: 1px solid #1a1a40; }
.block-container { padding: 2rem 2rem 1rem 2rem; }
h1,h2,h3 { font-family: 'Rajdhani', sans-serif; font-weight: 700; }
.title { font-size:2.8rem; font-weight:700; color:#00e5ff; letter-spacing:2px; }
.subtitle { color:#404070; font-size:0.8rem; letter-spacing:4px; text-transform:uppercase; }
.card { background:#0d0d20; border:1px solid #1a1a40; border-radius:12px; padding:1.2rem; margin-bottom:1rem; }
.result-tumor { background:#1a0510; border:2px solid #ff2060; border-radius:16px; padding:1.5rem; text-align:center; }
.result-safe  { background:#051a10; border:2px solid #00ff88; border-radius:16px; padding:1.5rem; text-align:center; }
.big-num { font-family:'Share Tech Mono',monospace; font-size:3rem; font-weight:700; }
.tag { background:#0d0d20; border:1px solid #1a1a40; border-radius:6px; padding:2px 10px; font-size:0.75rem; color:#6060a0; font-family:'Share Tech Mono',monospace; margin:2px; display:inline-block; }
.stButton>button { background:linear-gradient(135deg,#0050ff,#00e5ff) !important; border:none !important; border-radius:10px !important; color:white !important; font-family:'Rajdhani',sans-serif !important; font-weight:700 !important; font-size:1rem !important; letter-spacing:1px !important; padding:0.5rem 1.5rem !important; }
.stDownloadButton>button { background:linear-gradient(135deg,#006600,#00cc44) !important; border:none !important; border-radius:10px !important; color:white !important; font-family:'Rajdhani',sans-serif !important; font-weight:700 !important; }
</style>
""", unsafe_allow_html=True)

# ── Model — loads ONCE, cached forever ───────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model... (one time only)")
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 2))
    model.eval()
    return model

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(img):
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = torch.softmax(load_model()(tensor), dim=1)[0]
    return out.numpy()

# ── Charts ────────────────────────────────────────────────────────────────────
def bar_chart(probs):
    fig = go.Figure(go.Bar(
        x=["No Tumor", "Tumor Detected"],
        y=[p*100 for p in probs],
        marker_color=["#00ff88", "#ff2060"],
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside"
    ))
    fig.update_layout(
        paper_bgcolor="#0d0d20", plot_bgcolor="#0d0d20",
        font=dict(color="#c0c0e0", family="Rajdhani"),
        yaxis=dict(range=[0, 115], gridcolor="#1a1a40", title="Probability (%)"),
        margin=dict(t=20, b=20, l=10, r=10), height=260
    )
    return fig

def gauge_chart(conf, has_tumor):
    color = "#ff2060" if has_tumor else "#00ff88"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf,
        number={"suffix": "%", "font": {"color": color, "size": 38, "family": "Share Tech Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#404070"},
            "bar": {"color": color},
            "bgcolor": "#0d0d20",
            "steps": [{"range": [0, 50], "color": "#0a0a1a"}, {"range": [50, 100], "color": "#0d0d1a"}],
        }
    ))
    fig.update_layout(paper_bgcolor="#0d0d20", font={"color": "#c0c0e0"},
                      margin=dict(t=30, b=10, l=20, r=20), height=210)
    return fig

def pixel_chart(img):
    gray = np.array(img.convert("L"))
    hist, bins = np.histogram(gray.flatten(), bins=64, range=[0, 256])
    fig = go.Figure(go.Bar(x=bins[:-1], y=hist, marker_color="#0050ff", opacity=0.8))
    fig.update_layout(
        paper_bgcolor="#0d0d20", plot_bgcolor="#0d0d20",
        font=dict(color="#c0c0e0", family="Rajdhani"),
        xaxis=dict(title="Pixel Intensity", gridcolor="#1a1a40"),
        yaxis=dict(title="Count", gridcolor="#1a1a40"),
        margin=dict(t=10, b=20, l=10, r=10), height=200,
        title=dict(text="Pixel Intensity Distribution", font=dict(size=13))
    )
    return fig

# ── Report ────────────────────────────────────────────────────────────────────
def make_report(fname, probs, pred, conf, size):
    label = "TUMOR DETECTED" if pred == 1 else "NO TUMOR DETECTED"
    risk  = "HIGH" if pred == 1 and conf > 75 else "MEDIUM" if pred == 1 else "LOW"
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    eat   = (
        "AVOID: processed food, alcohol, sugar\nEAT: turmeric, leafy greens, omega-3, berries"
        if pred == 1 else
        "EAT: fruits, vegetables, whole grains, nuts\nExercise 30 min/day, stay hydrated"
    )
    return f"""
╔══════════════════════════════════════════════════════╗
║        BRAIN TUMOR DETECTION — AI REPORT             ║
╚══════════════════════════════════════════════════════╝

DATE & TIME   : {now}
FILE          : {fname}
IMAGE SIZE    : {size[0]} x {size[1]} px
MODEL         : ResNet50 (Pretrained)

──────────────────────────────────────────────────────
  RESULT      : {label}
  CONFIDENCE  : {conf:.2f}%
  RISK LEVEL  : {risk}
──────────────────────────────────────────────────────

CLASS PROBABILITIES
  No Tumor    : {probs[0]*100:.2f}%  {"█"*int(probs[0]*25)}
  Tumor       : {probs[1]*100:.2f}%  {"█"*int(probs[1]*25)}

──────────────────────────────────────────────────────
DIETARY RECOMMENDATION
  {eat}

──────────────────────────────────────────────────────
MEDICAL ADVICE
{"  ⚠ Consult neurologist immediately. Further imaging required." if pred==1 and conf>75
  else "  ⚠ Follow-up scan recommended within 2-4 weeks." if pred==1
  else "  ✅ No tumor detected. Maintain regular health check-ups."}

──────────────────────────────────────────────────────
DISCLAIMER: AI screening tool only.
NOT a substitute for professional medical diagnosis.
──────────────────────────────────────────────────────
"""

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="title" style="font-size:1.4rem">🧠 BrainAI</div>', unsafe_allow_html=True)
    st.divider()
    threshold       = st.slider("Confidence Threshold (%)", 50, 95, 70)
    enhance         = st.checkbox("Enhance Contrast", True)
    show_channels   = st.checkbox("Show RGB Channels", False)
    st.divider()
    st.markdown('<span class="tag">ResNet50</span> <span class="tag">PyTorch</span>', unsafe_allow_html=True)
    st.info("⚡ Model is cached — only loads once. Fast from 2nd scan onward!")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="title">Brain Tumor Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI · ResNet50 · Real-time MRI Analysis</div><br>', unsafe_allow_html=True)

# Pre-load model (so first scan is fast)
model = load_model()
st.success("⚡ Model ready!", icon="✅")

col1, col2 = st.columns([1, 1.6], gap="large")

# ── Left: Upload + Image Info ─────────────────────────────────────────────────
with col1:
    st.markdown("#### 📤 Upload MRI Scan")
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","bmp"], label_visibility="collapsed")

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        disp = ImageEnhance.Contrast(img).enhance(1.4) if enhance else img
        st.image(disp, caption="MRI Scan", use_column_width=True)
        st.markdown(f'<span class="tag">📐 {img.size[0]}×{img.size[1]}</span> <span class="tag">📦 {len(uploaded.getvalue())//1024} KB</span>', unsafe_allow_html=True)

        if show_channels:
            st.markdown("**RGB Channels**")
            r, g, b = img.split()
            c1,c2,c3 = st.columns(3)
            c1.image(r, caption="Red",   use_column_width=True)
            c2.image(g, caption="Green", use_column_width=True)
            c3.image(b, caption="Blue",  use_column_width=True)

        st.plotly_chart(pixel_chart(img), use_container_width=True)

# ── Right: Analyze + Results ──────────────────────────────────────────────────
with col2:
    if uploaded:
        if st.button("🔬 ANALYZE SCAN", use_container_width=True):
            t0 = time.time()
            with st.spinner("Analyzing MRI..."):
                probs = predict(img)
            elapsed = (time.time() - t0) * 1000

            pred      = int(np.argmax(probs))
            conf      = probs[pred] * 100
            has_tumor = pred == 1
            color     = "#ff2060" if has_tumor else "#00ff88"
            label     = "TUMOR DETECTED" if has_tumor else "NO TUMOR FOUND"
            icon      = "⚠️" if has_tumor else "✅"

            # ── Result Card ────────────────────────────────────────────────
            st.markdown(f"""
            <div class="{'result-tumor' if has_tumor else 'result-safe'}">
                <div style="font-size:2.5rem">{icon}</div>
                <div style="font-size:1.8rem;font-weight:700;color:{color};letter-spacing:2px">{label}</div>
                <div class="big-num" style="color:{color}">{conf:.1f}%</div>
                <div style="color:#505070;font-size:0.8rem;margin-top:4px">⏱ Inference: {elapsed:.0f} ms</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Gauge + Bar Chart ──────────────────────────────────────────
            g_col, b_col = st.columns(2)
            with g_col:
                st.markdown("**Confidence Gauge**")
                st.plotly_chart(gauge_chart(conf, has_tumor), use_container_width=True)
            with b_col:
                st.markdown("**Class Probabilities**")
                st.plotly_chart(bar_chart(probs), use_container_width=True)

            # ── Risk Bar ───────────────────────────────────────────────────
            risk_pct   = probs[1] * 100
            risk_color = "#ff2060" if risk_pct > 75 else "#ff8800" if risk_pct > 50 else "#00ff88"
            st.markdown(f"""
            <div class="card">
              <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span>Tumor Risk Meter</span>
                <span style="font-family:'Share Tech Mono',monospace;color:{risk_color}">{risk_pct:.1f}%</span>
              </div>
              <div style="background:#0a0a1a;border-radius:50px;height:14px;overflow:hidden">
                <div style="width:{risk_pct:.0f}%;height:100%;background:{risk_color};border-radius:50px"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── What to Eat ────────────────────────────────────────────────
            st.markdown("**🥗 What to Eat / Health Tips**")
            if has_tumor and conf >= threshold:
                st.error("""
**Immediate Action:**
- 🏥 See a neurologist/oncologist right away
- 🧪 MRI with contrast + CT scan + biopsy needed
- 🥦 Eat: turmeric, leafy greens, berries, omega-3 rich food
- 🚫 Avoid: processed food, alcohol, sugar, smoking
""")
            elif has_tumor:
                st.warning("""
**Follow-up Recommended:**
- 📅 Repeat MRI in 2–4 weeks
- 🥑 Eat: antioxidants, nuts, fresh vegetables
- 💧 Drink 2–3L water daily
- 😴 7–9 hours sleep
""")
            else:
                st.success("""
**Stay Healthy ✅**
- 🥗 Balanced diet: fruits, veggies, whole grains
- 🏃 Exercise 30 min/day
- 🧘 Manage stress: yoga / meditation
- 📅 Annual MRI check-up recommended
""")

            # ── Download Report ────────────────────────────────────────────
            report = make_report(uploaded.name, probs, pred, conf, img.size)
            st.download_button(
                "📄 Download Report (.txt)",
                data=report,
                file_name=f"brain_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:4rem 2rem;margin-top:2rem">
            <div style="font-size:4rem;opacity:0.3">🧠</div>
            <div style="color:#404070;margin-top:1rem">Upload a brain MRI scan to begin</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()
st.markdown('<div style="text-align:center;color:#303060;font-size:0.75rem">Brain Tumor AI · For Educational Use Only · Not a Medical Device</div>', unsafe_allow_html=True)
