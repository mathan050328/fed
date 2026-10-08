"""
AI Healthcare Prediction & Explainable AI (XAI) Consultant Dashboard.
Modern aesthetic with proper text visibility across all sections.
"""

import sys
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from src.config import DISEASES, DISEASE_DISPLAY_NAMES, get_disease_paths
from src.evaluation.xai_agent import ExplainableAIAgent

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedAI — Federated Clinical Intelligence",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

PPTX_FILE_PATH = BASE_DIR / "Federated_Disease_Prediction_Presentation.pptx"
ZIP_FILE_PATH  = BASE_DIR / "federated_disease_prediction_project.zip"


def md_to_html(text: str) -> str:
    """Convert basic markdown to HTML for safe inline rendering."""
    # Bold: **text** → <strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Inline code: `text`
    text = re.sub(r'`([^`]+)`', r'<code style="background:rgba(255,255,255,0.15);padding:2px 6px;border-radius:4px;font-size:0.88em;">\1</code>', text)
    # Numbered list items: 1. text
    text = re.sub(r'(?m)^(\d+)\.\s+(.+)', r'<div class="list-item"><span class="list-num">\1</span><span>\2</span></div>', text)
    # Bullet list items: - text or • text
    text = re.sub(r'(?m)^[-•]\s+(.+)', r'<div class="list-item"><span class="bullet">▸</span><span>\1</span></div>', text)
    # Line breaks
    text = text.replace('\n\n', '<br><br>').replace('\n', '<br>')
    return text


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stMarkdown, p, span, label, div {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1180px; }

/* ─── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1120 0%, #0f172a 50%, #0b1120 100%) !important;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] .stMarkdown *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] h2 { color: #38bdf8 !important; font-size: 1.05rem !important; font-weight: 800 !important; }
[data-testid="stSidebar"] h3 { color: #7dd3fc !important; font-size: 0.85rem !important; font-weight: 700 !important; }
[data-testid="stSidebar"] .stDownloadButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    padding: 10px !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] label { color: #94a3b8 !important; font-size: 0.78rem !important; }

/* ─── Hero Banner ─────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0b1120 0%, #0f2a4a 45%, #0e3c5e 100%);
    border-radius: 18px;
    padding: 34px 40px;
    margin-bottom: 26px;
    border: 1px solid #1d4ed8;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: -80px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 900;
    color: #f8fafc !important;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.hero-title span { color: #38bdf8 !important; }
.hero-sub {
    font-size: 0.95rem;
    color: #94a3b8 !important;
    margin: 0 0 18px 0;
    line-height: 1.65;
    max-width: 680px;
}
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.28);
    color: #7dd3fc !important;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 4px 13px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-right: 8px;
    margin-bottom: 6px;
}

/* ─── Section Headers ─────────────────────────────────────── */
.sec-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 26px 0 14px 0;
}
.sec-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.sec-title { font-size: 1.1rem; font-weight: 800; color: #0f172a !important; margin: 0; }
.sec-sub   { font-size: 0.78rem; color: #64748b !important; margin: 2px 0 0 0; }

/* ─── Submit Button ───────────────────────────────────────── */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #4f46e5 100%) !important;
    color: #ffffff !important;
    font-size: 1.0rem !important;
    font-weight: 800 !important;
    padding: 16px 32px !important;
    border-radius: 12px !important;
    border: none !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 6px 24px rgba(29,78,216,0.4) !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    box-shadow: 0 10px 32px rgba(29,78,216,0.55) !important;
    transform: translateY(-2px) !important;
}

/* ─── Risk Cards ──────────────────────────────────────────── */
.risk-high {
    background: linear-gradient(145deg, #fff1f2, #ffe4e6);
    border: 2px solid #fca5a5;
    border-radius: 18px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 6px 24px rgba(239,68,68,0.14);
}
.risk-low {
    background: linear-gradient(145deg, #f0fdf4, #dcfce7);
    border: 2px solid #86efac;
    border-radius: 18px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 6px 24px rgba(34,197,94,0.14);
}
.risk-icon  { font-size: 2.6rem; display: block; margin-bottom: 6px; }
.risk-lbl-h { font-size: 1.1rem; font-weight: 800; color: #b91c1c !important; margin: 0; }
.risk-lbl-l { font-size: 1.1rem; font-weight: 800; color: #15803d !important; margin: 0; }
.risk-pct-h { font-size: 2.6rem; font-weight: 900; color: #dc2626 !important; margin: 6px 0; line-height: 1; }
.risk-pct-l { font-size: 2.6rem; font-weight: 900; color: #16a34a !important; margin: 6px 0; line-height: 1; }
.risk-rec   {
    font-size: 0.78rem; font-weight: 600; color: #475569 !important;
    margin-top: 10px; padding: 5px 14px;
    background: rgba(255,255,255,0.65); border-radius: 20px; display: inline-block;
}

/* ─── Glass Info Card ─────────────────────────────────────── */
.glass-card {
    background: linear-gradient(145deg, #f8fafc, #f1f5f9);
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.glass-card-title {
    font-size: 0.92rem; font-weight: 700;
    color: #0f172a !important;
    margin: 0 0 12px 0;
}

/* ─── Briefing Box ────────────────────────────────────────── */
.briefing-box {
    background: linear-gradient(135deg, #eff6ff, #eef2ff);
    border: 1px solid #bfdbfe;
    border-left: 5px solid #3b82f6;
    border-radius: 0 14px 14px 0;
    padding: 22px 26px;
    margin: 10px 0;
    color: #1e3a8a !important;
    font-size: 0.93rem;
    line-height: 1.75;
}
.briefing-box strong { color: #1d4ed8 !important; font-weight: 700; }
.briefing-box p, .briefing-box li, .briefing-box span { color: #1e3a8a !important; }

/* ─── Quick Buttons ───────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(145deg, #f8fafc, #f1f5f9) !important;
    color: #1e293b !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    padding: 11px 6px !important;
    line-height: 1.4 !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: #ffffff !important;
    border-color: transparent !important;
    box-shadow: 0 5px 16px rgba(14,165,233,0.4) !important;
    transform: translateY(-2px) !important;
}

/* ─── Text Input ──────────────────────────────────────────── */
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid #cbd5e1 !important;
    padding: 11px 16px !important;
    font-size: 0.92rem !important;
    background: #f8fafc !important;
    color: #0f172a !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    outline: none !important;
}

/* ─── Consultant Dark Panel ───────────────────────────────── */
.consult-panel {
    background: linear-gradient(145deg, #0b1120, #0f1f35);
    border: 1px solid #1e3a5f;
    border-radius: 18px;
    padding: 30px 32px;
    margin-top: 18px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}
.con-badge {
    display: inline-block;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: #ffffff !important;
    font-size: 0.68rem;
    font-weight: 800;
    padding: 5px 15px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 14px;
}
.con-title {
    font-size: 1.3rem;
    font-weight: 900;
    color: #f0f9ff !important;
    margin: 0 0 18px 0;
    line-height: 1.3;
}
.con-body {
    color: #cbd5e1 !important;
    font-size: 0.94rem;
    line-height: 1.8;
}
.con-body strong { color: #7dd3fc !important; font-weight: 700; }
.con-body code   { color: #a5f3fc !important; }
.con-body .list-item {
    display: flex;
    gap: 10px;
    margin: 8px 0;
    color: #cbd5e1 !important;
    align-items: flex-start;
}
.con-body .list-num {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white !important;
    font-size: 0.72rem;
    font-weight: 800;
    width: 22px; height: 22px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}
.con-body .bullet {
    color: #38bdf8 !important;
    font-size: 0.85rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.con-body span { color: #cbd5e1 !important; }

/* ─── Gradient Divider ────────────────────────────────────── */
.gdiv {
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, #3b82f6 30%, #6366f1 70%, transparent 100%);
    border: none;
    margin: 30px 0;
    border-radius: 2px;
}

/* ─── Progress bar ────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444) !important;
    border-radius: 6px !important;
}

/* ─── Disclaimer ──────────────────────────────────────────── */
.disclaimer {
    background: linear-gradient(135deg, #fefce8, #fef9c3);
    border: 1px solid #fde047;
    border-left: 5px solid #eab308;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin-top: 32px;
    color: #713f12 !important;
    font-size: 0.84rem;
    line-height: 1.65;
}
.disclaimer strong { color: #92400e !important; }

/* ─── Dataframe ───────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }
</style>
""", unsafe_allow_html=True)


# ─── Cached Agent ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_xai_agent(disease: str):
    return ExplainableAIAgent(disease=disease)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 MedAI Platform")
    st.markdown("**Federated Clinical Intelligence**")
    st.markdown("---")

    selected_disease = st.selectbox(
        "🎯 Disease Domain",
        options=DISEASES,
        format_func=lambda x: {
            "heart":    "🫀  Heart Disease",
            "diabetes": "🩸  Diabetes",
            "ckd":      "🩺  Kidney Disease (CKD)",
        }[x],
    )

    disease_display = DISEASE_DISPLAY_NAMES[selected_disease]
    disease_icon    = {"heart": "🫀", "diabetes": "🩸", "ckd": "🩺"}[selected_disease]

    st.markdown("---")
    st.markdown("### 🏥 Network Status")
    st.markdown("""
<div style="font-size:0.82rem;line-height:2.2;color:#94a3b8;">
🟢 &nbsp;Hospital A — Active<br>
🟢 &nbsp;Hospital B — Active<br>
🟢 &nbsp;Hospital C — Active<br>
🔵 &nbsp;FedAvg Server — Online
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📥 Downloads")
    if PPTX_FILE_PATH.exists():
        with open(PPTX_FILE_PATH, "rb") as f:
            st.download_button("📊 Presentation (.pptx)", data=f,
                file_name="Federated_Disease_Prediction_Presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                use_container_width=True)
    if ZIP_FILE_PATH.exists():
        with open(ZIP_FILE_PATH, "rb") as f:
            st.download_button("📦 Full Project (.zip)", data=f,
                file_name="federated_disease_prediction_project.zip",
                mime="application/zip",
                use_container_width=True)

    st.markdown("---")
    st.caption("Privacy-preserving AI · Flower FedAvg · HIPAA-aligned")


# ─── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
  <p class="hero-title">{disease_icon} <span>{disease_display}</span> Risk Intelligence</p>
  <p class="hero-sub">
    Enter patient biomarkers below. The federated model — trained across 3 hospitals
    without sharing raw patient data — will estimate disease risk probability and generate
    an explainable AI diagnostic report.
  </p>
  <span class="hero-badge">🔒 Federated Learning</span>
  <span class="hero-badge">🧠 Explainable AI</span>
  <span class="hero-badge">🏥 Multi-Hospital</span>
  <span class="hero-badge">⚡ FedAvg · 5 Rounds</span>
</div>
""", unsafe_allow_html=True)


# ─── Patient Input Form ────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-head">
  <div class="sec-icon">📋</div>
  <div>
    <p class="sec-title">Patient Clinical Measurements</p>
    <p class="sec-sub">Fill biomarker values and submit for federated risk analysis</p>
  </div>
</div>
""", unsafe_allow_html=True)

with st.form("patient_clinical_form"):
    patient_input = {}

    # ── Heart Disease ──
    if selected_disease == "heart":
        c1, c2, c3 = st.columns(3)
        with c1:
            patient_input["age"]      = st.number_input("Age (years)", 18, 100, 56)
            patient_input["sex"]      = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
            patient_input["cp"]       = st.selectbox("Chest Pain Type", [0,1,2,3], index=2,
                                          format_func=lambda x: ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"][x])
            patient_input["trestbps"] = st.number_input("Resting BP (mm Hg)", 80, 220, 135)
            patient_input["chol"]     = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 250)
        with c2:
            patient_input["fbs"]      = st.selectbox("Fasting Blood Sugar > 120", [0,1], format_func=lambda x: "No (≤120)" if x==0 else "Yes (>120)")
            patient_input["restecg"]  = st.selectbox("Resting ECG", [0,1,2],
                                          format_func=lambda x: ["Normal","ST-T Abnormality","LV Hypertrophy"][x])
            patient_input["thalach"]  = st.number_input("Max Heart Rate (bpm)", 60, 220, 142)
            patient_input["exang"]    = st.selectbox("Exercise Angina", [0,1], index=1, format_func=lambda x: "No" if x==0 else "Yes")
        with c3:
            patient_input["oldpeak"]  = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.4, 0.1)
            patient_input["slope"]    = st.selectbox("ST Slope", [0,1,2], index=1,
                                          format_func=lambda x: ["Upsloping","Flat","Downsloping"][x])
            patient_input["ca"]       = st.selectbox("Major Vessels (ca)", [0,1,2,3], index=1)
            patient_input["thal"]     = st.selectbox("Thalassemia", [1,2,3], index=2,
                                          format_func=lambda x: {1:"Normal",2:"Fixed Defect",3:"Reversible Defect"}.get(x,str(x)))

    # ── Diabetes ──
    elif selected_disease == "diabetes":
        c1, c2 = st.columns(2)
        with c1:
            patient_input["pregnancies"]       = st.number_input("Pregnancies", 0, 17, 3)
            patient_input["glucose"]           = st.number_input("Plasma Glucose (mg/dl)", 50, 260, 148)
            patient_input["blood_pressure"]    = st.number_input("Diastolic BP (mm Hg)", 40, 130, 82)
            patient_input["skin_thickness"]    = st.number_input("Triceps Skinfold (mm)", 5, 99, 30)
        with c2:
            patient_input["insulin"]           = st.number_input("Serum Insulin (mu U/ml)", 0, 850, 130)
            patient_input["bmi"]               = st.number_input("BMI", 15.0, 65.0, 33.6, 0.1)
            patient_input["diabetes_pedigree"] = st.number_input("Diabetes Pedigree Function", 0.05, 2.50, 0.62, 0.01)
            patient_input["age"]               = st.number_input("Age (years)", 18, 100, 48)

    # ── CKD ──
    elif selected_disease == "ckd":
        c1, c2, c3 = st.columns(3)
        with c1:
            patient_input["age"]  = st.number_input("Age (years)", 10, 100, 58)
            patient_input["bp"]   = st.number_input("Blood Pressure (mm Hg)", 50, 140, 85)
            patient_input["sg"]   = st.selectbox("Specific Gravity", [1.005,1.010,1.015,1.020,1.025], index=2)
            patient_input["al"]   = st.selectbox("Albumin (al)", [0,1,2,3,4], index=2)
            patient_input["su"]   = st.selectbox("Sugar (su)", [0,1,2,3,4], index=1)
            patient_input["bgr"]  = st.number_input("Blood Glucose Random (mg/dl)", 50, 450, 150)
            patient_input["bu"]   = st.number_input("Blood Urea (mg/dl)", 10, 300, 58)
        with c2:
            patient_input["sc"]   = st.number_input("Serum Creatinine (mg/dl)", 0.4, 15.0, 2.4, 0.1)
            patient_input["sod"]  = st.number_input("Sodium (mEq/L)", 100, 160, 134)
            patient_input["pot"]  = st.number_input("Potassium (mEq/L)", 2.5, 8.0, 4.8, 0.1)
            patient_input["hemo"] = st.number_input("Hemoglobin (g/dl)", 5.0, 18.0, 10.5, 0.1)
            patient_input["pcv"]  = st.number_input("Packed Cell Volume", 15, 55, 33)
            patient_input["wbcc"] = st.number_input("WBC Count (cells/cumm)", 3000, 20000, 8900)
            patient_input["rbcc"] = st.number_input("RBC Count (millions/cmm)", 2.0, 7.5, 3.8, 0.1)
        with c3:
            patient_input["htn"]   = st.selectbox("Hypertension", [0,1], index=1, format_func=lambda x: "No" if x==0 else "Yes")
            patient_input["dm"]    = st.selectbox("Diabetes Mellitus", [0,1], index=1, format_func=lambda x: "No" if x==0 else "Yes")
            patient_input["cad"]   = st.selectbox("Coronary Artery Disease", [0,1], index=0, format_func=lambda x: "No" if x==0 else "Yes")
            patient_input["appet"] = st.selectbox("Appetite", [1,0], index=1, format_func=lambda x: "Good" if x==1 else "Poor")
            patient_input["pe"]    = st.selectbox("Pedal Edema", [0,1], index=1, format_func=lambda x: "No" if x==0 else "Yes")
            patient_input["ane"]   = st.selectbox("Anemia", [0,1], index=0, format_func=lambda x: "No" if x==0 else "Yes")

    predict_btn = st.form_submit_button(
        f"⚡  Analyze Patient — {disease_display} Risk Assessment",
        use_container_width=True,
    )


# ─── Results Section ──────────────────────────────────────────────────────────
# Clear stale session state when disease domain changes
if st.session_state.get("last_disease") != selected_disease:
    st.session_state.pop("last_patient_input", None)
    st.session_state.pop("last_disease", None)

if predict_btn or "last_patient_input" in st.session_state:
    if predict_btn:
        st.session_state["last_patient_input"] = patient_input
        st.session_state["last_disease"]       = selected_disease

    cur_input   = st.session_state.get("last_patient_input", patient_input)
    agent       = get_xai_agent(selected_disease)
    explanation = agent.explain_prediction(cur_input)

    proba = explanation["probability"]
    pred  = explanation["prediction"]

    # ── Divider + section head ──
    st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="sec-head">
  <div class="sec-icon">📊</div>
  <div>
    <p class="sec-title">Federated Diagnostic Assessment</p>
    <p class="sec-sub">Aggregated prediction from Hospital A · B · C via FedAvg</p>
  </div>
</div>
""", unsafe_allow_html=True)

    col_risk, col_attrs = st.columns([1, 1.4])

    # ── Risk card ──
    with col_risk:
        if pred == 1:
            st.markdown(f"""
<div class="risk-high">
  <span class="risk-icon">⚠️</span>
  <p class="risk-lbl-h">High Risk Detected</p>
  <p class="risk-pct-h">{proba:.1%}</p>
  <span class="risk-rec">🏥 Specialist Follow-up Recommended</span>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="risk-low">
  <span class="risk-icon">✅</span>
  <p class="risk-lbl-l">Low / Minimal Risk</p>
  <p class="risk-pct-l">{proba:.1%}</p>
  <span class="risk-rec">💚 Routine Preventative Care</span>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(proba)
        st.caption("Decision threshold: 50% · FedAvg across 3 hospitals · 5 FL rounds")

    # ── Attribution table ──
    with col_attrs:
        st.markdown('<div class="glass-card"><p class="glass-card-title">⚖️ Biomarker Influence Breakdown</p>', unsafe_allow_html=True)
        feat_df = pd.DataFrame(explanation["top_attributions"])[["feature","direction","contribution"]]
        feat_df["contribution"] = feat_df["contribution"].round(4)
        feat_df.columns = ["Biomarker", "Direction", "Attribution Score"]
        st.dataframe(feat_df, use_container_width=True, height=262)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Clinical norms ──
    if explanation.get("norms_table"):
        st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
        st.markdown("""
<div class="sec-head">
  <div class="sec-icon">🔬</div>
  <div>
    <p class="sec-title">Clinical Reference Comparison</p>
    <p class="sec-sub">Patient values vs. evidence-based clinical norms</p>
  </div>
</div>
""", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(explanation["norms_table"]), use_container_width=True)

    # ── XAI Briefing ──
    st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="sec-head">
  <div class="sec-icon">🤖</div>
  <div>
    <p class="sec-title">XAI Diagnostic Briefing</p>
    <p class="sec-sub">Automated clinical reasoning from the explainability engine</p>
  </div>
</div>
""", unsafe_allow_html=True)
    briefing_html = md_to_html(explanation["briefing"])
    st.markdown(f'<div class="briefing-box">{briefing_html}</div>', unsafe_allow_html=True)

    # ── Attribution chart ──
    st.markdown("""
<div class="sec-head" style="margin-top:24px;">
  <div class="sec-icon">📈</div>
  <div>
    <p class="sec-title">Visual Feature Attribution</p>
    <p class="sec-sub">SHAP-inspired local contribution · Red = risk escalator · Green = protective</p>
  </div>
</div>
""", unsafe_allow_html=True)

    top_attrs  = explanation["top_attributions"][:7]
    f_names    = [a["feature"] for a in top_attrs][::-1]
    f_contribs = [a["contribution"] for a in top_attrs][::-1]
    bar_colors = ["#ef4444" if c > 0 else "#22c55e" for c in f_contribs]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f1f5f9")
    bars = ax.barh(f_names, f_contribs, color=bar_colors, height=0.52, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=1.0, alpha=0.7)
    for bar, val in zip(bars, f_contribs):
        ax.text(
            val + (0.003 if val >= 0 else -0.003),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8.5, color="#374151", fontweight="600",
        )
    ax.set_xlabel("Contribution to Risk Log-Odds", fontsize=9, color="#475569")
    ax.set_title(f"{disease_display} — Patient Biomarker Attribution Map",
                 fontsize=10.5, fontweight="bold", color="#1e293b", pad=12)
    ax.tick_params(axis="both", labelsize=8.5, colors="#374151")
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#cbd5e1")
    ax.grid(axis="x", linestyle=":", alpha=0.4, color="#94a3b8")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ─── Interactive AI Consultant ─────────────────────────────────────────────
    st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
    st.markdown("""
<div class="sec-head">
  <div class="sec-icon">💬</div>
  <div>
    <p class="sec-title">Interactive Clinical AI Consultant</p>
    <p class="sec-sub">Ask the XAI agent anything about this patient's profile</p>
  </div>
</div>
""", unsafe_allow_html=True)

    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        btn_why  = st.button("🔍  Why this diagnosis?",   use_container_width=True, key="btn_why")
    with qc2:
        btn_plan = st.button("📉  How to reduce risk?",   use_container_width=True, key="btn_plan")
    with qc3:
        btn_labs = st.button("🧪  Recommended lab tests", use_container_width=True, key="btn_labs")
    with qc4:
        btn_priv = st.button("🔒  Privacy & FL audit",    use_container_width=True, key="btn_priv")

    st.markdown("<br>", unsafe_allow_html=True)
    custom_query = st.text_input(
        "", placeholder="💬  Ask a custom question, e.g. 'What diet changes will help?'",
        label_visibility="collapsed",
    )

    if btn_why:
        active_query = "why is the risk score high for this patient?"
    elif btn_plan:
        active_query = "how to reduce risk and what is the lifestyle treatment plan?"
    elif btn_labs:
        active_query = "what recommended laboratory tests should be ordered?"
    elif btn_priv:
        active_query = "is this model trained with federated learning privacy?"
    elif custom_query.strip():
        active_query = custom_query.strip()
    else:
        active_query = "why is the risk score high for this patient?"

    consultation = agent.ask_agent(cur_input, active_query)

    if isinstance(consultation, dict):
        badge   = consultation.get("badge",   "Consultant")
        title   = consultation.get("title",   "Clinical Analysis")
        content = consultation.get("content", "")
    else:
        badge, title, content = "Consultant", "Clinical Analysis", str(consultation)

    content_html = md_to_html(content)

    st.markdown(f"""
<div class="consult-panel">
  <span class="con-badge">{badge}</span>
  <p class="con-title">{title}</p>
  <div class="con-body">{content_html}</div>
</div>
""", unsafe_allow_html=True)


# ─── Disclaimer ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  <strong>⚠️ Academic Engineering Prototype:</strong>
  This platform demonstrates privacy-preserving Federated Learning (Flower · FedAvg) and
  Explainable AI feature attribution across a simulated 3-hospital network.
  All predictions are mathematical estimations for academic demonstration only and must
  <strong>not</strong> replace professional clinical evaluation or medical diagnosis.
</div>
""", unsafe_allow_html=True)
