"""Streamlit demo:  streamlit run app/demo.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import streamlit as st

from src.predict import load_model, predict_risk

st.set_page_config(page_title="CVD Risk Predictor", page_icon="❤️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg,#b91c1c,#7f1d1d); color:white;
       padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1.2rem;}
.hero h1 {margin:0; font-size: 1.9rem; color:white;}
.hero p {margin:.3rem 0 0 0; opacity:.9;}
.card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:1rem 1.2rem;
       margin-bottom:1rem; background: rgba(128,128,128,.05);}
.card h4 {margin-top:0;}
.result {text-align:center; padding: 1.2rem; border-radius:14px; color:white;}
.result .pct {font-size: 3.2rem; font-weight: 700; line-height:1.1;}
.result .lbl {font-size: 1.1rem; letter-spacing:.08em;}
.gauge {position:relative; height:14px; border-radius:7px; margin:1.1rem 0 .3rem 0;
        background: linear-gradient(90deg,#22c55e 0%,#eab308 35%,#ef4444 70%,#7f1d1d 100%);}
.gauge .pin {position:absolute; top:-6px; width:4px; height:26px; background:#111;
             border:1px solid white; border-radius:2px;}
.small {font-size:.8rem; opacity:.7;}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>❤️ 10-Year Cardiovascular Risk Predictor</h1>
<p>Predicts coronary heart disease (CHD) risk from routine clinical measurements ·
Framingham Heart Study · Educational project, not medical advice</p></div>""",
            unsafe_allow_html=True)

DEFAULTS = dict(male=1, age=55, edu=2, smoker=False, cigs=0, bpmeds=False, stroke=False,
                hyp=False, diab=False, chol=220, sys=130, dia=85, bmi=26.0, hr=75, glu=85)
PRESETS = {
    "Low risk": dict(male=0, age=38, edu=3, smoker=False, cigs=0, bpmeds=False, stroke=False,
                     hyp=False, diab=False, chol=175, sys=112, dia=72, bmi=22.5, hr=68, glu=78),
    "Moderate risk": dict(male=1, age=56, edu=2, smoker=True, cigs=10, bpmeds=False, stroke=False,
                          hyp=True, diab=False, chol=235, sys=142, dia=90, bmi=28.0, hr=76, glu=92),
    "High risk": dict(male=1, age=68, edu=1, smoker=True, cigs=25, bpmeds=True, stroke=False,
                      hyp=True, diab=True, chol=290, sys=185, dia=105, bmi=33.0, hr=88, glu=180),
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(name):
    st.session_state.update(PRESETS[name])


@st.cache_resource
def get_models():
    return {"XGBoost": load_model("xgboost"), "Logistic Regression": load_model("logreg")}


try:
    models = get_models()
except FileNotFoundError:
    st.error("Trained models not found. Run `python run_all.py` first.")
    st.stop()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    choice = st.radio("Primary model", list(models))
    st.divider()
    st.subheader("Quick patient presets")
    for name in PRESETS:
        st.button(name, on_click=apply_preset, args=(name,))
    st.button("Reset", on_click=apply_preset, args=("Low risk",),
              help="Resets to the low-risk example")
    st.divider()
    st.caption("Risk bands: <10% low · 10-20% moderate · >20% high\n\n"
               "Dataset average 10-year CHD rate: ~15%")

tab_pred, tab_perf, tab_about = st.tabs(["🩺 Predict", "📊 Model performance", "ℹ️ About"])

# ---------------- Predict tab ----------------
with tab_pred:
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown("#### Patient profile")
        a, b = st.columns(2)
        a.selectbox("Sex", [1, 0], key="male", format_func=lambda x: "Male" if x else "Female")
        b.selectbox("Education level", [1, 2, 3, 4], key="edu",
                    help="1 = some high school … 4 = college or higher")
        st.slider("Age (years)", 30, 90, key="age")
        st.divider()

        st.markdown("#### Vitals & labs")
        c, d = st.columns(2)
        c.slider("Systolic BP (mmHg)", 80, 250, key="sys")
        d.slider("Diastolic BP (mmHg)", 40, 150, key="dia")
        c.slider("Total cholesterol (mg/dL)", 100, 400, key="chol")
        d.slider("Glucose (mg/dL)", 40, 300, key="glu")
        c.slider("BMI", 15.0, 50.0, step=0.1, key="bmi")
        d.slider("Heart rate (bpm)", 40, 140, key="hr")
        st.divider()

        st.markdown("#### History & lifestyle")
        e, f = st.columns(2)
        e.checkbox("Current smoker", key="smoker")
        e.slider("Cigarettes per day", 0, 70, key="cigs")
        f.checkbox("On BP medication", key="bpmeds")
        f.checkbox("Diabetic", key="diab")
        f.checkbox("Prevalent stroke", key="stroke")
        f.checkbox("Diagnosed hypertension", key="hyp",
                   help="Also set automatically when BP ≥ 140/90")
        st.divider()

    s = st.session_state
    smoker = bool(s.smoker or s.cigs > 0)
    hyp = bool(s.hyp or s.sys >= 140 or s.dia >= 90)
    patient = dict(male=s.male, age=s.age, education=s.edu, currentSmoker=int(smoker),
                   cigsPerDay=s.cigs, BPMeds=int(s.bpmeds), prevalentStroke=int(s.stroke),
                   prevalentHyp=int(hyp), diabetes=int(s.diab), totChol=s.chol,
                   sysBP=s.sys, diaBP=s.dia, BMI=s.bmi, heartRate=s.hr, glucose=s.glu)

    risk = predict_risk(patient, models[choice])
    other = [m for m in models if m != choice][0]
    risk_other = predict_risk(patient, models[other])
    if risk < 0.10:
        label, color = "LOW RISK", "#16a34a"
    elif risk < 0.20:
        label, color = "MODERATE RISK", "#d97706"
    else:
        label, color = "HIGH RISK", "#dc2626"

    with right:
        st.markdown(f"""<div class="result" style="background:{color}">
            <div class="lbl">{label}</div><div class="pct">{risk:.1%}</div>
            <div>Predicted 10-year CHD risk · {choice}</div></div>""",
                    unsafe_allow_html=True)
        pin = min(risk / 0.6, 1.0) * 100
        st.markdown(f'<div class="gauge"><div class="pin" style="left:calc({pin}% - 2px)"></div></div>'
                    '<div class="small" style="display:flex;justify-content:space-between">'
                    '<span>0%</span><span>30%</span><span>60%+</span></div>',
                    unsafe_allow_html=True)

        st.write("")
        m1, m2 = st.columns(2)
        m1.metric(choice, f"{risk:.1%}")
        m2.metric(other, f"{risk_other:.1%}", delta=f"{(risk_other - risk) * 100:+.1f} pts",
                  delta_color="off")
        if abs(risk - risk_other) > 0.20:
            st.info("The models disagree strongly. Logistic regression extrapolates linearly "
                    "for extreme inputs, while tree models cap at what they saw in training.")

        st.markdown("##### Key flags")
        flags = []
        if s.sys >= 140 or s.dia >= 90: flags.append("Elevated blood pressure")
        if s.chol >= 240: flags.append("High cholesterol")
        if s.glu >= 126 or s.diab: flags.append("Diabetes / high glucose")
        if s.bmi >= 30: flags.append("Obesity (BMI ≥ 30)")
        if smoker: flags.append("Current smoker")
        if s.age >= 65: flags.append("Age 65+")
        if flags:
            for fl in flags: st.markdown(f"- ⚠️ {fl}")
        else:
            st.markdown("- ✅ No major risk flags")
        st.caption("Adjust inputs on the left, the prediction updates instantly.")

# ---------------- Performance tab ----------------
with tab_perf:
    mpath = ROOT / "results" / "metrics.csv"
    figs = ROOT / "results" / "figures"
    if mpath.exists():
        st.subheader("Test-set metrics")
        st.dataframe(pd.read_csv(mpath), hide_index=True)
    c1, c2 = st.columns(2)
    for col, fname, cap in [(c1, "roc_curves.png", "ROC curves"),
                            (c2, "pr_curves.png", "Precision-Recall curves"),
                            (c1, "shap_bar.png", "Feature importance (SHAP)"),
                            (c2, "fairness_xgboost.png", "Subgroup fairness (XGBoost)")]:
        if (figs / fname).exists():
            col.image(str(figs / fname), caption=cap)
    if not mpath.exists():
        st.info("Run `python run_all.py` to generate metrics and figures.")

# ---------------- About tab ----------------
with tab_about:
    st.markdown("""
**Problem:** estimate a patient's 10-year risk of coronary heart disease from routine measurements.

**Data:** Framingham Heart Study (4,240 patients, 15 features, ~15% positive).
It is a clinical cohort used as a stand-in for EHR data, which needs credentialed access.

**Models:** regularised Logistic Regression (Ridge / Lasso / Elastic Net) vs XGBoost,
tuned with 10-fold cross-validation and evaluated with AUROC / AUPRC on a held-out 20% test set.

**Fairness:** performance is checked across sex and age groups (see *Model performance*).

**Limitations:** small dataset, one population, not clinically validated.
""")