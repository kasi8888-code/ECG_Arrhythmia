"""
Streamlit Web Application for ECG Arrhythmia Detection System
AI-assisted cardiac rhythm analysis with explainability

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import joblib
import os
import io
import requests
import base64
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
# Load .env from the same directory as this script
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Import project modules
import config
from inference import ECGArrhythmiaDetector
from feature_engineering import ECGFeatureExtractor
from explainability import plot_gradcam_explanation

# Page configuration
st.set_page_config(
    page_title="ECG Arrhythmia Detection System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical-grade styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E40AF;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F9FAFB;
        border-radius: 0.75rem;
        padding: 1.5rem;
        border: 1px solid #E5E7EB;
    }
    .success-box {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .warning-box {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .danger-box {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Model loading helpers ──────────────────────────────────────────────────────

@st.cache_resource
def load_detector():
    """Load CNN Hybrid model."""
    for p in [config.MODEL_DIR / 'final_model.pt', config.MODEL_DIR / 'best_model.pt']:
        if p.exists():
            return ECGArrhythmiaDetector(model_path=p)
    st.warning("⚠️ No trained CNN model found.")
    return ECGArrhythmiaDetector()

@st.cache_resource
def load_ml_model(model_key: str):
    """Load a traditional ML model (.joblib) by key."""
    paths = {
        "Random Forest (96.52%)": config.MODEL_DIR / "ml_models" / "random_forest.joblib",
        "XGBoost (95.47%)": config.MODEL_DIR / "ml_models" / "xgboost.joblib",
        "SVM (91.38%)": config.MODEL_DIR / "ml_models" / "svm_rbf.joblib",
    }
    path = paths.get(model_key)
    if path and path.exists():
        return joblib.load(path)
    return None

@st.cache_resource
def load_feature_stats():
    """Load feature normalization stats saved inside the CNN checkpoint."""
    p = config.MODEL_DIR / 'final_model.pt'
    if p.exists():
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        mean = ckpt.get('feature_mean')
        std  = ckpt.get('feature_std')
        if mean is not None and std is not None:
            return np.array(mean), np.array(std)
    return None, None

def predict_with_ml(beat: np.ndarray, model_key: str):
    """Run inference using a traditional ML model and return (pred_idx, confidence, probs_dict)."""
    extractor = ECGFeatureExtractor(sampling_rate=config.SAMPLING_RATE)
    features = extractor.extract_all_features(beat)   # (25,)

    mean, std = load_feature_stats()
    if mean is not None:
        features = (features - mean) / (std + 1e-8)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    model = load_ml_model(model_key)
    if model is None:
        return None, None, None

    pred = int(model.predict(features.reshape(1, -1))[0])
    proba = model.predict_proba(features.reshape(1, -1))[0]  # shape (5,)
    conf  = float(proba[pred])
    probs_dict = {config.CLASS_NAMES[i]: float(proba[i]) for i in range(len(proba))}
    return pred, conf, probs_dict


# ── Groq AI Medical Advisor ────────────────────────────────────────────────

def groq_medical_advice(pred_name: str, confidence: float, needs_ref: bool) -> str:
    """Call Groq Llama3 to generate medical explanation for the prediction."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "Groq API key not found. Add GROQ_API_KEY to your .env file."
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        danger = "CRITICAL - requires IMMEDIATE medical attention" if pred_name == "Ventricular (V)" else \
                 "MODERATE - requires monitoring" if pred_name in ["Supraventricular (S)", "Fusion (F)"] else \
                 "LOW - normal heart rhythm" if pred_name == "Normal (N)" else "UNCERTAIN - needs expert review"
        referral_note = "The system has flagged this for cardiologist referral due to low confidence." if needs_ref else ""
        prompt = f"""You are a medical AI assistant specializing in ECG and cardiology.

A patient's ECG heartbeat has been analyzed:
- Predicted class: {pred_name}
- Confidence: {confidence:.1%}
- Danger level: {danger}
- {referral_note}

Please provide a concise response with these 4 sections:
1. WHAT IS IT: Brief explanation of {pred_name} (1-2 sentences)
2. CAUSE: Common causes of this condition (3-4 bullet points)
3. DANGER: Why this is {danger} and what can happen if untreated
4. REMEDY: Recommended actions and treatments (3-4 bullet points)

Keep it simple and understandable for a non-medical person. Add a disclaimer that this is AI-generated and not a substitute for professional medical advice."""
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate AI advice: {str(e)}"


def send_telegram_report(patient_name: str, pred_name: str, confidence: float,
                         decision: str, beat_idx: int, signal: np.ndarray,
                         true_label_name: str = None):
    """Send ECG report + waveform image to Telegram."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False, "Telegram credentials not found in .env"
    try:
        # Build text report
        true_info = f"\nTrue Label: {true_label_name}" if true_label_name else ""
        correct_icon = "" 
        if true_label_name:
            correct_icon = " [CORRECT]" if true_label_name == pred_name else " [INCORRECT]"
        msg = (
            f"ECG ARRHYTHMIA REPORT\n"
            f"{'='*30}\n"
            f"Patient: {patient_name or 'Anonymous'}\n"
            f"Beat Index: {beat_idx}\n"
            f"{'-'*30}\n"
            f"Predicted: {pred_name}{correct_icon}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Decision: {decision}{true_info}\n"
            f"{'-'*30}\n"
            f"Generated by ECG Arrhythmia Detection System"
        )
        # Send text message
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg},
            timeout=10
        )
        # Save and send ECG image
        fig = plot_ecg_waveform(signal, title=f"ECG Beat {beat_idx} - {pred_name}")
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data={"chat_id": chat_id, "caption": f"{pred_name} | {confidence:.1%} confidence"},
            files={"photo": ("ecg.png", buf, "image/png")},
            timeout=15
        )
        return True, "Report sent successfully!"
    except Exception as e:
        return False, str(e)

def get_decision_status(confidence, needs_referral):
    """Get decision status based on confidence."""
    if needs_referral or confidence < 0.7:
        return "❌ Refer to Cardiologist", "danger"
    elif confidence < 0.85:
        return "⚠️ Monitor", "warning"
    else:
        return "✅ Auto-Classified", "success"

def plot_ecg_waveform(signal, title="ECG Waveform", heatmap=None):
    """Plot ECG waveform with optional Grad-CAM overlay."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    time_ms = np.arange(len(signal)) / config.SAMPLING_RATE * 1000
    
    # Plot ECG signal
    ax.plot(time_ms, signal, 'b-', linewidth=1.5, label='ECG Signal')
    
    # Overlay heatmap if available
    if heatmap is not None:
        # Normalize heatmap for visualization
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Create color overlay
        for i in range(len(signal) - 1):
            intensity = heatmap_norm[i]
            if intensity > 0.3:  # Only show significant regions
                ax.axvspan(time_ms[i], time_ms[i+1], 
                          alpha=intensity * 0.5, 
                          color='red', linewidth=0)
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Amplitude (mV)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<p class="main-header">❤️ ECG Arrhythmia Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-assisted cardiac rhythm analysis powered by deep learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Dashboard", "📤 Upload ECG", "📈 Model Comparison", "📊 About Model"],
            label_visibility="collapsed"
        )

        st.divider()

        # ── Model Selector ──────────────────────────────────────────
        st.header("🤖 Active Model")
        ML_MODELS = ["Random Forest (96.52%)", "XGBoost (95.47%)", "SVM (91.38%)"]
        selected_model = st.selectbox(
            "Choose inference model",
            ML_MODELS + ["Hybrid CNN (72.38%)"],
            index=0,
            help="Random Forest is the most accurate model."
        )
        st.session_state['selected_model'] = selected_model

        # Show accuracy badge
        accuracy_map = {
            "Random Forest (96.52%)": ("96.52%", "🟢"),
            "XGBoost (95.47%)": ("95.47%", "🟢"),
            "SVM (91.38%)": ("91.38%", "🟡"),
            "Hybrid CNN (72.38%)": ("72.38%", "🔴"),
        }
        acc, badge = accuracy_map[selected_model]
        st.markdown(f"{badge} **Test Accuracy: {acc}**")

        st.divider()
        
        # Model Status
        st.header("🔌 System Status")
        try:
            detector = load_detector()
            st.success("✅ Model Loaded")
            st.info(f"Device: {detector.device}")
        except Exception as e:
            st.error(f"❌ Model Error: {str(e)}")
            detector = None
        
        st.divider()
        
        # Class Legend
        st.header("📖 Class Legend")
        classes = [
            ("N", "Normal", "#10B981"),
            ("S", "Supraventricular", "#3B82F6"),
            ("V", "Ventricular", "#EF4444"),
            ("F", "Fusion", "#8B5CF6"),
            ("Q", "Unknown", "#6B7280"),
        ]
        for abbrev, name, color in classes:
            st.markdown(f"<span style='color:{color}'>●</span> **{abbrev}**: {name}", unsafe_allow_html=True)
    
    # Main content based on page selection
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📤 Upload ECG":
        show_upload_page(detector)
    elif page == "📈 Model Comparison":
        show_comparison_page()
    elif page == "📊 About Model":
        show_about_page()

def show_dashboard():
    """Show dashboard page."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Classes", "5", help="AAMI standard classification")
    with col2:
        st.metric("🎯 Expected Accuracy", "~97%", help="On patient-wise test set")
    with col3:
        st.metric("⚡ Sampling Rate", "125 Hz", help="MIT-BIH resampled")
    with col4:
        st.metric("📏 Beat Length", "187", help="Samples per beat")
    
    st.divider()
    
    # Features
    st.header("🚀 System Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔬 5-Class Classification
        Classifies ECG beats according to AAMI EC57 standard:
        - Normal (N)
        - Supraventricular (S)
        - Ventricular (V)
        - Fusion (F)
        - Unknown (Q)
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Confidence-Based Decisions
        - **≥85%**: Auto-classified ✅
        - **70-85%**: Monitor ⚠️
        - **<70%**: Refer to cardiologist ❌
        
        Low-confidence cases are flagged for human review.
        """)
    
    with col3:
        st.markdown("""
        ### 🧠 Explainable AI
        - Grad-CAM visualizations
        - Highlights important ECG regions
        - Builds trust in predictions
        - Aids clinical validation
        """)

    st.divider()
    st.header("Quick Start")
    st.info("""
    1. Navigate to **Upload ECG** in the sidebar
    2. Upload a CSV file containing ECG beat data
    3. Select a beat index and click Analyze ECG
    """)


def show_upload_page(detector):
    """Show ECG upload and analysis page."""
    st.header("Upload ECG Data")

    selected_model = st.session_state.get('selected_model', 'Random Forest (96.52%)')
    st.info(f"Active model: **{selected_model}**")

    # Patient name input
    patient_name = st.text_input("Patient Name / ID", placeholder="e.g. John Doe or P-001")

    uploaded_file = st.file_uploader(
        "Upload ECG CSV file (MIT-BIH format)",
        type=['csv'],
        help="Each row should contain 187 samples (or 188 with label)"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            st.success(f"File loaded: {df.shape[0]} beats, {df.shape[1]} columns")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Beats", df.shape[0])
            with col2:
                st.metric("Samples/Beat", df.shape[1])
            with col3:
                has_label = df.shape[1] == 188
                st.metric("Has Labels", "Yes" if has_label else "No")

            st.divider()
            st.subheader("Select Beat to Analyze")
            col1, col2 = st.columns([3, 1])
            with col1:
                beat_idx = st.slider("Beat Index", 0, df.shape[0] - 1, 0)
            with col2:
                st.metric("Selected Beat", beat_idx)

            row = df.iloc[beat_idx].values
            if len(row) == 188:
                signal = row[:187].astype(float)
                true_label = int(row[187])
            else:
                signal = row[:187].astype(float)
                true_label = None

            st.subheader("ECG Waveform")
            fig = plot_ecg_waveform(signal, title=f"Beat {beat_idx}")
            st.pyplot(fig)
            plt.close(fig)

            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze = st.button("Analyze ECG", type="primary", use_container_width=True)

            if analyze:
                use_cnn = (selected_model == "Hybrid CNN (72.38%)")
                with st.spinner("Analyzing ECG beat..."):
                    if use_cnn:
                        if detector is None:
                            st.error("CNN model not loaded.")
                            return
                        result = detector.predict(signal, generate_explanation=True)
                        prediction = result.prediction
                        pred_name  = result.prediction_name
                        confidence = result.confidence
                        probs_dict = result.probabilities
                        needs_ref  = result.needs_referral
                        ref_reason = result.referral_reason
                        heatmap    = result.heatmap
                    else:
                        pred_idx, conf, probs_dict = predict_with_ml(signal, selected_model)
                        if pred_idx is None:
                            st.error(f"Could not load model: {selected_model}. Ensure models/ml_models/ contains the .joblib files.")
                            return
                        prediction = pred_idx
                        pred_name  = config.CLASS_NAMES[pred_idx]
                        confidence = conf
                        needs_ref  = confidence < config.REFERRAL_THRESHOLD
                        ref_reason = f"Low confidence ({confidence:.1%} < {config.REFERRAL_THRESHOLD:.0%})"
                        heatmap    = None

                # Save to session_state so Telegram button works after re-render
                decision, _ = get_decision_status(confidence, needs_ref)
                st.session_state['last_prediction'] = {
                    'pred_name':  pred_name,
                    'confidence': confidence,
                    'needs_ref':  needs_ref,
                    'decision':   decision,
                    'beat_idx':   beat_idx,
                    'signal':     signal.tolist(),
                    'true_label': true_label,
                    'patient_name': patient_name,
                }

                st.divider()
                st.header("AI Prediction Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Class", pred_name)
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                with col3:
                    decision, dtype = get_decision_status(confidence, needs_ref)
                    if dtype == "success":
                        st.success(decision)
                    elif dtype == "warning":
                        st.warning(decision)
                    else:
                        st.error(decision)

                if true_label is not None:
                    if prediction == true_label:
                        st.success(f"Correct! True label: {config.CLASS_NAMES[true_label]}")
                    else:
                        st.error(f"Incorrect. True label: {config.CLASS_NAMES[true_label]}")

                st.subheader("Class Probabilities")
                probs_df = pd.DataFrame({
                    'Class': list(probs_dict.keys()),
                    'Probability': list(probs_dict.values())
                }).sort_values('Probability', ascending=False)

                fig, ax = plt.subplots(figsize=(10, 4))
                colors = ['#1E40AF' if c == pred_name else '#94A3B8' for c in probs_df['Class']]
                bars = ax.barh(probs_df['Class'], probs_df['Probability'], color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability', fontsize=12)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                for bar, prob in zip(bars, probs_df['Probability']):
                    ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                            f'{prob:.1%}', va='center', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                if heatmap is not None:
                    st.subheader("Model Explanation (Grad-CAM)")
                    st.info("Highlighted regions show parts of the ECG that most influenced the model's decision.")
                    fig = plot_ecg_waveform(signal, title="Grad-CAM Overlay", heatmap=heatmap)
                    st.pyplot(fig)
                    plt.close(fig)
                elif not use_cnn:
                    st.info("Grad-CAM is only available for the Hybrid CNN model. Switch model in the sidebar to see it.")

                if needs_ref:
                    st.divider()
                    st.error(f"Referral Recommended — {ref_reason}")

                # ── Groq AI Medical Advisor ──
                st.divider()
                with st.expander("AI Medical Advisor (Powered by Groq Llama3)", expanded=True):
                    with st.spinner("Generating medical explanation..."):
                        advice = groq_medical_advice(pred_name, confidence, needs_ref)
                    st.markdown(advice)

            # ── Telegram Report Button (outside if analyze: so it persists) ──
            if st.session_state.get('last_prediction'):
                p = st.session_state['last_prediction']
                st.divider()
                st.markdown(f"**Last prediction:** {p['pred_name']} ({p['confidence']:.1%}) — Beat {p['beat_idx']}")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("Send Report to Telegram", use_container_width=True):
                        with st.spinner("Sending to Telegram..."):
                            ok, msg_result = send_telegram_report(
                                patient_name=p.get('patient_name', ''),
                                pred_name=p['pred_name'],
                                confidence=p['confidence'],
                                decision=p['decision'],
                                beat_idx=p['beat_idx'],
                                signal=np.array(p['signal']),
                                true_label_name=config.CLASS_NAMES[p['true_label']] if p.get('true_label') is not None else None
                            )
                        if ok:
                            st.success("Report sent! Check your Telegram (@kasi_dev).")
                        else:
                            st.error(f"Failed: {msg_result}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
    else:
        st.info("No file uploaded yet. Upload a CSV where each row is one ECG beat (187 samples).")

def show_comparison_page():
    """Show the model comparison page with generated visualizations."""
    st.header("📈 Model Comparison Results")
    st.write("This section compares the performance of our 4 different machine learning models evaluated on the test set using Patient-Wise Splitting to prevent data leakage.")
    
    import os
    from PIL import Image
    comp_dir = config.RESULTS_DIR / "model_comparison"
    
    if not comp_dir.exists() or not list(comp_dir.glob("*.png")):
        st.warning("⚠️ No comparison visualizations found. Please run `python compare_models.py` first to generate the charts.")
        return
        
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Accuracy", "Confusion Matrices", "Radar Chart"])
    
    with tab1:
        st.subheader("Comprehensive Dashboard")
        st.write("Overview of test accuracy, F1 scores, and training times across all models.")
        dash_path = comp_dir / "summary_dashboard.png"
        if dash_path.exists():
            st.image(Image.open(dash_path), use_column_width=True)
            
    with tab2:
        st.subheader("Test Accuracy Comparison")
        acc_path = comp_dir / "accuracy_comparison.png"
        if acc_path.exists():
            st.image(Image.open(acc_path), use_column_width=True)
            
    with tab3:
        st.subheader("Confusion Matrices")
        st.write("Shows how each model performs on specific arrhythmia classes. A darker diagonal means better performance.")
        cm_path = comp_dir / "confusion_matrices.png"
        if cm_path.exists():
            st.image(Image.open(cm_path), use_column_width=True)
            
    with tab4:
        st.subheader("Multi-Metric Radar Chart")
        radar_path = comp_dir / "radar_chart.png"
        if radar_path.exists():
            st.image(Image.open(radar_path), use_column_width=True)

def show_about_page():
    """Show about/model information page."""
    
    st.header("📊 About the Model")
    
    # Architecture
    st.subheader("🏗️ Model Architecture")
    st.markdown("""
    The system uses a **Hybrid Fusion Architecture** that combines:
    
    1. **1D CNN Backbone** (32→64→128→256 filters)
       - Learns morphological patterns from raw ECG
       - 4 convolutional blocks with batch normalization
    
    2. **Engineered Features** (25 features)
       - Morphological: R-peak amplitude, QRS duration, etc.
       - Statistical: Mean, std, skewness, kurtosis
       - Frequency: Spectral centroid, dominant frequency
       - Wavelet: Energy at multiple decomposition levels
       - Nonlinear: Sample entropy, zero crossings
    
    3. **Fusion Layer**
       - Combines CNN features with engineered features
       - Dense layers for final classification
    
    4. **Temperature Scaling**
       - Calibrates confidence scores
       - Ensures reliable uncertainty estimates
    """)
    
    st.divider()
    
    # Technical specs
    st.subheader("⚙️ Technical Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Sampling Rate | 125 Hz |
        | Beat Length | 187 samples |
        | Duration | ~1.5 seconds |
        | Classes | 5 (AAMI standard) |
        """)
    
    with col2:
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | CNN Filters | 32→64→128→256 |
        | Engineered Features | 25 |
        | Referral Threshold | 70% |
        | Dropout Rate | 30% |
        """)
    
    st.divider()
    
    # Performance
    st.subheader("📈 Expected Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Accuracy", "~97%")
    with col2:
        st.metric("Weighted F1", "~0.95")
    with col3:
        st.metric("Normal (N) F1", "~0.99")
    with col4:
        st.metric("Ventricular (V) F1", "~0.95")
    
    st.caption("*Performance measured on patient-wise split test set to prevent data leakage*")
    
    st.divider()
    
    # Disclaimer
    st.warning("""
    ⚠️ **Important Disclaimer**
    
    This system is designed for **research and educational purposes only**. It is NOT a certified medical device and should NOT be used for actual clinical diagnosis. Always consult a qualified healthcare professional for medical advice and diagnosis.
    """)

if __name__ == "__main__":
    main()
