"""
Streamlit Web Application for ECG Arrhythmia Detection System
AI-assisted cardiac rhythm analysis with explainability

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import io
import base64
import matplotlib.pyplot as plt

# Import project modules
import config
from inference import ECGArrhythmiaDetector
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

# Cache the detector to avoid reloading
@st.cache_resource
def load_detector():
    """Load the ECG Arrhythmia Detector with caching."""
    # Try both model paths
    model_paths = [
        config.MODEL_DIR / 'final_model.pt',
        config.MODEL_DIR / 'best_model.pt'
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            return ECGArrhythmiaDetector(model_path=model_path)
    
    # If no trained model, return detector without weights (for demo)
    st.warning("⚠️ No trained model found. Using uninitialized model for demo.")
    return ECGArrhythmiaDetector()

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
    
    # Quick Start
    st.header("🎯 Quick Start")
    st.info("""
    1. Navigate to **📤 Upload ECG** in the sidebar
    2. Upload a CSV file containing ECG beat data
    3. Select a beat to analyze
    4. View the AI prediction with Grad-CAM explanation
    """)

def show_upload_page(detector):
    """Show ECG upload and analysis page."""
    
    st.header("📤 Upload ECG Data")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload ECG CSV file (MIT-BIH format)",
        type=['csv'],
        help="Each row should contain 187 samples (or 188 with label)"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file, header=None)
            st.success(f"✅ File loaded: {df.shape[0]} beats, {df.shape[1]} columns")
            
            # File info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Beats", df.shape[0])
            with col2:
                st.metric("Samples/Beat", df.shape[1])
            with col3:
                has_label = df.shape[1] == 188
                st.metric("Has Labels", "Yes" if has_label else "No")
            
            st.divider()
            
            # Beat selector
            st.subheader("🎯 Select Beat to Analyze")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                beat_idx = st.slider("Beat Index", 0, df.shape[0] - 1, 0)
            with col2:
                st.metric("Selected Beat", beat_idx)
            
            # Get beat signal
            row = df.iloc[beat_idx].values
            if len(row) == 188:
                signal = row[:187]
                true_label = int(row[187])
            else:
                signal = row[:187]
                true_label = None
            
            # Visualize ECG
            st.subheader("📊 ECG Waveform")
            fig = plot_ecg_waveform(signal, title=f"Beat {beat_idx}")
            st.pyplot(fig)
            plt.close(fig)
            
            st.divider()
            
            # Analyze button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze = st.button("🔍 Analyze ECG", type="primary", use_container_width=True)
            
            if analyze:
                if detector is None:
                    st.error("Model not loaded. Cannot perform analysis.")
                    return
                
                with st.spinner("🧠 Analyzing ECG beat..."):
                    # Make prediction with explanation
                    result = detector.predict(signal, generate_explanation=True)
                
                st.divider()
                
                # Show results
                st.header("🎯 AI Prediction Results")
                
                # Main prediction metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Predicted Class",
                        result.prediction_name,
                        help="AAMI classification"
                    )
                
                with col2:
                    st.metric(
                        "Confidence",
                        f"{result.confidence:.1%}",
                        help="Calibrated confidence score"
                    )
                
                with col3:
                    decision, decision_type = get_decision_status(result.confidence, result.needs_referral)
                    if decision_type == "success":
                        st.success(decision)
                    elif decision_type == "warning":
                        st.warning(decision)
                    else:
                        st.error(decision)
                
                # Show true label if available
                if true_label is not None:
                    correct = result.prediction == true_label
                    if correct:
                        st.success(f"✅ Correct! True label: {config.CLASS_NAMES[true_label]}")
                    else:
                        st.error(f"❌ Incorrect. True label: {config.CLASS_NAMES[true_label]}")
                
                # Class probabilities
                st.subheader("📊 Class Probabilities")
                probs_df = pd.DataFrame({
                    'Class': list(result.probabilities.keys()),
                    'Probability': list(result.probabilities.values())
                })
                probs_df = probs_df.sort_values('Probability', ascending=False)
                
                # Create horizontal bar chart
                fig, ax = plt.subplots(figsize=(10, 4))
                colors = ['#1E40AF' if c == result.prediction_name else '#94A3B8' for c in probs_df['Class']]
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
                
                # Grad-CAM Explanation
                if result.heatmap is not None:
                    st.subheader("🧠 Model Explanation (Grad-CAM)")
                    
                    st.info("💡 **Highlighted regions** show parts of the ECG that most influenced the model's decision. Warmer colors indicate higher importance.")
                    
                    # Plot with heatmap overlay
                    fig = plot_ecg_waveform(
                        signal, 
                        title="Grad-CAM Overlay", 
                        heatmap=result.heatmap
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Referral info
                if result.needs_referral:
                    st.divider()
                    st.error(f"""
                    ### ⚠️ Referral Recommended
                    
                    **Reason**: {result.referral_reason}
                    
                    This prediction has been flagged for cardiologist review due to low confidence or ambiguous signals.
                    """)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
    
    else:
        # Show sample data info
        st.info("""
        📥 **No file uploaded yet**
        
        Upload a CSV file where each row represents one ECG beat with 187 samples.
        
        **Sample Data**: You can use data from the [Kaggle ECG Heartbeat Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
        """)

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
