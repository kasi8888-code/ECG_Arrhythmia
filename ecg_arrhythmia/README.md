# ECG Arrhythmia Detection System

A **reproducible, patient-safe ECG arrhythmia detection system** that classifies individual heartbeats with calibrated confidence scores and explainability.

## Key Features

- **1D CNN + Feature Fusion Architecture**: Combines 1D CNN (learned morphological features) with engineered HRV/morphological features
- **Confidence-Based Referral**: Low-confidence predictions are flagged for human review
- **Signal Attention Heatmap**: Visual explanations for every prediction
- **Patient-Wise Splitting**: Ensures no data leakage between train/val/test sets
- **5-Class AAMI Standard Classification**:
  - Normal (N)
  - Supraventricular (S)
  - Ventricular (V)
  - Fusion (F)
  - Unknown (Q)

## Project Structure

```
ecg_arrhythmia/
├── config.py              # Configuration settings
├── data_loader.py         # Data loading and patient-wise splitting
├── feature_engineering.py # HRV and morphological feature extraction
├── model.py               # 1D CNN Classifier model architecture
├── explainability.py      # Signal Attention Heatmap implementation
├── trainer.py             # Training pipeline
├── inference.py           # Inference with referral mechanism
├── visualization.py       # Plotting utilities
├── api.py                 # FastAPI REST API
├── main.py                # Main training script
├── data/                  # Dataset folder
├── models/                # Saved models
└── results/               # Evaluation results and plots
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install wfdb neurokit2 pywavelets
pip install torch torchvision
pip install fastapi uvicorn
```

## Dataset

Download the **Kaggle ECG Heartbeat Categorization Dataset**:
- URL: https://www.kaggle.com/datasets/shayanfazeli/heartbeat
- Place `mitbih_train.csv` and `mitbih_test.csv` in the `data/` folder

## Usage

### Training

```bash
python main.py --mode train
```

This will:
1. Load and split data (patient-wise)
2. Extract engineered features
3. Train the 1D CNN Classifier model
4. Calibrate confidence scores
5. Evaluate and save results

### Evaluation

```bash
python main.py --mode evaluate
```

### Demo (with explanations)

```bash
python main.py --mode demo
```

### REST API

```bash
# Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# API Endpoints:
# GET  /                        - Health check
# POST /predict                 - Single beat prediction
# POST /predict_with_explanation - Prediction with Signal Attention Heatmap
# POST /predict_batch           - Batch predictions
# GET  /class_info              - Class information
# GET  /model_info              - Model configuration
```

## Model Architecture

```
Input Beat (187 samples) ─────────────────────┐
                                              │
    ┌─────────────────────────────────────────┤
    │                                         │
    ▼                                         ▼
┌───────────────────┐                 ┌───────────────────┐
│   1D CNN Backbone │                 │ Feature Extractor │
│  (4 Conv Blocks)  │                 │ (25 Features)     │
│  32→64→128→256    │                 │ Morphological,    │
│                   │                 │ Statistical,      │
│                   │                 │ Wavelet, etc.     │
└────────┬──────────┘                 └────────┬──────────┘
         │                                     │
         │ Global Avg Pool                     │ MLP
         │                                     │
         └──────────────┬──────────────────────┘
                        │
                        ▼
               ┌────────────────┐
               │  Fusion Layer  │
               │   (128 → 64)   │
               └────────┬───────┘
                        │
                        ▼
               ┌────────────────┐
               │  Classifier    │
               │  (5 classes)   │
               └────────┬───────┘
                        │
                        ▼
               ┌────────────────┐
               │  Temperature   │
               │   Scaling      │
               └────────┬───────┘
                        │
                        ▼
         ┌──────────────┴──────────────┐
         │                             │
         ▼                             ▼
┌─────────────────┐           ┌─────────────────┐
│   Prediction    │           │  Signal         │
│   + Confidence  │           │  Attention Map  │
└─────────────────┘           └─────────────────┘
```

## Engineered Features (25 total)

### Morphological (7)
- R-peak amplitude, Q/S amplitudes
- QRS duration, R-wave slopes, Beat area

### Statistical (6)
- Mean, std, skewness, kurtosis, max, min

### Frequency Domain (4)
- Dominant frequency, spectral centroid/spread, total power

### Wavelet (6)
- Energy at 4 decomposition levels, entropy, max coefficient

### Nonlinear (2)
- Sample entropy approximation, zero crossings

## Referral Mechanism

Predictions with confidence below the threshold (default: 70%) are flagged for human review:

```python
if confidence < REFERRAL_THRESHOLD:
    # Flag for cardiologist review
    referral_reason = "Low confidence" or "Ambiguous prediction"
```

## Explainability

Every prediction includes a Signal Attention Heatmap showing which parts of the ECG beat were most important for the classification:

```python
from inference import ECGArrhythmiaDetector

detector = ECGArrhythmiaDetector(model_path="models/final_model.pt")
report = detector.generate_report(beat, true_label=0, save_path="explanation.png")
```

## Patient-Wise Splitting

**Critical for valid performance claims**: The dataset is split by patient ID, not by individual beats. This prevents data leakage where beats from the same patient appear in both training and test sets.

```python
# Beats from the same patient stay together
train_patients = {1, 2, 3, ..., 33}  # 70%
val_patients = {34, 35, 36, 37, 38}   # 15%
test_patients = {39, 40, ..., 48}     # 15%
```

## Expected Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | ~97% |
| Weighted F1 Score | ~0.95 |
| Normal (N) F1 | ~0.99 |
| Ventricular (V) F1 | ~0.95 |
| Referral Rate | ~5-10% |
| Non-Referred Accuracy | ~99% |

## Citation

If using the MIT-BIH dataset:
```
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. 
IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
```

## License

MIT License
