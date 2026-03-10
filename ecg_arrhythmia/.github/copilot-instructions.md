# ECG Arrhythmia Detection - Copilot Instructions

## Architecture Overview
This is a **full-stack medical AI system** for ECG arrhythmia classification with:
- **Python backend**: PyTorch hybrid CNN model + FastAPI REST API
- **Next.js frontend**: React 19 + Tailwind v4 web interface
- **Key design principle**: Patient safety via confidence-based referral mechanism

### Core Components
| Module | Purpose |
|--------|---------|
| [model.py](../model.py) | `HybridECGClassifier` - CNN backbone + engineered features fusion |
| [inference.py](../inference.py) | `ECGArrhythmiaDetector` class with referral logic |
| [api.py](../api.py) | FastAPI endpoints at `localhost:8000` |
| [frontend/](../frontend/) | Next.js app consuming the API |

### Data Flow
```
ECG Beat (187 samples) → Feature Extraction (25 features)
                       ↘
                         Hybrid Fusion → Temperature Scaling → Confidence Check
                       ↗                                      ↓
Raw Signal → 1D CNN →                        If <70%: Flag for human referral
```

## Critical Conventions

### Patient-Wise Data Splitting (CRITICAL)
**Never mix patient data across train/val/test splits.** The codebase simulates patient IDs in [data_loader.py](../data_loader.py#L55) to prevent data leakage. Any new data handling must preserve this pattern:
```python
# Example from patient_wise_split()
train_patients = set(unique_patients[:n_train])
# Beats from same patient stay together
```

### Confidence-Based Referral
All predictions must respect the referral threshold (`config.REFERRAL_THRESHOLD = 0.7`). Low-confidence predictions are flagged:
```python
needs_referral = confidence < self.referral_threshold
```

### Model Architecture Pattern
The `HybridECGClassifier` always expects both inputs:
- `waveform`: Shape `(batch, 1, 187)` - raw ECG signal
- `engineered_features`: Shape `(batch, 25)` - extracted from `ECGFeatureExtractor`

## Development Commands

### Backend (from `ecg_arrhythmia/` directory)
```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Train model
python main.py --mode train

# Start API server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend (from `ecg_arrhythmia/frontend/` directory)
```powershell
npm run dev          # Development server at localhost:3000
npm run build        # Production build
```
Set `NEXT_PUBLIC_API_URL` environment variable to point to backend.

## Key Files Reference

### Configuration
All constants are centralized in [config.py](../config.py):
- `BEAT_LENGTH = 187` - samples per beat
- `NUM_ENGINEERED_FEATURES = 25` - feature count
- `REFERRAL_THRESHOLD = 0.7` - confidence cutoff
- `CLASS_NAMES` - 5-class AAMI standard labels

### Feature Engineering
[feature_engineering.py](../feature_engineering.py) extracts 25 features:
- Morphological (7): R-peak, QRS duration, amplitudes
- Statistical (6): mean, std, skewness, kurtosis
- Frequency (4): FFT-based spectral features
- Wavelet (6): DWT decomposition
- Nonlinear (2): entropy, zero crossings

### API Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | POST | Single beat prediction |
| `/predict_with_explanation` | POST | Prediction + Grad-CAM heatmap |
| `/predict_batch` | POST | Batch processing with summary |
| `/class_info`, `/model_info` | GET | Metadata |

## Patterns to Follow

### Adding New API Endpoints
Follow the pattern in [api.py](../api.py) using Pydantic models:
```python
class NewInput(BaseModel):
    signal: List[float] = Field(..., description="...")

@app.post("/new_endpoint", response_model=ResponseModel)
async def new_endpoint(input: NewInput):
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Validate input length against config.BEAT_LENGTH
```

### Frontend API Calls
Use [frontend/src/lib/api.js](../frontend/src/lib/api.js) for backend communication. All functions use `API_BASE` from environment.

### Model Checkpoints
Saved to `models/` with structure:
```python
{
    'model_state_dict': ...,
    'temperature': ...,       # Calibration temperature
    'feature_mean': ...,      # For normalization
    'feature_std': ...
}
```

## Testing & Validation
- Dataset: Kaggle ECG Heartbeat (MIT-BIH) in `data/` directory
- Expected accuracy: ~97%, F1: ~0.95
- Run `python main.py --mode evaluate` to generate metrics in `results/`
