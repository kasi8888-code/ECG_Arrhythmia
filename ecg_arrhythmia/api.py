"""
FastAPI REST API for ECG Arrhythmia Detection
Provides endpoints for real-time prediction with explanations
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import numpy as np
import torch
from pathlib import Path
import io
import base64
import matplotlib.pyplot as plt
import tempfile

import config
from inference import ECGArrhythmiaDetector
from explainability import plot_gradcam_explanation

# Initialize FastAPI app
app = FastAPI(
    title="ECG Arrhythmia Detection API",
    description="""
    Patient-safe ECG arrhythmia detection with:
    - 1D CNN Classifier + Engineered Features
    - Calibrated confidence scores
    - Confidence-based referral mechanism
    - Signal Attention Heatmap for every prediction
    """,
    version="1.0.0"
)

# Global detector instance
detector: Optional[ECGArrhythmiaDetector] = None


# Request/Response Models
class BeatInput(BaseModel):
    """Single ECG beat input."""
    signal: List[float] = Field(..., description="ECG beat signal (187 samples)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "signal": [0.1, 0.2, 0.3] + [0.0] * 184  # 187 samples
            }
        }


class BatchBeatInput(BaseModel):
    """Batch of ECG beats."""
    signals: List[List[float]] = Field(..., description="List of ECG beats")


class PredictionResponse(BaseModel):
    """Prediction result."""
    prediction: int
    prediction_name: str
    confidence: float
    probabilities: Dict[str, float]
    needs_referral: bool
    referral_reason: Optional[str]
    key_regions_ms: Optional[List[float]]


class ExplanationResponse(BaseModel):
    """Prediction with explanation."""
    prediction: int
    prediction_name: str
    confidence: float
    probabilities: Dict[str, float]
    needs_referral: bool
    referral_reason: Optional[str]
    heatmap: List[float]
    key_regions: List[Dict]
    visualization_base64: Optional[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global detector
    
    model_path = config.MODEL_DIR / 'final_model.pt'
    
    if model_path.exists():
        detector = ECGArrhythmiaDetector(model_path=model_path)
        print(f"Model loaded from {model_path}")
    else:
        # Initialize without weights (for testing)
        detector = ECGArrhythmiaDetector()
        print("Warning: Model initialized without trained weights")


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=detector is not None,
        device=str(detector.device) if detector else "none"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return await root()


@app.post("/predict", response_model=PredictionResponse)
async def predict(beat_input: BeatInput):
    """
    Predict arrhythmia class for a single ECG beat.
    
    - **signal**: ECG beat signal (187 samples at 125Hz)
    
    Returns prediction, confidence, and referral status.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate input length
    if len(beat_input.signal) != config.BEAT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {config.BEAT_LENGTH} samples, got {len(beat_input.signal)}"
        )
    
    # Convert to numpy
    beat = np.array(beat_input.signal, dtype=np.float32)
    
    # Make prediction
    result = detector.predict(beat, generate_explanation=False)
    
    return PredictionResponse(
        prediction=result.prediction,
        prediction_name=result.prediction_name,
        confidence=result.confidence,
        probabilities=result.probabilities,
        needs_referral=result.needs_referral,
        referral_reason=result.referral_reason,
        key_regions_ms=None
    )


@app.post("/predict_with_explanation", response_model=ExplanationResponse)
async def predict_with_explanation(beat_input: BeatInput, include_visualization: bool = True):
    """
    Predict with Signal Attention Heatmap explanation.
    
    - **signal**: ECG beat signal (187 samples)
    - **include_visualization**: Whether to include base64-encoded plot
    
    Returns prediction, confidence, heatmap, and optionally a visualization.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate input
    if len(beat_input.signal) != config.BEAT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {config.BEAT_LENGTH} samples, got {len(beat_input.signal)}"
        )
    
    beat = np.array(beat_input.signal, dtype=np.float32)
    
    # Make prediction with explanation
    result = detector.predict(beat, generate_explanation=True)
    
    # Generate visualization if requested
    visualization_b64 = None
    if include_visualization and result.heatmap is not None:
        fig = plot_gradcam_explanation(
            waveform=beat,
            heatmap=result.heatmap,
            prediction=result.prediction,
            confidence=result.confidence
        )
        
        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        visualization_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
    
    return ExplanationResponse(
        prediction=result.prediction,
        prediction_name=result.prediction_name,
        confidence=result.confidence,
        probabilities=result.probabilities,
        needs_referral=result.needs_referral,
        referral_reason=result.referral_reason,
        heatmap=result.heatmap.tolist() if result.heatmap is not None else [],
        key_regions=result.explanation.get('key_regions', []) if result.explanation else [],
        visualization_base64=visualization_b64
    )


@app.post("/predict_batch")
async def predict_batch(batch_input: BatchBeatInput):
    """
    Predict for multiple beats.
    
    - **signals**: List of ECG beat signals
    
    Returns list of predictions with referral summary.
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate inputs
    for i, signal in enumerate(batch_input.signals):
        if len(signal) != config.BEAT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Beat {i}: Expected {config.BEAT_LENGTH} samples, got {len(signal)}"
            )
    
    beats = np.array(batch_input.signals, dtype=np.float32)
    
    # Make predictions
    results = detector.predict_batch(beats, generate_explanations=False)
    
    # Prepare response
    predictions = []
    for result in results:
        predictions.append({
            'prediction': result.prediction,
            'prediction_name': result.prediction_name,
            'confidence': result.confidence,
            'needs_referral': result.needs_referral
        })
    
    # Summary statistics
    n_total = len(results)
    n_referred = sum(1 for r in results if r.needs_referral)
    confidences = [r.confidence for r in results]
    
    return {
        'predictions': predictions,
        'summary': {
            'total_beats': n_total,
            'referred_count': n_referred,
            'referral_rate': n_referred / n_total if n_total > 0 else 0,
            'mean_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
    }


@app.get("/class_info")
async def get_class_info():
    """Get information about arrhythmia classes."""
    return {
        'classes': [
            {
                'id': i,
                'name': name,
                'abbreviation': name.split('(')[1].replace(')', '') if '(' in name else name[0]
            }
            for i, name in enumerate(config.CLASS_NAMES)
        ],
        'referral_threshold': config.REFERRAL_THRESHOLD
    }


@app.get("/model_info")
async def get_model_info():
    """Get model configuration information."""
    return {
        'sampling_rate': config.SAMPLING_RATE,
        'beat_length': config.BEAT_LENGTH,
        'num_classes': config.NUM_CLASSES,
        'class_names': config.CLASS_NAMES,
        'referral_threshold': config.REFERRAL_THRESHOLD,
        'model_type': '1D CNN Classifier + Engineered Features',
        'features': {
            'cnn_filters': config.CNN_FILTERS,
            'num_engineered_features': config.NUM_ENGINEERED_FEATURES,
            'fusion_hidden_dim': config.FUSION_HIDDEN_DIM
        }
    }


# Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
