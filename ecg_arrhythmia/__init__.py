"""
ECG Arrhythmia Detection System

A patient-safe arrhythmia detection system with:
- Random forest+ Engineered Features
- Calibrated confidence scores
- Confidence-based referral mechanism
- Grad-CAM explainability
"""

from .config import *
from .model import HybridECGClassifier
from .inference import ECGArrhythmiaDetector
from .feature_engineering import ECGFeatureExtractor

__version__ = "1.0.0"
__author__ = "ECG Arrhythmia Detection Team"
