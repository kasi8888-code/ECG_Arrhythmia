"""
Inference Module for ECG Arrhythmia Detection
Includes confidence-based referral mechanism and explainability
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import json

import config
from model import HybridECGClassifier, TemperatureScaling
from feature_engineering import ECGFeatureExtractor
from explainability import GradCAM, plot_gradcam_explanation, generate_explanation_report


@dataclass
class PredictionResult:
    """Structured prediction result."""
    prediction: int
    prediction_name: str
    confidence: float
    probabilities: Dict[str, float]
    needs_referral: bool
    referral_reason: Optional[str]
    heatmap: Optional[np.ndarray]
    explanation: Optional[Dict]


class ECGArrhythmiaDetector:
    """
    Main inference class for ECG arrhythmia detection.
    
    Provides:
    - Beat classification with calibrated confidence
    - Automatic referral for low-confidence predictions
    - Grad-CAM explanations for every prediction
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
        referral_threshold: float = None,
        calibration_temperature: float = None
    ):
        """
        Initialize detector.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use (auto-detect if None)
            referral_threshold: Confidence threshold for referral
            calibration_temperature: Temperature for confidence calibration
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Thresholds
        self.referral_threshold = referral_threshold or config.REFERRAL_THRESHOLD
        self.temperature = calibration_temperature or config.CALIBRATION_TEMPERATURE
        
        # Initialize model
        self.model = HybridECGClassifier(
            input_length=config.BEAT_LENGTH,
            num_engineered_features=config.NUM_ENGINEERED_FEATURES,
            num_classes=config.NUM_CLASSES
        )
        
        # Load trained weights if provided
        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            print("Warning: No model weights loaded. Using random initialization.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize feature extractor
        self.feature_extractor = ECGFeatureExtractor(sampling_rate=config.SAMPLING_RATE)
        
        # Initialize Grad-CAM
        self.gradcam = GradCAM(self.model, target_layer_idx=-1)
        
        # Normalization parameters (should be loaded from training)
        self.feature_mean = None
        self.feature_std = None
    
    def load_model(self, path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load calibration temperature if available
        if 'temperature' in checkpoint:
            self.temperature = checkpoint['temperature']
        
        # Load feature normalization parameters if available
        if 'feature_mean' in checkpoint:
            self.feature_mean = checkpoint['feature_mean']
            self.feature_std = checkpoint['feature_std']
        
        print(f"Model loaded from {path}")
        print(f"Calibration temperature: {self.temperature}")
    
    def preprocess_beat(self, beat: np.ndarray) -> tuple:
        """
        Preprocess a single beat for inference.
        
        Args:
            beat: Raw beat signal (length,)
        
        Returns:
            waveform_tensor, features_tensor
        """
        # Ensure correct length
        if len(beat) != config.BEAT_LENGTH:
            # Resample if needed
            beat = np.interp(
                np.linspace(0, 1, config.BEAT_LENGTH),
                np.linspace(0, 1, len(beat)),
                beat
            )
        
        # Normalize waveform (z-score)
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
        
        # Extract engineered features
        features = self.feature_extractor.extract_all_features(beat)
        
        # Normalize features if parameters available
        if self.feature_mean is not None:
            features = (features - self.feature_mean) / (self.feature_std + 1e-8)
        
        # Convert to tensors
        waveform_tensor = torch.FloatTensor(beat).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)  # (1, F)
        
        return waveform_tensor, features_tensor
    
    def predict(
        self,
        beat: np.ndarray,
        generate_explanation: bool = True
    ) -> PredictionResult:
        """
        Make prediction for a single beat with confidence and explanation.
        
        Args:
            beat: Raw beat signal (length,)
            generate_explanation: Whether to generate Grad-CAM explanation
        
        Returns:
            PredictionResult with prediction, confidence, and explanation
        """
        # Preprocess
        waveform, features = self.preprocess_beat(beat)
        waveform = waveform.to(self.device)
        features = features.to(self.device)
        
        # Get prediction with calibrated confidence
        with torch.no_grad():
            output = self.model.predict_with_confidence(
                waveform, features, temperature=self.temperature
            )
        
        prediction = output['predictions'].item()
        confidence = output['confidence'].item()
        probabilities = output['probabilities'].cpu().numpy()[0]
        
        # Check referral
        needs_referral = confidence < self.referral_threshold
        referral_reason = None
        
        if needs_referral:
            # Determine referral reason
            sorted_probs = np.sort(probabilities)[::-1]
            if sorted_probs[0] - sorted_probs[1] < 0.2:
                referral_reason = "Ambiguous prediction (close class probabilities)"
            else:
                referral_reason = f"Low confidence ({confidence:.1%} < {self.referral_threshold:.1%})"
        
        # Generate explanation
        heatmap = None
        explanation = None
        
        if generate_explanation:
            # Re-enable gradients for Grad-CAM
            waveform = waveform.requires_grad_(True)
            heatmap, _, _ = self.gradcam.generate_heatmap(waveform, features)
            
            explanation = {
                'heatmap': heatmap,
                'key_regions': self._identify_key_regions(heatmap),
                'feature_importance': self._get_feature_importance()
            }
        
        return PredictionResult(
            prediction=prediction,
            prediction_name=config.CLASS_NAMES[prediction],
            confidence=confidence,
            probabilities={
                config.CLASS_NAMES[i]: float(probabilities[i])
                for i in range(len(probabilities))
            },
            needs_referral=needs_referral,
            referral_reason=referral_reason,
            heatmap=heatmap,
            explanation=explanation
        )
    
    def predict_batch(
        self,
        beats: np.ndarray,
        generate_explanations: bool = False
    ) -> List[PredictionResult]:
        """
        Make predictions for multiple beats.
        
        Args:
            beats: Array of beats (n_samples, length)
            generate_explanations: Whether to generate Grad-CAM for each
        
        Returns:
            List of PredictionResults
        """
        results = []
        
        for i, beat in enumerate(beats):
            if (i + 1) % 100 == 0:
                print(f"Processing beat {i+1}/{len(beats)}")
            
            result = self.predict(beat, generate_explanation=generate_explanations)
            results.append(result)
        
        return results
    
    def _identify_key_regions(self, heatmap: np.ndarray, threshold: float = 0.7) -> List[Dict]:
        """Identify key regions from heatmap."""
        key_regions = []
        
        # Find contiguous high-attention regions
        above_threshold = heatmap > threshold
        changes = np.diff(above_threshold.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        
        if above_threshold[0]:
            starts = np.concatenate([[0], starts])
        if above_threshold[-1]:
            ends = np.concatenate([ends, [len(above_threshold)]])
        
        for start, end in zip(starts, ends):
            # Convert to time (ms)
            start_ms = start / config.SAMPLING_RATE * 1000
            end_ms = end / config.SAMPLING_RATE * 1000
            
            key_regions.append({
                'start_sample': int(start),
                'end_sample': int(end),
                'start_ms': float(start_ms),
                'end_ms': float(end_ms),
                'mean_attention': float(np.mean(heatmap[start:end]))
            })
        
        return key_regions
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (placeholder for future implementation)."""
        # This could be implemented using gradient-based feature importance
        # or SHAP values in a future version
        return {}
    
    def generate_report(
        self,
        beat: np.ndarray,
        true_label: Optional[int] = None,
        save_path: Optional[Path] = None
    ) -> Dict:
        """
        Generate comprehensive prediction report with visualization.
        
        Args:
            beat: Raw beat signal
            true_label: Ground truth label (optional)
            save_path: Path to save visualization
        
        Returns:
            Full report dictionary
        """
        result = self.predict(beat, generate_explanation=True)
        
        report = {
            'prediction': result.prediction,
            'prediction_name': result.prediction_name,
            'confidence': result.confidence,
            'probabilities': result.probabilities,
            'needs_referral': result.needs_referral,
            'referral_reason': result.referral_reason,
            'key_regions': result.explanation['key_regions'] if result.explanation else [],
            'true_label': true_label,
            'true_label_name': config.CLASS_NAMES[true_label] if true_label is not None else None,
            'correct': result.prediction == true_label if true_label is not None else None
        }
        
        # Generate visualization if heatmap available
        if result.heatmap is not None and save_path:
            fig = plot_gradcam_explanation(
                waveform=beat,
                heatmap=result.heatmap,
                prediction=result.prediction,
                confidence=result.confidence,
                true_label=true_label,
                save_path=str(save_path)
            )
            report['visualization_path'] = str(save_path)
        
        return report


def analyze_referrals(results: List[PredictionResult], labels: Optional[np.ndarray] = None) -> Dict:
    """
    Analyze referral patterns and their effectiveness.
    
    Args:
        results: List of prediction results
        labels: Ground truth labels (optional)
    
    Returns:
        Referral analysis statistics
    """
    n_total = len(results)
    n_referred = sum(1 for r in results if r.needs_referral)
    n_not_referred = n_total - n_referred
    
    analysis = {
        'total_predictions': n_total,
        'referred_count': n_referred,
        'not_referred_count': n_not_referred,
        'referral_rate': n_referred / n_total if n_total > 0 else 0,
    }
    
    if labels is not None:
        # Analyze accuracy by referral status
        referred_mask = np.array([r.needs_referral for r in results])
        predictions = np.array([r.prediction for r in results])
        
        # Accuracy on non-referred
        if n_not_referred > 0:
            non_referred_acc = np.mean(predictions[~referred_mask] == labels[~referred_mask])
            analysis['non_referred_accuracy'] = float(non_referred_acc)
        
        # Would-be accuracy on referred (if we had trusted the model)
        if n_referred > 0:
            referred_acc = np.mean(predictions[referred_mask] == labels[referred_mask])
            analysis['referred_accuracy'] = float(referred_acc)
        
        # Error reduction by referral
        # (Errors avoided by referring uncertain predictions)
        if n_referred > 0:
            referred_errors = np.sum(predictions[referred_mask] != labels[referred_mask])
            analysis['referred_errors'] = int(referred_errors)
            analysis['error_rate_in_referred'] = float(referred_errors / n_referred)
    
    # Referral reasons breakdown
    reasons = {}
    for r in results:
        if r.needs_referral and r.referral_reason:
            reason = r.referral_reason.split(' (')[0]  # Simplify reason
            reasons[reason] = reasons.get(reason, 0) + 1
    analysis['referral_reasons'] = reasons
    
    # Confidence statistics
    confidences = np.array([r.confidence for r in results])
    analysis['confidence_stats'] = {
        'mean': float(np.mean(confidences)),
        'std': float(np.std(confidences)),
        'min': float(np.min(confidences)),
        'max': float(np.max(confidences)),
        'median': float(np.median(confidences))
    }
    
    return analysis


if __name__ == "__main__":
    # Test inference
    print("Testing ECGArrhythmiaDetector...")
    
    # Initialize detector (without trained model for testing)
    detector = ECGArrhythmiaDetector()
    
    # Create synthetic beat
    t = np.linspace(0, 1, 187)
    synthetic_beat = np.sin(2 * np.pi * 5 * t) + 0.5 * np.exp(-((t - 0.5)**2) / 0.01)
    
    # Make prediction
    result = detector.predict(synthetic_beat, generate_explanation=True)
    
    print(f"\nPrediction Result:")
    print(f"  Class: {result.prediction_name}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Needs Referral: {result.needs_referral}")
    print(f"  Referral Reason: {result.referral_reason}")
    print(f"  Probabilities: {result.probabilities}")
    
    if result.explanation:
        print(f"  Key Regions: {len(result.explanation['key_regions'])} identified")
