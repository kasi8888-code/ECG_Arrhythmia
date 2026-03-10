"""
Grad-CAM Implementation for ECG Arrhythmia Detection
Provides visual explanations for model predictions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

import config


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for 1D CNN.
    
    Provides visual explanations showing which parts of the ECG beat
    are most important for the model's prediction.
    """
    
    def __init__(self, model: nn.Module, target_layer_idx: int = -1):
        """
        Args:
            model: Trained HybridECGClassifier model
            target_layer_idx: Index of target conv block (-1 for last)
        """
        self.model = model
        self.target_layer_idx = target_layer_idx
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        target_block = self.model.cnn_backbone.conv_blocks[self.target_layer_idx]
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_block.register_forward_hook(forward_hook)
        target_block.register_full_backward_hook(backward_hook)
    
    def generate_heatmap(
        self,
        waveform: torch.Tensor,
        engineered_features: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for a single input.
        
        Args:
            waveform: Input waveform (1, 1, length)
            engineered_features: Engineered features (1, num_features)
            target_class: Class to explain (None = predicted class)
        
        Returns:
            heatmap: Normalized heatmap array (length,)
            predicted_class: Predicted class index
            confidence: Prediction confidence
        """
        self.model.eval()
        
        # Ensure gradients are enabled
        waveform = waveform.requires_grad_(True)
        
        # Forward pass
        output = self.model(waveform, engineered_features)
        logits = output['logits']
        
        # Get prediction
        probs = F.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
        predicted_class = predicted.item()
        
        if target_class is None:
            target_class = predicted_class
        
        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Compute weights (global average pooling of gradients)
        weights = torch.mean(self.gradients, dim=2, keepdim=True)  # (1, C, 1)
        
        # Compute weighted activation map
        cam = torch.sum(weights * self.activations, dim=1)  # (1, L)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Upsample to original input length
        original_length = waveform.shape[-1]
        cam_upsampled = np.interp(
            np.linspace(0, 1, original_length),
            np.linspace(0, 1, len(cam)),
            cam
        )
        
        # Normalize to [0, 1]
        cam_min, cam_max = cam_upsampled.min(), cam_upsampled.max()
        if cam_max - cam_min > 0:
            cam_upsampled = (cam_upsampled - cam_min) / (cam_max - cam_min)
        
        return cam_upsampled, predicted_class, confidence.item()
    
    def generate_batch_heatmaps(
        self,
        waveforms: torch.Tensor,
        engineered_features: Optional[torch.Tensor] = None
    ) -> List[Dict]:
        """
        Generate heatmaps for a batch of inputs.
        
        Args:
            waveforms: Batch of waveforms (batch, 1, length)
            engineered_features: Batch of features (batch, num_features)
        
        Returns:
            List of dictionaries with heatmaps, predictions, and confidence
        """
        results = []
        
        for i in range(waveforms.shape[0]):
            wf = waveforms[i:i+1]
            feat = engineered_features[i:i+1] if engineered_features is not None else None
            
            heatmap, pred, conf = self.generate_heatmap(wf, feat)
            
            results.append({
                'heatmap': heatmap,
                'prediction': pred,
                'confidence': conf
            })
        
        return results


def plot_gradcam_explanation(
    waveform: np.ndarray,
    heatmap: np.ndarray,
    prediction: int,
    confidence: float,
    true_label: Optional[int] = None,
    save_path: Optional[str] = None,
    sampling_rate: int = 125
) -> plt.Figure:
    """
    Plot ECG beat with Grad-CAM heatmap overlay.
    
    Args:
        waveform: ECG beat signal (length,)
        heatmap: Grad-CAM heatmap (length,)
        prediction: Predicted class
        confidence: Prediction confidence
        true_label: Ground truth label (optional)
        save_path: Path to save figure
        sampling_rate: Sampling rate for time axis
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), height_ratios=[2, 1, 1])
    
    time = np.arange(len(waveform)) / sampling_rate * 1000  # Convert to ms
    
    # Plot 1: ECG with heatmap overlay
    ax1 = axes[0]
    ax1.plot(time, waveform, 'b-', linewidth=1.5, label='ECG Signal')
    
    # Create colormap overlay
    colors = plt.cm.jet(heatmap)
    for i in range(len(time) - 1):
        ax1.axvspan(time[i], time[i+1], alpha=0.3, color=colors[i], linewidth=0)
    
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    
    # Title with prediction info
    pred_name = config.CLASS_NAMES[prediction]
    title = f'Prediction: {pred_name} (Confidence: {confidence:.1%})'
    if true_label is not None:
        true_name = config.CLASS_NAMES[true_label]
        correct = '✓' if prediction == true_label else '✗'
        title += f'\nTrue: {true_name} {correct}'
    
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Heatmap bar
    ax2 = axes[1]
    heatmap_2d = heatmap.reshape(1, -1)
    im = ax2.imshow(heatmap_2d, aspect='auto', cmap='jet', extent=[time[0], time[-1], 0, 1])
    ax2.set_ylabel('Importance', fontsize=12)
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_title('Grad-CAM Attention Map', fontsize=12)
    ax2.set_yticks([])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, orientation='vertical', pad=0.02)
    cbar.set_label('Importance', fontsize=10)
    
    # Plot 3: ECG with highlighted regions
    ax3 = axes[2]
    ax3.plot(time, waveform, 'k-', linewidth=1, alpha=0.5)
    
    # Highlight high-attention regions
    threshold = 0.5
    high_attention = heatmap > threshold
    
    # Find contiguous regions
    changes = np.diff(high_attention.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    
    if high_attention[0]:
        starts = np.concatenate([[0], starts])
    if high_attention[-1]:
        ends = np.concatenate([ends, [len(high_attention)]])
    
    for start, end in zip(starts, ends):
        ax3.fill_between(
            time[start:end],
            waveform[start:end],
            alpha=0.7,
            color='red',
            label='High Attention' if start == starts[0] else None
        )
    
    ax3.set_xlabel('Time (ms)', fontsize=12)
    ax3.set_ylabel('Amplitude', fontsize=12)
    ax3.set_title('High Attention Regions (importance > 50%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    if len(starts) > 0:
        ax3.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Grad-CAM plot to {save_path}")
    
    return fig


def generate_explanation_report(
    model: nn.Module,
    waveform: torch.Tensor,
    engineered_features: Optional[torch.Tensor],
    true_label: Optional[int] = None,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Generate comprehensive explanation report for a single prediction.
    
    Args:
        model: Trained model
        waveform: Input waveform (1, 1, length)
        engineered_features: Engineered features (1, num_features)
        true_label: Ground truth label
        device: Device to use
    
    Returns:
        Dictionary with explanation data
    """
    model.to(device)
    waveform = waveform.to(device)
    if engineered_features is not None:
        engineered_features = engineered_features.to(device)
    
    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer_idx=-1)
    heatmap, prediction, confidence = gradcam.generate_heatmap(waveform, engineered_features)
    
    # Determine referral status
    needs_referral = confidence < config.REFERRAL_THRESHOLD
    
    # Identify key regions (top attention areas)
    threshold = 0.7
    key_regions = np.where(heatmap > threshold)[0]
    
    report = {
        'prediction': prediction,
        'prediction_name': config.CLASS_NAMES[prediction],
        'confidence': confidence,
        'probabilities': None,  # Will be filled below
        'needs_referral': needs_referral,
        'referral_reason': 'Low confidence' if needs_referral else None,
        'heatmap': heatmap,
        'key_regions_samples': key_regions.tolist(),
        'key_regions_ms': (key_regions / config.SAMPLING_RATE * 1000).tolist(),
        'true_label': true_label,
        'correct': prediction == true_label if true_label is not None else None
    }
    
    # Get full probabilities
    with torch.no_grad():
        output = model(waveform, engineered_features)
        probs = F.softmax(output['logits'], dim=1).cpu().numpy()[0]
        report['probabilities'] = {
            config.CLASS_NAMES[i]: float(probs[i])
            for i in range(len(probs))
        }
    
    return report


if __name__ == "__main__":
    # Test Grad-CAM
    print("Testing Grad-CAM implementation...")
    
    from model import HybridECGClassifier
    
    # Create model
    model = HybridECGClassifier(
        input_length=187,
        num_engineered_features=25,
        num_classes=5
    )
    
    # Test input
    waveform = torch.randn(1, 1, 187)
    features = torch.randn(1, 25)
    
    # Generate heatmap
    gradcam = GradCAM(model)
    heatmap, pred, conf = gradcam.generate_heatmap(waveform, features)
    
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Prediction: {pred}, Confidence: {conf:.3f}")
    print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    # Generate report
    report = generate_explanation_report(model, waveform, features, true_label=0)
    print(f"\nExplanation Report:")
    print(f"  Prediction: {report['prediction_name']}")
    print(f"  Confidence: {report['confidence']:.1%}")
    print(f"  Needs Referral: {report['needs_referral']}")
    print(f"  Key Regions (ms): {report['key_regions_ms'][:5]}...")
