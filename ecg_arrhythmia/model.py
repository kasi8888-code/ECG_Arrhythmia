"""
Hybrid CNN Model for ECG Arrhythmia Detection
Combines 1D CNN (learned features) with engineered features (HRV/morphological)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

import config


class ConvBlock(nn.Module):
    """
    Convolutional block: Conv1D -> BatchNorm -> ReLU -> MaxPool -> Dropout
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        pool_size: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class ECGCNNBackbone(nn.Module):
    """
    1D CNN backbone for learning morphological features from raw ECG beats.
    """
    
    def __init__(
        self,
        input_length: int = 187,
        filters: list = None,
        kernel_size: int = 5,
        pool_size: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        filters = filters or config.CNN_FILTERS
        
        self.conv_blocks = nn.ModuleList()
        
        in_channels = 1
        for i, out_channels in enumerate(filters):
            self.conv_blocks.append(
                ConvBlock(in_channels, out_channels, kernel_size, pool_size, dropout)
            )
            in_channels = out_channels
        
        # Calculate output size after conv blocks
        self._output_length = input_length
        for _ in filters:
            self._output_length = self._output_length // pool_size
        
        self.output_dim = filters[-1] * self._output_length
        
        # Global average pooling alternative
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim_global = filters[-1]
    
    def forward(self, x: torch.Tensor, return_feature_maps: bool = False) -> torch.Tensor:
        """
        Forward pass through CNN backbone.
        
        Args:
            x: Input tensor (batch, 1, length)
            return_feature_maps: If True, return intermediate feature maps for Grad-CAM
        
        Returns:
            Feature tensor (batch, output_dim)
        """
        feature_maps = []
        
        for block in self.conv_blocks:
            x = block(x)
            feature_maps.append(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        if return_feature_maps:
            return x, feature_maps
        
        return x


class HybridECGClassifier(nn.Module):
    """
    Hybrid model combining CNN features with engineered features.
    
    Architecture:
    1. CNN backbone extracts learned features from raw waveform
    2. Engineered features are processed through a small MLP
    3. Both are concatenated and fused for final classification
    """
    
    def __init__(
        self,
        input_length: int = 187,
        num_engineered_features: int = 25,
        num_classes: int = 5,
        cnn_filters: list = None,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # CNN backbone for raw waveform
        self.cnn_backbone = ECGCNNBackbone(
            input_length=input_length,
            filters=cnn_filters,
            dropout=dropout
        )
        
        # MLP for engineered features
        self.feature_mlp = nn.Sequential(
            nn.Linear(num_engineered_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        fusion_input_dim = self.cnn_backbone.output_dim_global + 64
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.BatchNorm1d(fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Linear(fusion_hidden_dim // 2, num_classes)
        
        # Store intermediate activations for Grad-CAM
        self.cnn_features = None
        self.feature_maps = None
    
    def forward(
        self,
        waveform: torch.Tensor,
        engineered_features: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid model.
        
        Args:
            waveform: Raw ECG beat (batch, 1, length)
            engineered_features: Pre-computed features (batch, num_features)
            return_features: If True, return intermediate features
        
        Returns:
            Dictionary with logits and optionally features
        """
        # CNN features
        cnn_out, feature_maps = self.cnn_backbone(waveform, return_feature_maps=True)
        self.cnn_features = cnn_out
        self.feature_maps = feature_maps
        
        if engineered_features is not None:
            # Process engineered features
            eng_out = self.feature_mlp(engineered_features)
            
            # Concatenate and fuse
            fused = torch.cat([cnn_out, eng_out], dim=1)
        else:
            # CNN-only mode
            fused = cnn_out
            # Pad to match expected input size
            fused = F.pad(fused, (0, 64))
        
        # Fusion layers
        fused = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        output = {'logits': logits}
        
        if return_features:
            output['cnn_features'] = cnn_out
            output['fused_features'] = fused
            output['feature_maps'] = feature_maps
        
        return output
    
    def predict_with_confidence(
        self,
        waveform: torch.Tensor,
        engineered_features: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with calibrated confidence scores.
        
        Args:
            waveform: Input waveform
            engineered_features: Engineered features
            temperature: Temperature for calibration
        
        Returns:
            Dictionary with predictions, probabilities, and confidence
        """
        output = self.forward(waveform, engineered_features)
        logits = output['logits']
        
        # Temperature scaling for calibration
        scaled_logits = logits / temperature
        probabilities = F.softmax(scaled_logits, dim=1)
        
        # Get predictions and confidence
        confidence, predictions = torch.max(probabilities, dim=1)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence': confidence,
            'logits': logits
        }


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for confidence calibration.
    Learns optimal temperature on validation set.
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature
    
    def calibrate(
        self,
        model: nn.Module,
        val_loader,
        device: torch.device,
        max_iter: int = 50
    ) -> float:
        """
        Learn optimal temperature on validation set.
        
        Args:
            model: Trained model
            val_loader: Validation data loader
            device: Device to use
            max_iter: Maximum optimization iterations
        
        Returns:
            Optimal temperature value
        """
        model.eval()
        
        # Collect all logits and labels
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                waveform = batch['waveform'].to(device)
                labels = batch['label'].to(device)
                features = batch.get('features')
                if features is not None:
                    features = features.to(device)
                
                output = model(waveform, features)
                all_logits.append(output['logits'])
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def eval_fn():
            optimizer.zero_grad()
            scaled_logits = self.forward(all_logits)
            loss = criterion(scaled_logits, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_fn)
        
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        
        return self.temperature.item()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing HybridECGClassifier...")
    
    model = HybridECGClassifier(
        input_length=187,
        num_engineered_features=25,
        num_classes=5
    )
    
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 32
    waveform = torch.randn(batch_size, 1, 187)
    features = torch.randn(batch_size, 25)
    
    output = model(waveform, features, return_features=True)
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"CNN features shape: {output['cnn_features'].shape}")
    print(f"Fused features shape: {output['fused_features'].shape}")
    
    # Test prediction with confidence
    pred_output = model.predict_with_confidence(waveform, features)
    print(f"Predictions shape: {pred_output['predictions'].shape}")
    print(f"Confidence shape: {pred_output['confidence'].shape}")
    print(f"Sample confidence values: {pred_output['confidence'][:5]}")
