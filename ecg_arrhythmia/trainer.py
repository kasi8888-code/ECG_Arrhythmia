"""
Training Pipeline for ECG Arrhythmia Detection
Includes training loop, validation, and model checkpointing
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
import time
import json

import config
from model import HybridECGClassifier, TemperatureScaling


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """
    Training manager for ECG arrhythmia detection model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        save_dir: Path = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir or config.MODEL_DIR
        
        # Loss function with class weights
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            mode='min'
        )
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.best_model_path = None
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            waveform = batch['waveform'].to(self.device)
            labels = batch['label'].to(self.device)
            features = batch.get('features')
            if features is not None:
                features = features.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(waveform, features)
            logits = output['logits']
            
            # Loss and backward
            loss = self.criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, Dict]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in self.val_loader:
            waveform = batch['waveform'].to(self.device)
            labels = batch['label'].to(self.device)
            features = batch.get('features')
            if features is not None:
                features = features.to(self.device)
            
            output = self.model(waveform, features)
            logits = output['logits']
            
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        for i, name in enumerate(config.CLASS_NAMES):
            if i < len(per_class_f1):
                metrics[f'{name}_f1'] = per_class_f1[i]
        
        return avg_loss, accuracy, metrics
    
    def train(self, num_epochs: int = 50) -> Dict:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
        
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Starting training on {self.device}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f} | LR: {current_lr:.2e}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_path = self.save_dir / 'best_model.pt'
                self.save_model(self.best_model_path, val_metrics)
                print(f"  ✓ New best model saved!")
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
        
        return self.history
    
    def save_model(self, path: Path, metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': {
                'num_classes': self.model.num_classes,
                'input_length': config.BEAT_LENGTH,
            }
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, path)
    
    def load_model(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        print(f"Loaded model from {path}")


def evaluate_model(
    model: nn.Module,
    test_loader,
    device: torch.device,
    temperature: float = 1.0
) -> Dict:
    """
    Comprehensive model evaluation on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to use
        temperature: Temperature for calibration
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_confidence = []
    
    with torch.no_grad():
        for batch in test_loader:
            waveform = batch['waveform'].to(device)
            labels = batch['label'].to(device)
            features = batch.get('features')
            if features is not None:
                features = features.to(device)
            
            output = model.predict_with_confidence(waveform, features, temperature)
            
            all_preds.extend(output['predictions'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(output['probabilities'].cpu().numpy())
            all_confidence.extend(output['confidence'].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_confidence = np.array(all_confidence)
    
    # Basic metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class metrics
    class_report = classification_report(
        all_labels, all_preds,
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    
    # Referral analysis
    referral_mask = all_confidence < config.REFERRAL_THRESHOLD
    referral_rate = np.mean(referral_mask)
    
    # Accuracy on non-referred predictions
    if np.sum(~referral_mask) > 0:
        non_referred_accuracy = accuracy_score(
            all_labels[~referral_mask],
            all_preds[~referral_mask]
        )
    else:
        non_referred_accuracy = 0
    
    # Accuracy on referred predictions (would have been wrong)
    if np.sum(referral_mask) > 0:
        referred_accuracy = accuracy_score(
            all_labels[referral_mask],
            all_preds[referral_mask]
        )
    else:
        referred_accuracy = 0
    
    # Calibration metrics (Expected Calibration Error)
    ece = compute_ece(all_probs, all_labels, n_bins=10)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'referral_rate': referral_rate,
        'non_referred_accuracy': non_referred_accuracy,
        'referred_accuracy': referred_accuracy,
        'expected_calibration_error': ece,
        'mean_confidence': np.mean(all_confidence),
        'confidence_std': np.std(all_confidence)
    }
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"\nCalibration:")
    print(f"  ECE: {ece:.4f}")
    print(f"  Mean Confidence: {np.mean(all_confidence):.4f}")
    print(f"\nReferral Analysis (threshold={config.REFERRAL_THRESHOLD}):")
    print(f"  Referral Rate: {referral_rate:.1%}")
    print(f"  Non-referred Accuracy: {non_referred_accuracy:.4f}")
    print(f"  Referred Accuracy: {referred_accuracy:.4f}")
    print("\nPer-class F1 Scores:")
    for name in config.CLASS_NAMES:
        if name in class_report:
            print(f"  {name}: {class_report[name]['f1-score']:.4f}")
    print("="*60)
    
    return results


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities (n_samples, n_classes)
        labels: True labels (n_samples,)
        n_bins: Number of bins
    
    Returns:
        ECE value
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    
    for i in range(n_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        bin_size = np.sum(bin_mask)
        
        if bin_size > 0:
            bin_accuracy = np.mean(accuracies[bin_mask])
            bin_confidence = np.mean(confidences[bin_mask])
            ece += (bin_size / len(labels)) * np.abs(bin_accuracy - bin_confidence)
    
    return ece


def save_results(results: Dict, path: Path):
    """Save evaluation results to JSON."""
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results = convert(results)
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {path}")
