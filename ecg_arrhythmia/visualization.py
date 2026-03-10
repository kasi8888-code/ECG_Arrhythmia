"""
Visualization utilities for ECG Arrhythmia Detection
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import config


def plot_training_history(history: Dict, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot training history (loss, accuracy, learning rate).
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Loss difference (generalization gap)
    ax4 = axes[1, 1]
    gap = np.array(history['val_loss']) - np.array(history['train_loss'])
    ax4.plot(epochs, gap, 'm-', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Val Loss - Train Loss', fontsize=12)
    ax4.set_title('Generalization Gap', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Whether to normalize by row
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot ROC curves for each class.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities (n_samples, n_classes)
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    # Binarize labels
    n_classes = y_probs.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) > 0:  # Only plot if class exists in data
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(
                fpr, tpr,
                color=colors[i],
                linewidth=2,
                label=f'{config.CLASS_NAMES[i]} (AUC = {roc_auc:.3f})'
            )
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    return fig


def plot_confidence_distribution(
    confidences: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    referral_threshold: float = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot confidence distribution for correct and incorrect predictions.
    
    Args:
        confidences: Prediction confidences
        y_true: True labels
        y_pred: Predicted labels
        referral_threshold: Threshold for referral
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    referral_threshold = referral_threshold or config.REFERRAL_THRESHOLD
    
    correct = y_pred == y_true
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of confidences
    ax1 = axes[0]
    ax1.hist(confidences[correct], bins=50, alpha=0.7, label='Correct', color='green', density=True)
    ax1.hist(confidences[~correct], bins=50, alpha=0.7, label='Incorrect', color='red', density=True)
    ax1.axvline(x=referral_threshold, color='orange', linestyle='--', linewidth=2, label=f'Referral Threshold ({referral_threshold})')
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Confidence Distribution by Correctness', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Reliability diagram (calibration curve)
    ax2 = axes[1]
    
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1])
        if np.sum(mask) > 0:
            bin_accuracies.append(np.mean(correct[mask]))
            bin_confidences.append(np.mean(confidences[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(0)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)
    
    # Plot calibration curve
    ax2.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, label='Accuracy')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    ax2.set_xlabel('Mean Predicted Confidence', fontsize=12)
    ax2.set_ylabel('Fraction of Positives', fontsize=12)
    ax2.set_title('Reliability Diagram (Calibration Curve)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confidence distribution saved to {save_path}")
    
    return fig


def plot_sample_beats(
    beats: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 3,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot sample beats from each class.
    
    Args:
        beats: Beat waveforms (n_samples, length)
        labels: Labels
        n_samples: Number of samples per class
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    n_classes = len(config.CLASS_NAMES)
    
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(4*n_samples, 3*n_classes))
    
    time = np.arange(beats.shape[1]) / config.SAMPLING_RATE * 1000
    
    for class_idx in range(n_classes):
        class_mask = labels == class_idx
        class_beats = beats[class_mask]
        
        for sample_idx in range(n_samples):
            ax = axes[class_idx, sample_idx] if n_classes > 1 else axes[sample_idx]
            
            if sample_idx < len(class_beats):
                ax.plot(time, class_beats[sample_idx], 'b-', linewidth=1)
                ax.set_title(f'{config.CLASS_NAMES[class_idx]}' if sample_idx == 1 else '')
            else:
                ax.text(0.5, 0.5, 'No sample', ha='center', va='center', transform=ax.transAxes)
            
            if sample_idx == 0:
                ax.set_ylabel(config.CLASS_NAMES[class_idx], fontsize=10)
            
            if class_idx == n_classes - 1:
                ax.set_xlabel('Time (ms)', fontsize=10)
            
            ax.grid(True, alpha=0.3)
    
    fig.suptitle('Sample ECG Beats by Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sample beats plot saved to {save_path}")
    
    return fig


def plot_class_distribution(
    y: np.ndarray,
    split_name: str = 'Dataset',
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot class distribution.
    
    Args:
        y: Labels
        split_name: Name of the data split
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    unique, counts = np.unique(y, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique)))
    
    bars = ax.bar(
        [config.CLASS_NAMES[i] for i in unique],
        counts,
        color=colors
    )
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f'{count:,}\n({100*count/len(y):.1f}%)',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Class Distribution - {split_name}', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    return fig


def plot_referral_analysis(
    referral_analysis: Dict,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot referral analysis results.
    
    Args:
        referral_analysis: Referral analysis dictionary
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Pie chart: Referred vs Not Referred
    ax1 = axes[0]
    sizes = [referral_analysis['referred_count'], referral_analysis['not_referred_count']]
    labels = ['Referred', 'Not Referred']
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title('Referral Distribution', fontsize=14)
    
    # Bar chart: Accuracy comparison
    ax2 = axes[1]
    categories = ['Non-Referred', 'Referred']
    accuracies = [
        referral_analysis.get('non_referred_accuracy', 0),
        referral_analysis.get('referred_accuracy', 0)
    ]
    
    bars = ax2.bar(categories, accuracies, color=['green', 'orange'])
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy by Referral Status', fontsize=14)
    ax2.set_ylim([0, 1])
    
    for bar, acc in zip(bars, accuracies):
        ax2.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                f'{acc:.1%}', ha='center', fontsize=12)
    
    # Confidence distribution
    ax3 = axes[2]
    stats = referral_analysis.get('confidence_stats', {})
    
    box_data = {
        'Mean': stats.get('mean', 0),
        'Median': stats.get('median', 0),
        'Std': stats.get('std', 0)
    }
    
    ax3.bar(box_data.keys(), box_data.values(), color='steelblue')
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Confidence Statistics', fontsize=14)
    ax3.axhline(y=config.REFERRAL_THRESHOLD, color='red', linestyle='--',
                label=f'Threshold ({config.REFERRAL_THRESHOLD})')
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Referral analysis plot saved to {save_path}")
    
    return fig
