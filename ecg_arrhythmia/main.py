"""
Main Training Script for ECG Arrhythmia Detection System

Usage:
    python main.py --mode train
    python main.py --mode evaluate
    python main.py --mode demo
"""
import argparse
import torch
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import config
from data_loader import (
    load_kaggle_ecg_data, create_patient_ids, patient_wise_split,
    compute_class_weights, create_data_loaders, download_instructions
)
from feature_engineering import ECGFeatureExtractor
from model import HybridECGClassifier, TemperatureScaling
from trainer import Trainer, evaluate_model, save_results
from inference import ECGArrhythmiaDetector, analyze_referrals
from visualization import (
    plot_training_history, plot_confusion_matrix, plot_roc_curves,
    plot_confidence_distribution, plot_sample_beats, plot_class_distribution
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data():
    """Load and prepare data with patient-wise splitting."""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPARATION")
    print("="*60)
    
    try:
        # Load data
        X_train, y_train, X_test, y_test = load_kaggle_ecg_data()
    except FileNotFoundError:
        download_instructions()
        raise SystemExit("Please download the dataset and run again.")
    
    # Create patient IDs for patient-wise splitting
    patient_ids = create_patient_ids(X_train, y_train, num_simulated_patients=48)
    
    # Patient-wise split
    splits = patient_wise_split(
        X_train, y_train, patient_ids,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_state=config.RANDOM_SEED
    )
    
    # Add original test set to splits
    splits['test'] = (X_test, y_test)
    
    print("\n" + "="*60)
    print("STEP 2: FEATURE EXTRACTION")
    print("="*60)
    
    # Extract engineered features
    extractor = ECGFeatureExtractor(sampling_rate=config.SAMPLING_RATE)
    
    engineered_features = {}
    feature_stats = {}
    
    for split_name, (X, y) in splits.items():
        print(f"\nExtracting features for {split_name} set...")
        features = extractor.extract_batch_features(X, verbose=True)
        
        if split_name == 'train':
            # Normalize using training statistics
            features, mean, std = extractor.normalize_features(features)
            feature_stats['mean'] = mean
            feature_stats['std'] = std
        else:
            # Use training statistics for val/test
            features, _, _ = extractor.normalize_features(
                features,
                mean=feature_stats['mean'],
                std=feature_stats['std']
            )
        
        engineered_features[split_name] = features
    
    # Plot class distributions
    for split_name, (X, y) in splits.items():
        plot_class_distribution(
            y, split_name=split_name.upper(),
            save_path=config.PLOTS_DIR / f'class_distribution_{split_name}.png'
        )
    
    # Plot sample beats
    plot_sample_beats(
        splits['train'][0], splits['train'][1],
        n_samples=3,
        save_path=config.PLOTS_DIR / 'sample_beats.png'
    )
    
    return splits, engineered_features, feature_stats


def train_model(splits, engineered_features, feature_stats):
    """Train the hybrid ECG model."""
    print("\n" + "="*60)
    print("STEP 3: MODEL TRAINING")
    print("="*60)
    
    # Create data loaders
    loaders = create_data_loaders(
        splits,
        engineered_features=engineered_features,
        batch_size=config.BATCH_SIZE,
        use_weighted_sampling=True
    )
    
    # Compute class weights
    class_weights = compute_class_weights(splits['train'][1])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Initialize model
    model = HybridECGClassifier(
        input_length=config.BEAT_LENGTH,
        num_engineered_features=len(engineered_features['train'][0]),
        num_classes=config.NUM_CLASSES,
        cnn_filters=config.CNN_FILTERS,
        fusion_hidden_dim=config.FUSION_HIDDEN_DIM,
        dropout=config.DROPOUT_RATE
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        device=device,
        class_weights=class_weights if config.USE_CLASS_WEIGHTS else None,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        save_dir=config.MODEL_DIR
    )
    
    # Train
    history = trainer.train(num_epochs=config.NUM_EPOCHS)
    
    # Plot training history
    plot_training_history(
        history,
        save_path=config.PLOTS_DIR / 'training_history.png'
    )
    
    # Calibrate confidence
    print("\n" + "="*60)
    print("STEP 4: CONFIDENCE CALIBRATION")
    print("="*60)
    
    # Load best model
    trainer.load_model(trainer.best_model_path)
    
    # Temperature scaling
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.calibrate(model, loaders['val'], device)
    
    # Save final model with calibration
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'temperature': optimal_temp,
        'feature_mean': feature_stats['mean'],
        'feature_std': feature_stats['std'],
        'config': {
            'num_classes': config.NUM_CLASSES,
            'input_length': config.BEAT_LENGTH,
            'num_features': len(engineered_features['train'][0])
        }
    }
    
    final_model_path = config.MODEL_DIR / 'final_model.pt'
    torch.save(final_checkpoint, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return model, loaders, device, optimal_temp


def evaluate_and_visualize(model, loaders, device, temperature):
    """Evaluate model and create visualizations."""
    print("\n" + "="*60)
    print("STEP 5: EVALUATION")
    print("="*60)
    
    # Evaluate on test set
    results = evaluate_model(model, loaders['test'], device, temperature)
    
    # Save results
    save_results(results, config.RESULTS_DIR / 'evaluation_results.json')
    
    # Get predictions for visualization
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_confidence = []
    
    with torch.no_grad():
        for batch in loaders['test']:
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
    
    # Confusion matrix
    plot_confusion_matrix(
        all_labels, all_preds,
        normalize=True,
        save_path=config.PLOTS_DIR / 'confusion_matrix.png'
    )
    
    # ROC curves
    plot_roc_curves(
        all_labels, all_probs,
        save_path=config.PLOTS_DIR / 'roc_curves.png'
    )
    
    # Confidence distribution
    plot_confidence_distribution(
        all_confidence, all_labels, all_preds,
        referral_threshold=config.REFERRAL_THRESHOLD,
        save_path=config.PLOTS_DIR / 'confidence_distribution.png'
    )
    
    return results, all_preds, all_labels, all_probs, all_confidence


def run_demo():
    """Run a demo with sample predictions and explanations."""
    print("\n" + "="*60)
    print("DEMO MODE")
    print("="*60)
    
    model_path = config.MODEL_DIR / 'final_model.pt'
    
    if not model_path.exists():
        print("No trained model found. Training a demo model...")
        # Use synthetic data for demo
        print("Please run with --mode train first to train the model.")
        return
    
    # Initialize detector
    detector = ECGArrhythmiaDetector(model_path=model_path)
    
    # Load some test data
    try:
        X_train, y_train, X_test, y_test = load_kaggle_ecg_data()
        
        # Get sample beats from each class
        print("\nGenerating predictions with explanations...")
        
        for class_idx in range(config.NUM_CLASSES):
            class_mask = y_test == class_idx
            if np.sum(class_mask) > 0:
                # Get first beat of this class
                beat = X_test[class_mask][0]
                true_label = class_idx
                
                # Generate report with explanation
                report = detector.generate_report(
                    beat,
                    true_label=true_label,
                    save_path=config.PLOTS_DIR / f'gradcam_class_{class_idx}.png'
                )
                
                print(f"\n--- Class {class_idx}: {config.CLASS_NAMES[class_idx]} ---")
                print(f"  Prediction: {report['prediction_name']}")
                print(f"  Confidence: {report['confidence']:.1%}")
                print(f"  Correct: {report['correct']}")
                print(f"  Needs Referral: {report['needs_referral']}")
                if report['needs_referral']:
                    print(f"  Referral Reason: {report['referral_reason']}")
        
        print(f"\nSignal Attention Heatmap visualizations saved to {config.PLOTS_DIR}")
        
    except FileNotFoundError:
        download_instructions()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='ECG Arrhythmia Detection System')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'demo'],
        default='train',
        help='Mode: train, evaluate, or demo'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=config.RANDOM_SEED,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    print("\n" + "="*60)
    print("ECG ARRHYTHMIA DETECTION SYSTEM")
    print("1D CNN Classifier + Engineered Features with Explainability")
    print("="*60)
    
    if args.mode == 'train':
        # Full training pipeline
        splits, engineered_features, feature_stats = prepare_data()
        model, loaders, device, temperature = train_model(splits, engineered_features, feature_stats)
        results, preds, labels, probs, confidences = evaluate_and_visualize(model, loaders, device, temperature)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Model saved to: {config.MODEL_DIR}")
        print(f"Results saved to: {config.RESULTS_DIR}")
        print(f"Plots saved to: {config.PLOTS_DIR}")
        
    elif args.mode == 'evaluate':
        # Evaluate existing model
        model_path = config.MODEL_DIR / 'final_model.pt'
        if not model_path.exists():
            print("No trained model found. Run with --mode train first.")
            return
        
        splits, engineered_features, feature_stats = prepare_data()
        loaders = create_data_loaders(splits, engineered_features, batch_size=config.BATCH_SIZE)
        
        # Load model
        checkpoint = torch.load(model_path, weights_only=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = HybridECGClassifier(
            input_length=config.BEAT_LENGTH,
            num_engineered_features=len(engineered_features['train'][0]),
            num_classes=config.NUM_CLASSES
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        temperature = checkpoint.get('temperature', 1.0)
        
        evaluate_and_visualize(model, loaders, device, temperature)
        
    elif args.mode == 'demo':
        run_demo()


if __name__ == "__main__":
    main()
