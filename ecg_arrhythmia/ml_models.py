"""
Traditional ML Model Training for ECG Arrhythmia Detection
Trains Random Forest, SVM, and XGBoost on engineered features
and compares with the existing CNN model.

Usage:
    python ml_models.py
"""
import numpy as np
import pandas as pd
import time
import json
import warnings
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, f1_score
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARNING] XGBoost not installed. Will use GradientBoosting as fallback.")
    print("  Install with: pip install xgboost")

import config
from data_loader import load_kaggle_ecg_data, create_patient_ids, patient_wise_split
from feature_engineering import ECGFeatureExtractor


# ======================================================================
# RESULTS DIRECTORY
# ======================================================================
COMPARISON_DIR = config.RESULTS_DIR / "model_comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
ML_MODEL_DIR = config.MODEL_DIR / "ml_models"
ML_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    """Load data and extract engineered features for ML models."""
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATA & EXTRACTING FEATURES")
    print("=" * 60)

    # Load raw data
    X_train, y_train, X_test, y_test = load_kaggle_ecg_data()

    # Create patient IDs and do patient-wise split
    patient_ids = create_patient_ids(X_train, y_train, num_simulated_patients=48)
    splits = patient_wise_split(
        X_train, y_train, patient_ids,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_state=config.RANDOM_SEED
    )
    # Use original test set
    splits['test'] = (X_test, y_test)

    # Extract engineered features
    extractor = ECGFeatureExtractor(sampling_rate=config.SAMPLING_RATE)

    features = {}
    labels = {}
    feature_stats = {}

    for split_name in ['train', 'val', 'test']:
        X, y = splits[split_name]
        print(f"\nExtracting features for {split_name} set ({len(y)} samples)...")
        feat = extractor.extract_batch_features(X, verbose=True)

        if split_name == 'train':
            feat, mean, std = extractor.normalize_features(feat)
            feature_stats['mean'] = mean
            feature_stats['std'] = std
        else:
            feat, _, _ = extractor.normalize_features(
                feat, mean=feature_stats['mean'], std=feature_stats['std']
            )

        # Replace NaN/Inf with 0
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        features[split_name] = feat
        labels[split_name] = y

    # Also keep raw waveforms for CNN
    waveforms = {
        'train': splits['train'][0],
        'val': splits['val'][0],
        'test': splits['test'][0]
    }

    print(f"\n✓ Feature extraction complete!")
    print(f"  Training samples: {len(labels['train'])}")
    print(f"  Validation samples: {len(labels['val'])}")
    print(f"  Test samples: {len(labels['test'])}")
    print(f"  Features per sample: {features['train'].shape[1]}")

    return features, labels, waveforms, feature_stats


def compute_class_weights_dict(y):
    """Compute class weights for imbalanced data."""
    classes = np.unique(y)
    n_samples = len(y)
    n_classes = len(classes)
    weights = {}
    for c in classes:
        weights[int(c)] = n_samples / (n_classes * np.sum(y == c))
    return weights


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest classifier."""
    print("\n" + "-" * 50)
    print("🌲 TRAINING: Random Forest")
    print("-" * 50)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=0
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Metrics
    results = compute_metrics(y_test, y_pred, y_train, y_pred_train, train_time, "Random Forest")

    # Feature importance
    results['feature_importance'] = model.feature_importances_.tolist()

    # Save model
    joblib.dump(model, ML_MODEL_DIR / "random_forest.joblib")
    print(f"  Model saved to {ML_MODEL_DIR / 'random_forest.joblib'}")

    return results


def train_svm(X_train, y_train, X_test, y_test):
    """Train SVM classifier."""
    print("\n" + "-" * 50)
    print("📊 TRAINING: SVM (RBF Kernel)")
    print("-" * 50)

    class_weights = compute_class_weights_dict(y_train)

    model = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        class_weight=class_weights,
        random_state=config.RANDOM_SEED,
        probability=True,
        verbose=False
    )

    # SVM is slow on large datasets, so subsample if needed
    n_train = len(y_train)
    max_svm_samples = 30000  # Limit for reasonable training time

    if n_train > max_svm_samples:
        print(f"  Subsampling {max_svm_samples} from {n_train} for SVM (speed)...")
        np.random.seed(config.RANDOM_SEED)
        indices = np.random.choice(n_train, max_svm_samples, replace=False)
        X_train_sub = X_train[indices]
        y_train_sub = y_train[indices]
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    start_time = time.time()
    model.fit(X_train_sub, y_train_sub)
    train_time = time.time() - start_time

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train_sub)

    # Metrics
    results = compute_metrics(y_test, y_pred, y_train_sub, y_pred_train, train_time, "SVM (RBF)")

    # Save model
    joblib.dump(model, ML_MODEL_DIR / "svm_rbf.joblib")
    print(f"  Model saved to {ML_MODEL_DIR / 'svm_rbf.joblib'}")

    return results


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier."""
    print("\n" + "-" * 50)
    print("🚀 TRAINING: XGBoost")
    print("-" * 50)

    class_weights = compute_class_weights_dict(y_train)

    if HAS_XGBOOST:
        # Compute sample weights
        sample_weights = np.array([class_weights[int(y)] for y in y_train])

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            eval_metric='mlogloss',
            verbosity=0
        )

        start_time = time.time()
        model.fit(X_train, y_train, sample_weight=sample_weights)
        train_time = time.time() - start_time
    else:
        # Fallback to sklearn GradientBoosting
        print("  Using sklearn GradientBoostingClassifier (XGBoost not available)")
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=config.RANDOM_SEED,
            verbose=0
        )
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    model_name = "XGBoost" if HAS_XGBOOST else "GradientBoosting"
    results = compute_metrics(y_test, y_pred, y_train, y_pred_train, train_time, model_name)

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        results['feature_importance'] = model.feature_importances_.tolist()

    # Save model
    save_name = "xgboost.joblib" if HAS_XGBOOST else "gradient_boosting.joblib"
    joblib.dump(model, ML_MODEL_DIR / save_name)
    print(f"  Model saved to {ML_MODEL_DIR / save_name}")

    return results


def train_cnn(waveforms, features, labels, feature_stats):
    """Train the 1D CNN Classifier model (existing architecture)."""
    import torch
    from model import HybridECGClassifier, TemperatureScaling
    from data_loader import compute_class_weights, create_data_loaders
    from trainer import Trainer, evaluate_model

    print("\n" + "-" * 50)
    print("🧠 TRAINING: 1D CNN Classifier (1D CNN + Engineered Features)")
    print("-" * 50)

    # Prepare splits for data loader
    splits = {
        'train': (waveforms['train'], labels['train']),
        'val': (waveforms['val'], labels['val']),
        'test': (waveforms['test'], labels['test'])
    }

    engineered_features = {
        'train': features['train'],
        'val': features['val'],
        'test': features['test']
    }

    # Create data loaders
    loaders = create_data_loaders(
        splits,
        engineered_features=engineered_features,
        batch_size=config.BATCH_SIZE,
        use_weighted_sampling=True
    )

    # Compute class weights
    class_weights = compute_class_weights(labels['train'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Training on: {device}")

    # Initialize model
    model = HybridECGClassifier(
        input_length=config.BEAT_LENGTH,
        num_engineered_features=features['train'].shape[1],
        num_classes=config.NUM_CLASSES,
        cnn_filters=config.CNN_FILTERS,
        fusion_hidden_dim=config.FUSION_HIDDEN_DIM,
        dropout=config.DROPOUT_RATE
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Trainer
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
    start_time = time.time()
    history = trainer.train(num_epochs=config.NUM_EPOCHS)
    train_time = time.time() - start_time

    # Load best model
    trainer.load_model(trainer.best_model_path)

    # Temperature calibration
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.calibrate(model, loaders['val'], device)

    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'temperature': optimal_temp,
        'feature_mean': feature_stats['mean'],
        'feature_std': feature_stats['std'],
        'config': {
            'num_classes': config.NUM_CLASSES,
            'input_length': config.BEAT_LENGTH,
            'num_features': features['train'].shape[1]
        }
    }
    torch.save(final_checkpoint, config.MODEL_DIR / 'final_model.pt')

    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loaders['test']:
            waveform = batch['waveform'].to(device)
            batch_labels = batch['label']
            feat = batch.get('features')
            if feat is not None:
                feat = feat.to(device)

            output = model.predict_with_confidence(waveform, feat, optimal_temp)
            all_preds.extend(output['predictions'].cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    # Train accuracy (on a subset)
    train_preds = []
    train_labels_list = []
    with torch.no_grad():
        for i, batch in enumerate(loaders['train']):
            if i >= 10:  # Only first 10 batches for speed
                break
            waveform = batch['waveform'].to(device)
            batch_labels = batch['label']
            feat = batch.get('features')
            if feat is not None:
                feat = feat.to(device)
            output = model.predict_with_confidence(waveform, feat, optimal_temp)
            train_preds.extend(output['predictions'].cpu().numpy())
            train_labels_list.extend(batch_labels.numpy())

    results = compute_metrics(
        y_true, y_pred,
        np.array(train_labels_list), np.array(train_preds),
        train_time,
        "1D CNN Classifier"
    )
    results['epochs_trained'] = len(history.get('train_loss', []))
    results['best_val_accuracy'] = max(history.get('val_accuracy', [0]))
    results['parameters'] = n_params

    return results


def compute_metrics(y_test, y_pred, y_train, y_pred_train, train_time, model_name):
    """Compute all metrics for a model."""
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_pred_train)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[0, 1, 2, 3, 4]
    )
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
    report = classification_report(
        y_test, y_pred,
        target_names=config.CLASS_NAMES,
        labels=[0, 1, 2, 3, 4],
        output_dict=True
    )

    results = {
        'model_name': model_name,
        'test_accuracy': float(test_acc),
        'train_accuracy': float(train_acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'training_time_seconds': float(train_time),
        'training_time_display': format_time(train_time),
    }

    # Print summary
    print(f"\n  ✅ {model_name} Results:")
    print(f"     Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"     Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"     Macro F1:       {macro_f1:.4f}")
    print(f"     Weighted F1:    {weighted_f1:.4f}")
    print(f"     Training Time:  {format_time(train_time)}")
    print(f"\n     Per-class F1:")
    for i, name in enumerate(config.CLASS_NAMES):
        print(f"       {name}: {f1[i]:.4f}")

    return results


def format_time(seconds):
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}hr"


def save_all_results(all_results):
    """Save all model results to JSON."""
    # Save individual results
    for result in all_results:
        name = result['model_name'].lower().replace(' ', '_').replace('(', '').replace(')', '')
        path = COMPARISON_DIR / f"{name}_results.json"
        with open(path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved: {path}")

    # Save combined comparison
    comparison = {
        'models': [r['model_name'] for r in all_results],
        'test_accuracies': [r['test_accuracy'] for r in all_results],
        'macro_f1_scores': [r['macro_f1'] for r in all_results],
        'weighted_f1_scores': [r['weighted_f1'] for r in all_results],
        'training_times': [r['training_time_display'] for r in all_results],
        'detailed_results': all_results
    }

    path = COMPARISON_DIR / "all_models_comparison.json"
    with open(path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"  Saved: {path}")


def print_comparison_table(all_results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON TABLE")
    print("=" * 80)

    # Header
    print(f"\n{'Model':<25} {'Test Acc':>10} {'Macro F1':>10} {'Weighted F1':>12} {'Time':>10}")
    print("-" * 70)

    # Sort by test accuracy
    sorted_results = sorted(all_results, key=lambda x: x['test_accuracy'], reverse=True)

    for i, r in enumerate(sorted_results):
        marker = " 🏆" if i == 0 else ""
        print(f"{r['model_name']:<25} {r['test_accuracy']*100:>9.2f}% {r['macro_f1']:>10.4f} {r['weighted_f1']:>12.4f} {r['training_time_display']:>10}{marker}")

    # Per-class breakdown
    print(f"\n\n{'Per-Class F1 Scores':}")
    print("-" * 90)
    print(f"{'Model':<25}", end="")
    for name in config.CLASS_NAMES:
        short = name.split('(')[1].replace(')', '').strip() if '(' in name else name[:6]
        print(f" {short:>10}", end="")
    print()
    print("-" * 90)

    for r in sorted_results:
        print(f"{r['model_name']:<25}", end="")
        for f1_val in r['per_class_f1']:
            print(f" {f1_val:>10.4f}", end="")
        print()

    # Best model
    best = sorted_results[0]
    print(f"\n🏆 BEST MODEL: {best['model_name']} with {best['test_accuracy']*100:.2f}% accuracy")


def main():
    """Main training pipeline."""
    np.random.seed(config.RANDOM_SEED)

    print("\n" + "=" * 60)
    print("ECG ARRHYTHMIA DETECTION — ML MODEL COMPARISON")
    print("Models: Random Forest, SVM, XGBoost, 1D CNN Classifier")
    print("=" * 60)

    # Step 1: Load data and extract features
    features, labels, waveforms, feature_stats = load_and_prepare_data()

    X_train = features['train']
    y_train = labels['train']
    X_test = features['test']
    y_test = labels['test']

    all_results = []

    # Step 2: Train traditional ML models
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING TRADITIONAL ML MODELS")
    print("=" * 60)

    # 2a. Random Forest
    rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    all_results.append(rf_results)

    # 2b. SVM
    svm_results = train_svm(X_train, y_train, X_test, y_test)
    all_results.append(svm_results)

    # 2c. XGBoost
    xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
    all_results.append(xgb_results)

    # Step 3: Train CNN
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING DEEP LEARNING MODEL")
    print("=" * 60)

    cnn_results = train_cnn(waveforms, features, labels, feature_stats)
    all_results.append(cnn_results)

    # Step 4: Compare
    print_comparison_table(all_results)

    # Step 5: Save results
    print("\n" + "=" * 60)
    print("STEP 4: SAVING RESULTS")
    print("=" * 60)
    save_all_results(all_results)

    print("\n" + "=" * 60)
    print("✅ ALL MODELS TRAINED AND COMPARED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nResults saved to: {COMPARISON_DIR}")
    print(f"ML models saved to: {ML_MODEL_DIR}")
    print(f"\nNext: Run 'python compare_models.py' for detailed visualizations.")


if __name__ == "__main__":
    main()
