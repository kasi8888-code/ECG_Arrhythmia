"""
Data Loading and Patient-Wise Splitting for ECG Arrhythmia Detection
Supports Kaggle ECG Heartbeat Categorization Dataset (MIT-BIH format)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import config


def load_kaggle_ecg_data(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Kaggle ECG Heartbeat Categorization Dataset.
    
    Expected files:
    - mitbih_train.csv: Training data (87554 samples)
    - mitbih_test.csv: Test data (21892 samples)
    
    Each row: 187 values (186 signal samples + 1 label)
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    train_path = train_path or config.DATA_DIR / "mitbih_train.csv"
    test_path = test_path or config.DATA_DIR / "mitbih_test.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Dataset not found. Please download from Kaggle:\n"
            f"https://www.kaggle.com/datasets/shayanfazeli/heartbeat\n"
            f"Place files in: {config.DATA_DIR}\n"
            f"Expected files: mitbih_train.csv, mitbih_test.csv"
        )
    
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path, header=None)
    
    print(f"Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path, header=None)
    
    # Split features and labels (last column is label)
    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(np.int64)
    
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.int64)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Beat length: {X_train.shape[1]} samples")
    print(f"Classes: {np.unique(y_train)}")
    
    return X_train, y_train, X_test, y_test


def create_patient_ids(
    X: np.ndarray,
    y: np.ndarray,
    num_simulated_patients: int = 48
) -> np.ndarray:
    """
    Create simulated patient IDs for patient-wise splitting.
    
    Note: The Kaggle dataset doesn't include patient IDs directly.
    We simulate patient-wise splitting by:
    1. Grouping consecutive beats (simulating recording segments)
    2. Assigning patient IDs to ensure beats from same "patient" stay together
    
    For true patient-wise splitting with MIT-BIH:
    - Records 100-124: Patient group 1 (DS1 - training)
    - Records 200-234: Patient group 2 (DS2 - testing)
    
    Args:
        X: Feature array
        y: Label array
        num_simulated_patients: Number of simulated patients to create
    
    Returns:
        patient_ids: Array of patient IDs
    """
    n_samples = len(X)
    
    # Create patient IDs by dividing data into segments
    # Each segment represents a "patient's recording"
    samples_per_patient = n_samples // num_simulated_patients
    
    patient_ids = np.zeros(n_samples, dtype=np.int32)
    
    for i in range(num_simulated_patients):
        start_idx = i * samples_per_patient
        end_idx = start_idx + samples_per_patient if i < num_simulated_patients - 1 else n_samples
        patient_ids[start_idx:end_idx] = i
    
    print(f"Created {num_simulated_patients} simulated patient groups")
    print(f"Samples per patient: ~{samples_per_patient}")
    
    return patient_ids


def patient_wise_split(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split data ensuring no patient appears in multiple splits.
    
    This is CRITICAL for valid performance claims - mixing patients
    across splits leads to data leakage and inflated metrics.
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Label array (n_samples,)
        patient_ids: Patient ID for each sample
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with train/val/test splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    np.random.seed(random_state)
    
    # Get unique patient IDs
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    
    # Shuffle patients
    np.random.shuffle(unique_patients)
    
    # Calculate split points
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    # Assign patients to splits
    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train + n_val])
    test_patients = set(unique_patients[n_train + n_val:])
    
    # Create masks for each split
    train_mask = np.array([pid in train_patients for pid in patient_ids])
    val_mask = np.array([pid in val_patients for pid in patient_ids])
    test_mask = np.array([pid in test_patients for pid in patient_ids])
    
    splits = {
        'train': (X[train_mask], y[train_mask]),
        'val': (X[val_mask], y[val_mask]),
        'test': (X[test_mask], y[test_mask])
    }
    
    # Verify no patient overlap
    assert len(train_patients & val_patients) == 0, "Patient overlap: train & val"
    assert len(train_patients & test_patients) == 0, "Patient overlap: train & test"
    assert len(val_patients & test_patients) == 0, "Patient overlap: val & test"
    
    print("\n=== Patient-Wise Split Summary ===")
    print(f"Total patients: {n_patients}")
    print(f"Train patients: {len(train_patients)} ({len(splits['train'][0])} beats)")
    print(f"Val patients: {len(val_patients)} ({len(splits['val'][0])} beats)")
    print(f"Test patients: {len(test_patients)} ({len(splits['test'][0])} beats)")
    
    # Print class distribution per split
    for split_name, (X_split, y_split) in splits.items():
        unique, counts = np.unique(y_split, return_counts=True)
        print(f"\n{split_name.upper()} class distribution:")
        for cls, cnt in zip(unique, counts):
            print(f"  Class {cls} ({config.CLASS_NAMES[cls]}): {cnt} ({100*cnt/len(y_split):.1f}%)")
    
    return splits


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced data.
    Uses inverse frequency weighting.
    
    Args:
        y: Label array
    
    Returns:
        Class weights tensor
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    # Inverse frequency weighting
    weights = total / (len(unique) * counts)
    
    # Normalize
    weights = weights / weights.sum() * len(unique)
    
    print("\n=== Class Weights ===")
    for cls, (cnt, w) in enumerate(zip(counts, weights)):
        print(f"Class {cls}: count={cnt}, weight={w:.3f}")
    
    return torch.FloatTensor(weights)


class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG beats.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        engineered_features: Optional[np.ndarray] = None,
        transform=None
    ):
        """
        Args:
            X: Beat waveforms (n_samples, beat_length)
            y: Labels (n_samples,)
            engineered_features: Pre-computed engineered features (n_samples, n_features)
            transform: Optional transform to apply
        """
        self.X = torch.FloatTensor(X).unsqueeze(1)  # (N, 1, L) for Conv1d
        self.y = torch.LongTensor(y)
        self.engineered_features = (
            torch.FloatTensor(engineered_features) 
            if engineered_features is not None 
            else None
        )
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            'waveform': self.X[idx],
            'label': self.y[idx]
        }
        
        if self.engineered_features is not None:
            sample['features'] = self.engineered_features[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def create_data_loaders(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    engineered_features: Optional[Dict[str, np.ndarray]] = None,
    batch_size: int = 256,
    use_weighted_sampling: bool = True
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test splits.
    
    Args:
        splits: Dictionary with train/val/test data
        engineered_features: Optional dict with engineered features for each split
        batch_size: Batch size
        use_weighted_sampling: Use weighted random sampling for training
    
    Returns:
        Dictionary of DataLoaders
    """
    loaders = {}
    
    for split_name, (X, y) in splits.items():
        features = engineered_features.get(split_name) if engineered_features else None
        
        dataset = ECGDataset(X, y, features)
        
        if split_name == 'train' and use_weighted_sampling:
            # Weighted sampling to handle class imbalance
            class_counts = np.bincount(y)
            sample_weights = 1.0 / class_counts[y]
            sample_weights = torch.FloatTensor(sample_weights)
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=0,
                pin_memory=True
            )
        else:
            loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == 'train'),
                num_workers=0,
                pin_memory=True
            )
    
    return loaders


def download_instructions():
    """Print instructions for downloading the dataset."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    DATASET DOWNLOAD INSTRUCTIONS                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  1. Go to: https://www.kaggle.com/datasets/shayanfazeli/heartbeat     ║
║                                                                        ║
║  2. Download the dataset (requires Kaggle account)                     ║
║                                                                        ║
║  3. Extract and place these files in the 'data' folder:               ║
║     - mitbih_train.csv                                                 ║
║     - mitbih_test.csv                                                  ║
║                                                                        ║
║  Alternative (Kaggle CLI):                                             ║
║  $ kaggle datasets download -d shayanfazeli/heartbeat                  ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    # Test data loading
    try:
        X_train, y_train, X_test, y_test = load_kaggle_ecg_data()
        
        # Create patient IDs
        patient_ids = create_patient_ids(X_train, y_train)
        
        # Patient-wise split
        splits = patient_wise_split(X_train, y_train, patient_ids)
        
        # Compute class weights
        class_weights = compute_class_weights(splits['train'][1])
        
    except FileNotFoundError:
        download_instructions()
