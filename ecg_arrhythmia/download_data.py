"""
Download ECG Heartbeat Dataset from Kaggle
"""
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from pathlib import Path
import shutil

# Data directory
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

print("Downloading ECG Heartbeat dataset from Kaggle...")
print("This may take a few minutes...")

# Download train file
print("\n1. Loading training data...")
df_train = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "shayanfazeli/heartbeat",
    "mitbih_train.csv",
)
print(f"   Training samples: {len(df_train)}")
print(f"   Shape: {df_train.shape}")

# Download test file
print("\n2. Loading test data...")
df_test = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "shayanfazeli/heartbeat",
    "mitbih_test.csv",
)
print(f"   Test samples: {len(df_test)}")
print(f"   Shape: {df_test.shape}")

# Save to data directory
train_path = DATA_DIR / "mitbih_train.csv"
test_path = DATA_DIR / "mitbih_test.csv"

print(f"\n3. Saving to {DATA_DIR}...")
df_train.to_csv(train_path, index=False, header=False)
df_test.to_csv(test_path, index=False, header=False)

print(f"\n✓ Dataset downloaded successfully!")
print(f"  - {train_path}")
print(f"  - {test_path}")

# Show class distribution
print("\n4. Class distribution (Training set):")
class_names = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
labels = df_train.iloc[:, -1].astype(int)
for i, name in enumerate(class_names):
    count = (labels == i).sum()
    pct = 100 * count / len(labels)
    print(f"   Class {i} - {name}: {count:,} ({pct:.1f}%)")

print("\n✓ Ready to train! Run: python main.py --mode train")
