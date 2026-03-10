"""
Model Comparison Visualization for ECG Arrhythmia Detection
Generates comparison charts and summary from trained model results.

Usage:
    python compare_models.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import json
from pathlib import Path

import config

# Results directory
COMPARISON_DIR = config.RESULTS_DIR / "model_comparison"


def load_results():
    """Load all model results from JSON files."""
    results_file = COMPARISON_DIR / "all_models_comparison.json"

    if not results_file.exists():
        print("❌ No comparison results found!")
        print("   Run 'python ml_models.py' first to train all models.")
        return None

    with open(results_file, 'r') as f:
        data = json.load(f)

    return data['detailed_results']


def plot_accuracy_comparison(results, save_path):
    """Bar chart comparing test accuracy of all models."""
    models = [r['model_name'] for r in results]
    accuracies = [r['test_accuracy'] * 100 for r in results]
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']

    # Sort by accuracy
    sorted_pairs = sorted(zip(models, accuracies, colors), key=lambda x: x[1], reverse=True)
    models_s, acc_s, colors_s = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models_s, acc_s, color=colors_s, edgecolor='white', linewidth=1.5, width=0.6)

    # Add value labels on bars
    for bar, acc in zip(bars, acc_s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)

    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_title('Model Comparison — Test Accuracy', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(0, max(acc_s) + 8)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_f1_comparison(results, save_path):
    """Grouped bar chart comparing per-class F1 scores."""
    models = [r['model_name'] for r in results]
    class_short = ['N', 'S', 'V', 'F', 'Q']

    x = np.arange(len(class_short))
    width = 0.18
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (r, color) in enumerate(zip(results, colors)):
        f1_scores = r['per_class_f1']
        offset = (i - len(results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1_scores, width, label=r['model_name'],
                       color=color, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Arrhythmia Class', fontsize=13)
    ax.set_ylabel('F1 Score', fontsize=13)
    ax.set_title('Per-Class F1 Scores Comparison', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}\n{n}' for s, n in zip(class_short, config.CLASS_NAMES)], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrices(results, save_path):
    """Side-by-side confusion matrices for all models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    class_short = ['N', 'S', 'V', 'F', 'Q']

    for ax, r in zip(axes, results):
        cm = np.array(r['confusion_matrix'])
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_short, yticklabels=class_short,
                    ax=ax, cbar=False, linewidths=0.5)
        ax.set_title(f'{r["model_name"]}\nAcc: {r["test_accuracy"]*100:.1f}%',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.suptitle('Normalized Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_time(results, save_path):
    """Bar chart comparing training times."""
    models = [r['model_name'] for r in results]
    times = [r['training_time_seconds'] for r in results]
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, times, color=colors, edgecolor='white', linewidth=1.5, width=0.6)

    for bar, t in zip(bars, times):
        label = f'{t:.1f}s' if t < 60 else f'{t/60:.1f}min'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.02,
                label, ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylabel('Training Time (seconds)', fontsize=13)
    ax.set_title('Training Time Comparison', fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_radar_chart(results, save_path):
    """Radar chart comparing models across multiple metrics."""
    metrics = ['Test Accuracy', 'Macro F1', 'N (Normal)', 'V (Ventricle)', 'S (Supra.)']

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for r, color in zip(results, colors):
        values = [
            r['test_accuracy'],
            r['macro_f1'],
            r['per_class_f1'][0],  # Normal
            r['per_class_f1'][2],  # Ventricular
            r['per_class_f1'][1],  # Supraventricular
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=r['model_name'], color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title('Multi-Metric Model Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_summary_dashboard(results, save_path):
    """Create a comprehensive summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    models = [r['model_name'] for r in results]
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']

    # Sort by accuracy for consistent ordering
    sorted_idx = sorted(range(len(results)), key=lambda i: results[i]['test_accuracy'], reverse=True)
    models_s = [models[i] for i in sorted_idx]
    colors_s = [colors[i] for i in sorted_idx]
    results_s = [results[i] for i in sorted_idx]

    # ---- Plot 1: Accuracy ----
    ax = axes[0, 0]
    accs = [r['test_accuracy'] * 100 for r in results_s]
    bars = ax.barh(models_s, accs, color=colors_s, edgecolor='white', height=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{acc:.2f}%', va='center', fontweight='bold', fontsize=11)
    ax.set_xlim(0, max(accs) + 10)
    ax.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ---- Plot 2: F1 Scores (Macro & Weighted) ----
    ax = axes[0, 1]
    x = np.arange(len(models_s))
    w = 0.35
    macro = [r['macro_f1'] for r in results_s]
    weighted = [r['weighted_f1'] for r in results_s]
    ax.bar(x - w / 2, macro, w, label='Macro F1', color='#6366F1', edgecolor='white')
    ax.bar(x + w / 2, weighted, w, label='Weighted F1', color='#8B5CF6', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(models_s, fontsize=9, rotation=15)
    ax.set_ylim(0, 1.15)
    ax.set_title('F1 Scores', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ---- Plot 3: Per-Class F1 Heatmap ----
    ax = axes[1, 0]
    class_short = ['N', 'S', 'V', 'F', 'Q']
    f1_matrix = np.array([r['per_class_f1'] for r in results_s])
    sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=class_short, yticklabels=models_s,
                ax=ax, linewidths=0.5, vmin=0, vmax=1)
    ax.set_title('Per-Class F1 Scores', fontsize=14, fontweight='bold')

    # ---- Plot 4: Training Time ----
    ax = axes[1, 1]
    times = [r['training_time_seconds'] for r in results_s]
    bars = ax.barh(models_s, times, color=colors_s, edgecolor='white', height=0.5)
    for bar, t in zip(bars, times):
        label = f'{t:.1f}s' if t < 60 else f'{t/60:.1f}min'
        ax.text(bar.get_width() + max(times) * 0.02, bar.get_y() + bar.get_height() / 2,
                label, va='center', fontweight='bold', fontsize=11)
    ax.set_title('Training Time', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('ECG Arrhythmia Detection — Model Comparison Dashboard',
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def print_final_summary(results):
    """Print the final summary with winner."""
    sorted_results = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
    best = sorted_results[0]

    print("\n" + "=" * 70)
    print("🏆 FINAL MODEL COMPARISON SUMMARY")
    print("=" * 70)

    # Table
    print(f"\n{'Rank':<5} {'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'Time':>10}")
    print("-" * 65)
    for i, r in enumerate(sorted_results):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"{medal} {i+1}  {r['model_name']:<25} {r['test_accuracy']*100:>9.2f}% {r['macro_f1']:>10.4f} {r['training_time_display']:>10}")

    print(f"\n{'='*65}")
    print(f"🏆 WINNER: {best['model_name']} — {best['test_accuracy']*100:.2f}% Test Accuracy")
    print(f"{'='*65}")

    # Analysis
    print("\n📋 Key Observations:")
    for i, r in enumerate(sorted_results):
        print(f"  {i+1}. {r['model_name']}:")
        print(f"     - Accuracy: {r['test_accuracy']*100:.2f}%, F1: {r['macro_f1']:.4f}")
        # Find best and worst class
        best_class_idx = np.argmax(r['per_class_f1'])
        worst_class_idx = np.argmin(r['per_class_f1'])
        print(f"     - Best class: {config.CLASS_NAMES[best_class_idx]} (F1={r['per_class_f1'][best_class_idx]:.4f})")
        print(f"     - Worst class: {config.CLASS_NAMES[worst_class_idx]} (F1={r['per_class_f1'][worst_class_idx]:.4f})")


def main():
    """Generate all comparison visualizations."""
    print("\n" + "=" * 60)
    print("ECG ARRHYTHMIA — MODEL COMPARISON VISUALIZATIONS")
    print("=" * 60)

    results = load_results()
    if results is None:
        return

    print(f"\nLoaded results for {len(results)} models:")
    for r in results:
        print(f"  - {r['model_name']}: {r['test_accuracy']*100:.2f}%")

    print("\n📊 Generating comparison plots...")

    # Generate all plots
    plot_accuracy_comparison(results, COMPARISON_DIR / "accuracy_comparison.png")
    plot_f1_comparison(results, COMPARISON_DIR / "f1_comparison.png")
    plot_confusion_matrices(results, COMPARISON_DIR / "confusion_matrices.png")
    plot_training_time(results, COMPARISON_DIR / "training_time.png")
    plot_radar_chart(results, COMPARISON_DIR / "radar_chart.png")
    plot_summary_dashboard(results, COMPARISON_DIR / "summary_dashboard.png")

    # Print summary
    print_final_summary(results)

    print(f"\n✅ All visualizations saved to: {COMPARISON_DIR}")
    print("\nGenerated files:")
    for f in sorted(COMPARISON_DIR.glob("*.png")):
        print(f"  📈 {f.name}")


if __name__ == "__main__":
    main()
