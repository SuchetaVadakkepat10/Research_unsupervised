"""
Visualization script for comparing Model 1 and Model 2
Generates plots for loss, accuracy, AUROC, F1-score, and confusion matrices
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import config

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_curves(model1_history, model2_history, save_dir):
    """
    Plot training and validation curves for both models
    
    Args:
        model1_history: Training history for Model 1
        model2_history: Training history for Model 2
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Model 1: Loss
    ax = axes[0, 0]
    ax.plot(model1_history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax.plot(model1_history['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Model 1: Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Model 1: Accuracy
    ax = axes[0, 1]
    ax.plot(model1_history['train_acc'], label='Train Accuracy', linewidth=2, marker='o', markersize=4)
    ax.plot(model1_history['val_acc'], label='Val Accuracy', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model 1: Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Model 2: Classifier Loss
    ax = axes[1, 0]
    classifier_hist = model2_history['classifier']
    ax.plot(classifier_hist['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax.plot(classifier_hist['val_loss'], label='Val Loss', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Model 2: Classifier Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Model 2: Classifier Accuracy
    ax = axes[1, 1]
    ax.plot(classifier_hist['train_acc'], label='Train Accuracy', linewidth=2, marker='o', markersize=4)
    ax.plot(classifier_hist['val_acc'], label='Val Accuracy', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model 2: Classifier Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved training curves to {os.path.join(save_dir, 'training_curves.png')}")
    plt.close()


def plot_vae_loss(vae_history, save_dir):
    """
    Plot VAE training curves (reconstruction and KL loss)
    
    Args:
        vae_history: VAE training history
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Total Loss
    ax = axes[0]
    ax.plot(vae_history['train_total_loss'], label='Train Total Loss', linewidth=2, marker='o', markersize=4)
    ax.plot(vae_history['val_total_loss'], label='Val Total Loss', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('VAE: Total Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Reconstruction Loss
    ax = axes[1]
    ax.plot(vae_history['train_recon_loss'], label='Train Recon Loss', linewidth=2, marker='o', markersize=4)
    ax.plot(vae_history['val_recon_loss'], label='Val Recon Loss', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('VAE: Reconstruction Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # KL Divergence
    ax = axes[2]
    ax.plot(vae_history['train_kl_loss'], label='Train KL Loss', linewidth=2, marker='o', markersize=4)
    ax.plot(vae_history['val_kl_loss'], label='Val KL Loss', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('VAE: KL Divergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vae_training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved VAE training curves to {os.path.join(save_dir, 'vae_training_curves.png')}")
    plt.close()


def plot_metrics_comparison(model1_results, model2_results, save_dir):
    """
    Plot bar chart comparing metrics between models
    
    Args:
        model1_results: Results for Model 1
        model2_results: Results for Model 2
        save_dir: Directory to save plots
    """
    metrics = ['Accuracy', 'AUROC', 'F1-Score']
    model1_values = [
        model1_results['accuracy'],
        model1_results['auroc'] * 100,  # Convert to percentage for visualization
        model1_results['f1_macro'] * 100
    ]
    model2_values = [
        model2_results['accuracy'],
        model2_results['auroc'] * 100,
        model2_results['f1_macro'] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, model1_values, width, label='Model 1 (CNN)', alpha=0.8)
    bars2 = ax.bar(x + width/2, model2_values, width, label='Model 2 (VAE+MLP)', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: Test Set Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics comparison to {os.path.join(save_dir, 'metrics_comparison.png')}")
    plt.close()


def plot_f1_per_class(model1_results, model2_results, save_dir):
    """
    Plot per-class F1 scores for both models
    
    Args:
        model1_results: Results for Model 1
        model2_results: Results for Model 2
        save_dir: Directory to save plots
    """
    classes = config.CLASSES
    model1_f1 = [model1_results['f1_per_class'][cls] * 100 for cls in classes]
    model2_f1 = [model2_results['f1_per_class'][cls] * 100 for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, model1_f1, width, label='Model 1 (CNN)', alpha=0.8)
    bars2 = ax.bar(x + width/2, model2_f1, width, label='Model 2 (VAE+MLP)', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([cls.replace('_', ' ').title() for cls in classes], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_per_class.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved per-class F1 scores to {os.path.join(save_dir, 'f1_per_class.png')}")
    plt.close()


def plot_confusion_matrices(model1_results, model2_results, save_dir):
    """
    Plot confusion matrices for both models side by side
    
    Args:
        model1_results: Results for Model 1
        model2_results: Results for Model 2
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    classes_display = [cls.replace('_', ' ').title() for cls in config.CLASSES]
    
    # Model 1 Confusion Matrix
    cm1 = np.array(model1_results['confusion_matrix'])
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes_display,
                yticklabels=classes_display,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Model 1: Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Model 2 Confusion Matrix
    cm2 = np.array(model2_results['confusion_matrix'])
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens',
                xticklabels=classes_display,
                yticklabels=classes_display,
                ax=axes[1], cbar_kws={'label': 'Count'})
    axes[1].set_title('Model 2: Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrices to {os.path.join(save_dir, 'confusion_matrices.png')}")
    plt.close()


def plot_roc_curves(model1_results, model2_results, save_dir):
    """
    Plot ROC curves for both models (one-vs-rest for each class)
    
    Args:
        model1_results: Results for Model 1
        model2_results: Results for Model 2
        save_dir: Directory to save plots
    """
    n_classes = len(config.CLASSES)
    
    # Prepare data
    y_true = np.array(model1_results['labels'])
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    model1_probs = np.array(model1_results['probabilities'])
    model2_probs = np.array(model2_results['probabilities'])
    
    fig, axes = plt.subplots(1, n_classes, figsize=(18, 5))
    
    for i, class_name in enumerate(config.CLASSES):
        ax = axes[i]
        
        # Model 1 ROC
        fpr1, tpr1, _ = roc_curve(y_true_bin[:, i], model1_probs[:, i])
        roc_auc1 = auc(fpr1, tpr1)
        ax.plot(fpr1, tpr1, linewidth=2, label=f'Model 1 (AUC = {roc_auc1:.3f})')
        
        # Model 2 ROC
        fpr2, tpr2, _ = roc_curve(y_true_bin[:, i], model2_probs[:, i])
        roc_auc2 = auc(fpr2, tpr2)
        ax.plot(fpr2, tpr2, linewidth=2, label=f'Model 2 (AUC = {roc_auc2:.3f})')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'{class_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
    
    plt.suptitle('ROC Curves: One-vs-Rest for Each Class', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC curves to {os.path.join(save_dir, 'roc_curves.png')}")
    plt.close()


def create_summary_table(model1_results, model2_results, save_dir):
    """
    Create a summary table comparing all metrics
    
    Args:
        model1_results: Results for Model 1
        model2_results: Results for Model 2
        save_dir: Directory to save table
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    metrics = [
        ['Metric', 'Model 1 (CNN)', 'Model 2 (VAE+MLP)', 'Winner'],
        ['Accuracy (%)', f'{model1_results["accuracy"]:.2f}', f'{model2_results["accuracy"]:.2f}', 
         'Model 1' if model1_results['accuracy'] > model2_results['accuracy'] else 'Model 2'],
        ['AUROC', f'{model1_results["auroc"]:.4f}', f'{model2_results["auroc"]:.4f}',
         'Model 1' if model1_results['auroc'] > model2_results['auroc'] else 'Model 2'],
        ['F1-Score (Macro)', f'{model1_results["f1_macro"]:.4f}', f'{model2_results["f1_macro"]:.4f}',
         'Model 1' if model1_results['f1_macro'] > model2_results['f1_macro'] else 'Model 2'],
        ['Test Loss', f'{model1_results["test_loss"]:.4f}', f'{model2_results["test_loss"]:.4f}',
         'Model 1' if model1_results['test_loss'] < model2_results['test_loss'] else 'Model 2'],
    ]
    
    # Add per-class F1 scores
    for i, class_name in enumerate(config.CLASSES):
        metrics.append([
            f'F1 ({class_name.replace("_", " ").title()})',
            f'{model1_results["f1_per_class"][class_name]:.4f}',
            f'{model2_results["f1_per_class"][class_name]:.4f}',
            'Model 1' if model1_results["f1_per_class"][class_name] > model2_results["f1_per_class"][class_name] else 'Model 2'
        ])
    
    table = ax.table(cellText=metrics, cellLoc='center', loc='center',
                    colWidths=[0.3, 0.25, 0.25, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(metrics)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Complete Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, 'summary_table.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary table to {os.path.join(save_dir, 'summary_table.png')}")
    plt.close()


def visualize_all():
    """
    Main function to create all visualizations
    """
    print("\n" + "="*50)
    print("Generating Visualizations")
    print("="*50)
    
    save_dir = config.PLOTS_DIR
    
    # Load results
    print("\nLoading results...")
    with open(config.MODEL1_RESULTS, 'r') as f:
        model1_results = json.load(f)
    
    with open(config.MODEL2_RESULTS, 'r') as f:
        model2_results = json.load(f)
    
    # Load training histories
    with open(config.MODEL1_RESULTS.replace('.json', '_history.json'), 'r') as f:
        model1_history = json.load(f)
    
    with open(config.MODEL2_RESULTS.replace('.json', '_history.json'), 'r') as f:
        model2_history = json.load(f)
    
    print("\nCreating plots...")
    
    # Training curves
    plot_training_curves(model1_history, model2_history, save_dir)
    
    # VAE training curves
    plot_vae_loss(model2_history['vae'], save_dir)
    
    # Metrics comparison
    plot_metrics_comparison(model1_results, model2_results, save_dir)
    
    # Per-class F1 scores
    plot_f1_per_class(model1_results, model2_results, save_dir)
    
    # Confusion matrices
    plot_confusion_matrices(model1_results, model2_results, save_dir)
    
    # ROC curves
    plot_roc_curves(model1_results, model2_results, save_dir)
    
    # Summary table
    create_summary_table(model1_results, model2_results, save_dir)
    
    print("\n" + "="*50)
    print("All visualizations saved to:", save_dir)
    print("="*50)


if __name__ == "__main__":
    visualize_all()
