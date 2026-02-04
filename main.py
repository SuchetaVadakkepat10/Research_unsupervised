"""
Main orchestration script for Brain MRI Classification
Executes the full pipeline: training both models, evaluation, and visualization
"""
import os
import sys
import torch
import numpy as np
import random
import config
from data_loader import create_data_loaders
from train_model1 import train_model1
from train_model2 import train_model2
from evaluate import load_and_evaluate
from visualize_results import visualize_all


def set_seeds(seed=config.RANDOM_SEED):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60)


def main():
    """Main execution pipeline"""
    
    print_header("BRAIN MRI CLASSIFICATION PROJECT")
    print(f"\nDevice: {config.DEVICE}")
    print(f"Random Seed: {config.RANDOM_SEED}")
    print(f"Classes: {', '.join(config.CLASSES)}")
    
    # Set random seeds
    set_seeds()
    
    # Check if data directory exists
    if not os.path.exists(config.DATA_DIR):
        print(f"\n❌ ERROR: Data directory '{config.DATA_DIR}' not found!")
        print("\nPlease organize your data as follows:")
        print("  data/")
        print("    ├── no_tumor/")
        print("    │   └── *.png")
        print("    ├── glioblastoma/")
        print("    │   └── *.png")
        print("    └── metastasis/")
        print("        └── *.png")
        sys.exit(1)
    
    # ==================
    # Load Data
    # ==================
    print_header("LOADING DATA")
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # ==================
    # Train Model 1
    # ==================
    print_header("TRAINING MODEL 1: END-TO-END CNN")
    
    if os.path.exists(config.MODEL1_PATH):
        print(f"\n⚠ Model 1 checkpoint found at {config.MODEL1_PATH}")
        response = input("Skip training and use existing model? (y/n): ").strip().lower()
        if response == 'y':
            print("✓ Skipping Model 1 training")
        else:
            model1, history1 = train_model1(train_loader, val_loader)
    else:
        model1, history1 = train_model1(train_loader, val_loader)
    
    # ==================
    # Train Model 2
    # ==================
    print_header("TRAINING MODEL 2: VAE + MLP CLASSIFIER")
    
    if os.path.exists(config.VAE_PATH) and os.path.exists(config.CLASSIFIER_PATH):
        print(f"\n⚠ Model 2 checkpoints found:")
        print(f"  - VAE: {config.VAE_PATH}")
        print(f"  - Classifier: {config.CLASSIFIER_PATH}")
        response = input("Skip training and use existing models? (y/n): ").strip().lower()
        if response == 'y':
            print("✓ Skipping Model 2 training")
        else:
            vae, classifier, history2 = train_model2(train_loader, val_loader)
    else:
        vae, classifier, history2 = train_model2(train_loader, val_loader)
    
    # ==================
    # Evaluate Models
    # ==================
    print_header("EVALUATING MODELS")
    model1_results, model2_results = load_and_evaluate()
    
    # ==================
    # Generate Visualizations
    # ==================
    print_header("GENERATING VISUALIZATIONS")
    visualize_all()
    
    # ==================
    # Final Summary
    # ==================
    print_header("PIPELINE COMPLETED SUCCESSFULLY")
    
    print("\n📁 Generated Files:")
    print(f"  Models:")
    print(f"    - {config.MODEL1_PATH}")
    print(f"    - {config.VAE_PATH}")
    print(f"    - {config.CLASSIFIER_PATH}")
    
    print(f"\n  Results:")
    print(f"    - {config.MODEL1_RESULTS}")
    print(f"    - {config.MODEL2_RESULTS}")
    
    print(f"\n  Visualizations:")
    print(f"    - {config.PLOTS_DIR}/")
    
    print("\n📊 Final Comparison:")
    print(f"  {'Metric':<20} {'Model 1':<15} {'Model 2':<15}")
    print("  " + "-"*50)
    print(f"  {'Accuracy (%)':<20} {model1_results['accuracy']:<15.2f} {model2_results['accuracy']:<15.2f}")
    print(f"  {'AUROC':<20} {model1_results['auroc']:<15.4f} {model2_results['auroc']:<15.4f}")
    print(f"  {'F1-Score (Macro)':<20} {model1_results['f1_macro']:<15.4f} {model2_results['f1_macro']:<15.4f}")
    
    # Determine winner
    model1_wins = 0
    model2_wins = 0
    
    if model1_results['accuracy'] > model2_results['accuracy']:
        model1_wins += 1
    else:
        model2_wins += 1
    
    if model1_results['auroc'] > model2_results['auroc']:
        model1_wins += 1
    else:
        model2_wins += 1
    
    if model1_results['f1_macro'] > model2_results['f1_macro']:
        model1_wins += 1
    else:
        model2_wins += 1
    
    print("\n🏆 Winner:")
    if model1_wins > model2_wins:
        print("  Model 1 (End-to-end CNN) performs better overall")
    elif model2_wins > model1_wins:
        print("  Model 2 (VAE + MLP) performs better overall")
    else:
        print("  Both models perform similarly")
    
    print("\n" + "="*60)
    print("Thank you for using the Brain MRI Classification Pipeline!")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
