"""
Evaluation script for both models
Computes accuracy, loss, AUROC, and F1-score on test set
"""
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
import numpy as np
import json
from tqdm import tqdm
import config
from model1_cnn import get_model1
from model2_vae import get_model2_vae, get_model2_classifier


def evaluate_model1(model, test_loader, device):
    """
    Evaluate Model 1 on test set
    
    Args:
        model: Trained Model 1
        test_loader: Test data loader
        device: Device to run evaluation on
    
    Returns:
        results: Dictionary containing all metrics
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_labels = []
    all_preds = []
    all_probs = []
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating Model 1")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            running_loss += loss.item() * images.size(0)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    test_loss = running_loss / len(test_loader.dataset)
    accuracy = 100.0 * np.mean(all_labels == all_preds)
    
    # AUROC (one-vs-rest for multiclass)
    auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    
    # F1-score (macro average)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    # Per-class F1 scores
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=config.CLASSES,
        output_dict=True
    )
    
    results = {
        'model': 'Model 1 (End-to-end CNN)',
        'test_loss': float(test_loss),
        'accuracy': float(accuracy),
        'auroc': float(auroc),
        'f1_macro': float(f1_macro),
        'f1_per_class': {
            config.CLASSES[i]: float(f1_per_class[i])
            for i in range(len(config.CLASSES))
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': all_preds.tolist(),
        'probabilities': all_probs.tolist(),
        'labels': all_labels.tolist()
    }
    
    return results


def evaluate_model2(vae, classifier, test_loader, device):
    """
    Evaluate Model 2 on test set
    
    Args:
        vae: Trained VAE encoder
        classifier: Trained MLP classifier
        test_loader: Test data loader
        device: Device to run evaluation on
    
    Returns:
        results: Dictionary containing all metrics
    """
    vae.eval()
    classifier.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_labels = []
    all_preds = []
    all_probs = []
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating Model 2")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Get embeddings from VAE
            embeddings = vae.get_embedding(images)
            
            # Forward pass through classifier
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            running_loss += loss.item() * images.size(0)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    test_loss = running_loss / len(test_loader.dataset)
    accuracy = 100.0 * np.mean(all_labels == all_preds)
    
    # AUROC (one-vs-rest for multiclass)
    auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    
    # F1-score (macro average)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    # Per-class F1 scores
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=config.CLASSES,
        output_dict=True
    )
    
    results = {
        'model': 'Model 2 (VAE + MLP)',
        'test_loss': float(test_loss),
        'accuracy': float(accuracy),
        'auroc': float(auroc),
        'f1_macro': float(f1_macro),
        'f1_per_class': {
            config.CLASSES[i]: float(f1_per_class[i])
            for i in range(len(config.CLASSES))
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': all_preds.tolist(),
        'probabilities': all_probs.tolist(),
        'labels': all_labels.tolist()
    }
    
    return results


def load_and_evaluate():
    """
    Load trained models and evaluate on test set
    
    Returns:
        model1_results: Results for Model 1
        model2_results: Results for Model 2
    """
    print("\n" + "="*50)
    print("Evaluating Models on Test Set")
    print("="*50)
    
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Load data
    from data_loader import create_data_loaders
    print("\nLoading data...")
    _, _, test_loader = create_data_loaders()
    print(f"Test set size: {len(test_loader.dataset)}")
    
    # ==================
    # Evaluate Model 1
    # ==================
    print("\n" + "-"*50)
    print("Model 1: End-to-end CNN")
    print("-"*50)
    
    model1 = get_model1(num_classes=config.NUM_CLASSES, device=device)
    checkpoint = torch.load(config.MODEL1_PATH, map_location=device)
    model1.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Loaded trained Model 1")
    
    model1_results = evaluate_model1(model1, test_loader, device)
    
    print(f"\nModel 1 Results:")
    print(f"  Accuracy: {model1_results['accuracy']:.2f}%")
    print(f"  AUROC:    {model1_results['auroc']:.4f}")
    print(f"  F1-Score: {model1_results['f1_macro']:.4f}")
    print(f"  Test Loss: {model1_results['test_loss']:.4f}")
    
    # Save results
    with open(config.MODEL1_RESULTS, 'w') as f:
        json.dump(model1_results, f, indent=2)
    print(f"✓ Saved results to {config.MODEL1_RESULTS}")
    
    # ==================
    # Evaluate Model 2
    # ==================
    print("\n" + "-"*50)
    print("Model 2: VAE + MLP Classifier")
    print("-"*50)
    
    vae = get_model2_vae(latent_dim=config.LATENT_DIM, device=device)
    vae_checkpoint = torch.load(config.VAE_PATH, map_location=device)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    
    classifier = get_model2_classifier(
        input_dim=config.LATENT_DIM,
        num_classes=config.NUM_CLASSES,
        device=device
    )
    classifier_checkpoint = torch.load(config.CLASSIFIER_PATH, map_location=device)
    classifier.load_state_dict(classifier_checkpoint['model_state_dict'])
    print("✓ Loaded trained Model 2 (VAE + Classifier)")
    
    model2_results = evaluate_model2(vae, classifier, test_loader, device)
    
    print(f"\nModel 2 Results:")
    print(f"  Accuracy: {model2_results['accuracy']:.2f}%")
    print(f"  AUROC:    {model2_results['auroc']:.4f}")
    print(f"  F1-Score: {model2_results['f1_macro']:.4f}")
    print(f"  Test Loss: {model2_results['test_loss']:.4f}")
    
    # Save results
    with open(config.MODEL2_RESULTS, 'w') as f:
        json.dump(model2_results, f, indent=2)
    print(f"✓ Saved results to {config.MODEL2_RESULTS}")
    
    # ==================
    # Comparison Summary
    # ==================
    print("\n" + "="*50)
    print("Comparison Summary")
    print("="*50)
    print(f"\n{'Metric':<15} {'Model 1':<15} {'Model 2':<15}")
    print("-" * 45)
    print(f"{'Accuracy (%)':<15} {model1_results['accuracy']:<15.2f} {model2_results['accuracy']:<15.2f}")
    print(f"{'AUROC':<15} {model1_results['auroc']:<15.4f} {model2_results['auroc']:<15.4f}")
    print(f"{'F1-Score':<15} {model1_results['f1_macro']:<15.4f} {model2_results['f1_macro']:<15.4f}")
    print(f"{'Test Loss':<15} {model1_results['test_loss']:<15.4f} {model2_results['test_loss']:<15.4f}")
    
    return model1_results, model2_results


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    # Evaluate both models
    model1_results, model2_results = load_and_evaluate()
