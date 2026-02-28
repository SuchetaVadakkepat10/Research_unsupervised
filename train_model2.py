"""
Training script for Model 2: VAE + MLP Classifier
Stage 1: Train VAE with reconstruction + KL divergence loss
Stage 2: Freeze VAE, train MLP classifier with cross-entropy loss
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import csv
import config
from model2_vae import get_model2_vae, get_model2_classifier, vae_loss
from data_loader import create_data_loaders


def train_vae_epoch(vae, train_loader, optimizer, device, kl_weight):
    """Train VAE for one epoch"""
    vae.train()
    running_total_loss = 0.0
    running_recon_loss = 0.0
    running_kl_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc="Training VAE", leave=False)
    for images, _ in pbar:  # Ignore labels for unsupervised training
        images = images.to(device)
        batch_size = images.size(0)
        
        # Forward pass
        optimizer.zero_grad()
        reconstruction, mu, logvar = vae(images)
        
        # Compute loss
        total_loss, recon_loss, kl_loss = vae_loss(
            reconstruction, images, mu, logvar, kl_weight=kl_weight
        )
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Statistics
        running_total_loss += total_loss.item() * batch_size
        running_recon_loss += recon_loss.item() * batch_size
        running_kl_loss += kl_loss.item() * batch_size
        total_samples += batch_size
        
        # Update progress bar
        pbar.set_postfix({
            'total': f'{total_loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })
    
    epoch_total_loss = running_total_loss / total_samples
    epoch_recon_loss = running_recon_loss / total_samples
    epoch_kl_loss = running_kl_loss / total_samples
    
    return epoch_total_loss, epoch_recon_loss, epoch_kl_loss


def validate_vae(vae, val_loader, device, kl_weight):
    """Validate VAE"""
    vae.eval()
    running_total_loss = 0.0
    running_recon_loss = 0.0
    running_kl_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating VAE", leave=False)
        for images, _ in pbar:
            images = images.to(device)
            batch_size = images.size(0)
            
            # Forward pass
            reconstruction, mu, logvar = vae(images)
            
            # Compute loss
            total_loss, recon_loss, kl_loss = vae_loss(
                reconstruction, images, mu, logvar, kl_weight=kl_weight
            )
            
            # Statistics
            running_total_loss += total_loss.item() * batch_size
            running_recon_loss += recon_loss.item() * batch_size
            running_kl_loss += kl_loss.item() * batch_size
            total_samples += batch_size
            
            pbar.set_postfix({
                'total': f'{total_loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })
    
    epoch_total_loss = running_total_loss / total_samples
    epoch_recon_loss = running_recon_loss / total_samples
    epoch_kl_loss = running_kl_loss / total_samples
    
    return epoch_total_loss, epoch_recon_loss, epoch_kl_loss


def train_classifier_epoch(vae, classifier, train_loader, criterion, optimizer, device):
    """Train classifier for one epoch with frozen VAE"""
    vae.eval()  # VAE in eval mode
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training Classifier", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Get embeddings from frozen VAE
        with torch.no_grad():
            embeddings = vae.get_embedding(images)
        
        # Forward pass through classifier
        optimizer.zero_grad()
        outputs = classifier(embeddings)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate_classifier(vae, classifier, val_loader, criterion, device):
    """Validate classifier"""
    vae.eval()
    classifier.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating Classifier", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Get embeddings from VAE
            embeddings = vae.get_embedding(images)
            
            # Forward pass through classifier
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_model2(train_loader, val_loader, 
                 vae_epochs=config.VAE_EPOCHS,
                 classifier_epochs=config.CLASSIFIER_EPOCHS):
    """
    Train Model 2: VAE + MLP Classifier (two-stage training)
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        vae_epochs: Number of epochs for VAE training
        classifier_epochs: Number of epochs for classifier training
    
    Returns:
        vae: Trained VAE model
        classifier: Trained classifier model
        history: Training history dictionary
    """
    print("\n" + "="*50)
    print("Training Model 2: VAE + MLP Classifier")
    print("="*50)
    
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # ==================
    # Stage 1: Train VAE
    # ==================
    print("\n" + "-"*50)
    print("Stage 1: Training VAE (Unsupervised)")
    print("-"*50)
    
    vae = get_model2_vae(latent_dim=config.LATENT_DIM, device=device)
    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    
    vae_optimizer = optim.Adam(
        vae.parameters(),
        lr=config.VAE_LR,
        weight_decay=config.VAE_WEIGHT_DECAY
    )
    
    vae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        vae_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    vae_history = {
        'train_total_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'val_total_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': []
    }
    
    best_vae_loss = float('inf')
    
    # Initialize VAE CSV log
    with open(config.VAE_LOSS_LOG, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_total', 'train_recon', 'train_kl', 'val_total', 'val_recon', 'val_kl'])
    
    for epoch in range(vae_epochs):
        print(f"\nVAE Epoch {epoch+1}/{vae_epochs}")
        print("-" * 30)
        
        # Train
        train_total, train_recon, train_kl = train_vae_epoch(
            vae, train_loader, vae_optimizer, device, config.KL_WEIGHT
        )
        
        # Validate
        val_total, val_recon, val_kl = validate_vae(
            vae, val_loader, device, config.KL_WEIGHT
        )
        
        # Update learning rate
        vae_scheduler.step(val_total)
        
        # Save history
        vae_history['train_total_loss'].append(train_total)
        vae_history['train_recon_loss'].append(train_recon)
        vae_history['train_kl_loss'].append(train_kl)
        vae_history['val_total_loss'].append(val_total)
        vae_history['val_recon_loss'].append(val_recon)
        vae_history['val_kl_loss'].append(val_kl)
        
        # Log VAE losses to CSV
        with open(config.VAE_LOSS_LOG, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_total, train_recon, train_kl, val_total, val_recon, val_kl])
        
        # Print summary
        print(f"Train - Total: {train_total:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        print(f"Val   - Total: {val_total:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}")
        
        # Save best model
        if val_total < best_vae_loss:
            best_vae_loss = val_total
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': vae_optimizer.state_dict(),
                'val_loss': val_total,
            }, config.VAE_PATH)
            print(f"✓ Saved new best VAE (Val Loss: {val_total:.4f})")
    
    print(f"\nVAE Training completed! Best Val Loss: {best_vae_loss:.4f}")
    
    # Load best VAE
    checkpoint = torch.load(config.VAE_PATH)
    vae.load_state_dict(checkpoint['model_state_dict'])
    
    # Freeze VAE
    for param in vae.parameters():
        param.requires_grad = False
    print("\n✓ VAE frozen for downstream training")
    
    # =============================
    # Stage 2: Train MLP Classifier
    # =============================
    print("\n" + "-"*50)
    print("Stage 2: Training MLP Classifier (Supervised)")
    print("-"*50)
    
    classifier = get_model2_classifier(
        input_dim=config.LATENT_DIM,
        num_classes=config.NUM_CLASSES,
        device=device
    )
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    classifier_optimizer = optim.Adam(
        classifier.parameters(),
        lr=config.CLASSIFIER_LR,
        weight_decay=config.CLASSIFIER_WEIGHT_DECAY
    )
    
    classifier_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        classifier_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    classifier_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_classifier_acc = 0.0
    
    # Initialize Classifier CSV log
    with open(config.CLASSIFIER_LOSS_LOG, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    
    for epoch in range(classifier_epochs):
        print(f"\nClassifier Epoch {epoch+1}/{classifier_epochs}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_classifier_epoch(
            vae, classifier, train_loader, criterion, classifier_optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate_classifier(
            vae, classifier, val_loader, criterion, device
        )
        
        # Update learning rate
        classifier_scheduler.step(val_loss)
        
        # Save history
        classifier_history['train_loss'].append(train_loss)
        classifier_history['train_acc'].append(train_acc)
        classifier_history['val_loss'].append(val_loss)
        classifier_history['val_acc'].append(val_acc)
        
        # Log Classifier losses to CSV
        with open(config.CLASSIFIER_LOSS_LOG, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])
        
        # Print summary
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_classifier_acc:
            best_classifier_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': classifier_optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, config.CLASSIFIER_PATH)
            print(f"✓ Saved new best classifier (Val Acc: {val_acc:.2f}%)")
    
    print(f"\nClassifier Training completed! Best Val Acc: {best_classifier_acc:.2f}%")
    
    # Load best classifier
    checkpoint = torch.load(config.CLASSIFIER_PATH)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    
    # Combine histories
    history = {
        'vae': vae_history,
        'classifier': classifier_history
    }
    
    # Save training history
    with open(config.MODEL2_RESULTS.replace('.json', '_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return vae, classifier, history


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # Train model
    vae, classifier, history = train_model2(train_loader, val_loader)
    
    print("\nVAE saved to:", config.VAE_PATH)
    print("Classifier saved to:", config.CLASSIFIER_PATH)
