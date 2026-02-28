"""
Training script for Model 1: End-to-end CNN Classifier
Trains with cross-entropy loss in a supervised fashion
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import csv
import config
from model1_cnn import get_model1
from data_loader import create_data_loaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_model1(train_loader, val_loader, num_epochs=config.MODEL1_EPOCHS):
    """
    Train Model 1: End-to-end CNN
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
    
    Returns:
        model: Trained model
        history: Training history dictionary
    """
    print("\n" + "="*50)
    print("Training Model 1: End-to-end CNN Classifier")
    print("="*50)
    
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Create model
    model = get_model1(num_classes=config.NUM_CLASSES, device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.MODEL1_LR,
        weight_decay=config.MODEL1_WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Initialize CSV log
    with open(config.MODEL1_LOSS_LOG, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Log to CSV
        with open(config.MODEL1_LOSS_LOG, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, config.MODEL1_PATH)
            print(f"✓ Saved new best model (Val Acc: {val_acc:.2f}%)")
    
    print(f"\nTraining completed! Best Val Acc: {best_val_acc:.2f}%")
    
    # Load best model
    checkpoint = torch.load(config.MODEL1_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save training history
    with open(config.MODEL1_RESULTS.replace('.json', '_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # Train model
    model, history = train_model1(train_loader, val_loader)
    
    print("\nModel saved to:", config.MODEL1_PATH)
