"""
Model 2: VAE Encoder + Supervised MLP Classifier
First trains a VAE with reconstruction + KL divergence loss,
then trains an MLP classifier on the frozen encoder embeddings
"""
import torch
import torch.nn as nn
import config


class VAEEncoder(nn.Module):
    """
    Variational Autoencoder for unsupervised representation learning
    """
    
    def __init__(self, latent_dim=config.LATENT_DIM):
        super(VAEEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder: 128x128 -> latent_dim
        self.encoder = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Calculate flattened size: 256 * 8 * 8 = 16384
        self.flatten_size = 256 * 8 * 8
        
        # Latent space projection
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder: latent_dim -> 128x128
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in range [-1, 1] to match normalized input
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        """
        Encode input to latent distribution parameters
        
        Args:
            x: Input tensor of shape (batch_size, 1, 128, 128)
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent vector to reconstructed image
        
        Args:
            z: Latent vector
        
        Returns:
            reconstruction: Reconstructed image
        """
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 8, 8)  # Reshape for decoder
        reconstruction = self.decoder(h)
        return reconstruction
    
    def forward(self, x):
        """
        Forward pass through VAE
        
        Args:
            x: Input tensor
        
        Returns:
            reconstruction: Reconstructed image
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def get_embedding(self, x):
        """
        Get latent embedding for input (for downstream classifier)
        
        Args:
            x: Input tensor
        
        Returns:
            z: Latent embedding (using mean, no sampling)
        """
        mu, _ = self.encode(x)
        return mu


class MLPClassifier(nn.Module):
    """
    MLP Classifier that takes frozen VAE embeddings as input
    """
    
    def __init__(self, input_dim=config.LATENT_DIM, num_classes=config.NUM_CLASSES):
        super(MLPClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input embeddings of shape (batch_size, input_dim)
        
        Returns:
            logits: Output logits of shape (batch_size, num_classes)
        """
        return self.classifier(x)


def vae_loss(reconstruction, original, mu, logvar, kl_weight=config.KL_WEIGHT):
    """
    VAE loss = Reconstruction loss + KL divergence
    
    Args:
        reconstruction: Reconstructed images
        original: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        kl_weight: Weight for KL divergence term
    
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence component
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(reconstruction, original, reduction='mean')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / original.size(0)  # Average over batch
    
    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss


def get_model2_vae(latent_dim=config.LATENT_DIM, device=config.DEVICE):
    """
    Create and return VAE model
    
    Args:
        latent_dim: Dimension of latent space
        device: Device to place the model on
    
    Returns:
        vae: VAEEncoder model
    """
    vae = VAEEncoder(latent_dim=latent_dim)
    vae = vae.to(device)
    return vae


def get_model2_classifier(input_dim=config.LATENT_DIM, num_classes=config.NUM_CLASSES, device=config.DEVICE):
    """
    Create and return MLP classifier for Model 2
    
    Args:
        input_dim: Input dimension (latent dimension from VAE)
        num_classes: Number of output classes
        device: Device to place the model on
    
    Returns:
        classifier: MLPClassifier model
    """
    classifier = MLPClassifier(input_dim=input_dim, num_classes=num_classes)
    classifier = classifier.to(device)
    return classifier


if __name__ == "__main__":
    # Test the models
    print("Testing Model 2: VAE + MLP Classifier")
    
    # Test VAE
    vae = get_model2_vae()
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)
    
    with torch.no_grad():
        reconstruction, mu, logvar = vae(dummy_input)
        embedding = vae.get_embedding(dummy_input)
    
    print(f"\nVAE:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Reconstruction shape: {reconstruction.shape}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Number of parameters: {sum(p.numel() for p in vae.parameters()):,}")
    
    # Test MLP Classifier
    classifier = get_model2_classifier()
    with torch.no_grad():
        output = classifier(embedding)
    
    print(f"\nMLP Classifier:")
    print(f"  Input shape: {embedding.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of parameters: {sum(p.numel() for p in classifier.parameters()):,}")
