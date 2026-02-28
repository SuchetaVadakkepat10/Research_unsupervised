"""
Script to run both Brain MRI Classification models for a single image.
"""
import os
import sys
import numpy as np
from PIL import Image

# Import config (which now handles torch import robustly)
import config

def run_pipeline_for_one():
    print("="*60)
    print("RUNNING BRAIN MRI MODELS FOR ONE IMAGE")
    print("="*60)
    
    # Check if torch is available
    if config.torch is None:
        print("\n❌ ERROR: PyTorch is not functional on this system (DLL load failed).")
        print("Please ensure VC++ Redistributable and MKL dependencies are installed.")
        return
    
    import torch
    from torchvision import transforms
    from data_loader import get_data_transforms
    from model1_cnn import get_model1
    from model2_vae import get_model2_vae, get_model2_classifier
    
    # Selection of image
    premo_dir = config.DATA_DIR
    images = [f for f in os.listdir(premo_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print(f"Error: No images found in {premo_dir}")
        return
    
    img_name = images[0]
    img_path = os.path.join(premo_dir, img_name)
    print(f"\n[1] Processing Image: {img_name}")
    
    # Load and transform
    try:
        image = Image.open(img_path).convert('L')
        transform = get_data_transforms(augment=False)
        image_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
        print(f"    - Input tensor shape: {image_tensor.shape}")
        print(f"    - Value range: [{image_tensor.min():.4f}, {image_tensor.max():.4f}]")
    except Exception as e:
        print(f"    - Error loading image: {e}")
        return

    # Results dictionary
    results = {}

    # MODEL 1: CNN
    print(f"\n[2] Running Model 1: End-to-End CNN")
    try:
        model1 = get_model1()
        model1.eval()
        with torch.no_grad():
            logits1 = model1(image_tensor)
            probs1 = torch.softmax(logits1, dim=1)
            pred1 = torch.argmax(probs1, dim=1).item()
            results['model1'] = {
                'prediction': config.CLASSES[pred1],
                'confidence': probs1[0][pred1].item()
            }
        print(f"    - Prediction: {results['model1']['prediction']} ({results['model1']['confidence']:.2%})")
    except Exception as e:
        print(f"    - Error running Model 1: {e}")

    # MODEL 2: VAE + Classifier
    print(f"\n[3] Running Model 2: VAE + MLP Classifier")
    try:
        vae = get_model2_vae()
        vae.eval()
        classifier = get_model2_classifier()
        classifier.eval()
        
        with torch.no_grad():
            reconstruction, mu, logvar = vae(image_tensor)
            embedding = vae.get_embedding(image_tensor)
            logits2 = classifier(embedding)
            probs2 = torch.softmax(logits2, dim=1)
            pred2 = torch.argmax(probs2, dim=1).item()
            
            # Reconstruction quality (MSE)
            recon_error = torch.nn.functional.mse_loss(reconstruction, image_tensor).item()
            
            results['model2'] = {
                'prediction': config.CLASSES[pred2],
                'confidence': probs2[0][pred2].item(),
                'recon_error': recon_error
            }
        print(f"    - Prediction: {results['model2']['prediction']} ({results['model2']['confidence']:.2%})")
        print(f"    - VAE Recon Error: {results['model2']['recon_error']:.6f}")
    except Exception as e:
        print(f"    - Error running Model 2: {e}")

    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    if 'model1' in results:
        print(f"Model 1: {results['model1']['prediction']:<15} (Conf: {results['model1']['confidence']:.2%})")
    if 'model2' in results:
        print(f"Model 2: {results['model2']['prediction']:<15} (Conf: {results['model2']['confidence']:.2%})")
    print("="*60)

if __name__ == "__main__":
    run_pipeline_for_one()
