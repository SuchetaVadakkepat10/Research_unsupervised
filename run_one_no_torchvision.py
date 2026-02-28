"""
Lightweight runner that doesn't depend on torchvision.
Opens the first image from the `premo` folder, resizes with PIL,
converts to a torch tensor in [0,1], and runs both models.
"""
import os
import sys
import numpy as np
from PIL import Image
import torch

import config
from model1_cnn import get_model1
from model2_vae import get_model2_vae, get_model2_classifier


def run_pipeline_for_one_no_torchvision():
    print("="*60)
    print("RUNNING BRAIN MRI MODELS FOR ONE IMAGE (no torchvision)")
    print("="*60)

    if not os.path.isdir(config.DATA_DIR):
        print(f"Error: data directory '{config.DATA_DIR}' not found")
        return

    images = [f for f in os.listdir(config.DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print(f"Error: No images found in {config.DATA_DIR}")
        return

    img_name = images[0]
    img_path = os.path.join(config.DATA_DIR, img_name)
    print(f"\n[1] Processing Image: {img_name}")

    try:
        img = Image.open(img_path).convert('L')
        img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), resample=Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        tensor = tensor.to(config.DEVICE)
        print(f"    - Input tensor shape: {tensor.shape}")
        print(f"    - Value range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    except Exception as e:
        print(f"    - Error loading image: {e}")
        return

    results = {}

    # Model 1
    print(f"\n[2] Running Model 1: End-to-End CNN")
    try:
        model1 = get_model1()
        model1.eval()
        with torch.no_grad():
            logits1 = model1(tensor)
            probs1 = torch.softmax(logits1, dim=1)
            pred1 = torch.argmax(probs1, dim=1).item()
            results['model1'] = {
                'prediction': config.CLASSES[pred1],
                'confidence': probs1[0][pred1].item()
            }
        print(f"    - Prediction: {results['model1']['prediction']} ({results['model1']['confidence']:.2%})")
    except Exception as e:
        print(f"    - Error running Model 1: {e}")

    # Model 2
    print(f"\n[3] Running Model 2: VAE + MLP Classifier")
    try:
        vae = get_model2_vae()
        vae.eval()
        classifier = get_model2_classifier()
        classifier.eval()

        with torch.no_grad():
            reconstruction, mu, logvar = vae(tensor)
            embedding = vae.get_embedding(tensor)
            logits2 = classifier(embedding)
            probs2 = torch.softmax(logits2, dim=1)
            pred2 = torch.argmax(probs2, dim=1).item()

            recon_error = torch.nn.functional.mse_loss(reconstruction, tensor).item()

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


if __name__ == '__main__':
    run_pipeline_for_one_no_torchvision()
