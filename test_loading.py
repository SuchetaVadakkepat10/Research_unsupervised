import os
from PIL import Image
import numpy as np
import config

def test_loading():
    print(f"Testing image loading from: {config.DATA_DIR}")
    print(f"Target size: {config.IMAGE_SIZE}")
    
    if not os.path.exists(config.DATA_DIR):
        print(f"Error: Directory {config.DATA_DIR} not found.")
        return
    
    images = [f for f in os.listdir(config.DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No images found.")
        return
    
    img_name = images[0]
    img_path = os.path.join(config.DATA_DIR, img_name)
    print(f"Loading image: {img_path}")
    
    img = Image.open(img_path).convert('L')
    print(f"Original size: {img.size}")
    
    img_resized = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    print(f"Resized size: {img_resized.size}")
    
    img_array = np.array(img_resized)
    print(f"Array shape: {img_array.shape}")
    print(f"Pixel range: {img_array.min()} to {img_array.max()}")
    
    print("\nImage loading test passed!")

if __name__ == "__main__":
    test_loading()
