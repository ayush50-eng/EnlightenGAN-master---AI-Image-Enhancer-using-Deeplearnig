#!/usr/bin/env python3
"""
Simple test script for EnlightenGAN inference using ONNX model
"""

import cv2
import numpy as np
import os
from enlighten_inference import EnlightenOnnxModel

def test_enlighten_inference():
    """Test EnlightenGAN inference on a sample image"""
    
    # Initialize the model (will use CPU by default)
    print("Loading EnlightenGAN model...")
    model = EnlightenOnnxModel(providers=["CPUExecutionProvider"])
    print("Model loaded successfully!")
    
    # Find a test image
    test_image_path = None
    test_dir = "./test_dataset/testA/data/DICM"
    
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                test_image_path = os.path.join(test_dir, file)
                break
    
    if not test_image_path:
        print("No test image found!")
        return
    
    print(f"Processing image: {test_image_path}")
    
    # Load and process image
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"Could not load image: {test_image_path}")
        return
    
    print(f"Original image shape: {img.shape}")
    
    # Run inference
    print("Running inference...")
    enhanced_img = model.predict(img)
    
    print(f"Enhanced image shape: {enhanced_img.shape}")
    
    # Save results
    os.makedirs("./results", exist_ok=True)
    original_name = os.path.basename(test_image_path)
    name, ext = os.path.splitext(original_name)
    
    original_save_path = f"./results/{name}_original{ext}"
    enhanced_save_path = f"./results/{name}_enhanced{ext}"
    
    cv2.imwrite(original_save_path, img)
    cv2.imwrite(enhanced_save_path, enhanced_img)
    
    print(f"Results saved:")
    print(f"  Original: {original_save_path}")
    print(f"  Enhanced: {enhanced_save_path}")
    print("Inference completed successfully!")

if __name__ == "__main__":
    test_enlighten_inference()
