#!/usr/bin/env python3
"""
Test Enhancement Quality Script
Compare different enhancement methods and parameters
"""

import cv2
import numpy as np
import os
from enlighten_inference import EnlightenOnnxModel
import matplotlib.pyplot as plt

def analyze_image_brightness(img):
    """Analyze image brightness and contrast"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    stats = {
        'mean_brightness': np.mean(gray),
        'std_brightness': np.std(gray),
        'min_brightness': np.min(gray),
        'max_brightness': np.max(gray),
        'contrast_ratio': np.max(gray) / (np.min(gray) + 1e-6)
    }
    
    return stats

def enhance_with_clahe(img, clip_limit=2.0):
    """Enhance using CLAHE"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def enhance_with_gamma(img, gamma=0.7):
    """Enhance using gamma correction"""
    enhanced = np.power(img / 255.0, gamma) * 255.0
    return np.uint8(enhanced)

def enhance_with_histogram_equalization(img):
    """Enhance using histogram equalization"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    l = cv2.equalizeHist(l)
    
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def test_enhancement_methods(image_path):
    """Test different enhancement methods on an image"""
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Testing enhancement methods on: {os.path.basename(image_path)}")
    print(f"Image shape: {img.shape}")
    
    # Analyze original image
    original_stats = analyze_image_brightness(img)
    print(f"\nOriginal image stats:")
    print(f"  Mean brightness: {original_stats['mean_brightness']:.2f}")
    print(f"  Contrast ratio: {original_stats['contrast_ratio']:.2f}")
    print(f"  Brightness range: {original_stats['min_brightness']}-{original_stats['max_brightness']}")
    
    # Initialize EnlightenGAN model
    print("\nLoading EnlightenGAN model...")
    model = EnlightenOnnxModel(providers=["CPUExecutionProvider"])
    print("Model loaded successfully!")
    
    # Test different methods
    methods = {
        'Original': img,
        'EnlightenGAN': model.predict(img),
        'CLAHE (clip=2.0)': enhance_with_clahe(img, 2.0),
        'CLAHE (clip=3.0)': enhance_with_clahe(img, 3.0),
        'Gamma (0.7)': enhance_with_gamma(img, 0.7),
        'Gamma (0.8)': enhance_with_gamma(img, 0.8),
        'Histogram Equalization': enhance_with_histogram_equalization(img),
    }
    
    # Analyze each method
    results = {}
    for method_name, enhanced_img in methods.items():
        stats = analyze_image_brightness(enhanced_img)
        results[method_name] = {
            'image': enhanced_img,
            'stats': stats
        }
        
        print(f"\n{method_name}:")
        print(f"  Mean brightness: {stats['mean_brightness']:.2f}")
        print(f"  Contrast ratio: {stats['contrast_ratio']:.2f}")
        print(f"  Brightness range: {stats['min_brightness']}-{stats['max_brightness']}")
    
    # Save results
    output_dir = 'enhancement_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    for method_name, result in results.items():
        output_path = os.path.join(output_dir, f"{base_name}_{method_name.replace(' ', '_').replace('(', '').replace(')', '')}.jpg")
        cv2.imwrite(output_path, result['image'])
        print(f"Saved: {output_path}")
    
    # Find best method based on brightness improvement
    original_brightness = original_stats['mean_brightness']
    best_method = None
    best_improvement = 0
    
    for method_name, result in results.items():
        if method_name == 'Original':
            continue
        
        improvement = result['stats']['mean_brightness'] - original_brightness
        if improvement > best_improvement:
            best_improvement = improvement
            best_method = method_name
    
    print(f"\n🎯 Best enhancement method: {best_method}")
    print(f"   Brightness improvement: +{best_improvement:.2f}")
    
    return results

if __name__ == "__main__":
    # Test with a sample image
    test_image = "./test_dataset/testA/data/DICM/01.jpg"
    
    if os.path.exists(test_image):
        test_enhancement_methods(test_image)
    else:
        print(f"Test image not found: {test_image}")
        print("Please provide a valid image path.")
