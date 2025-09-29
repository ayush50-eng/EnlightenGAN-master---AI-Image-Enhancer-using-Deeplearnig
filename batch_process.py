#!/usr/bin/env python3
"""
Batch Processing Script for EnlightenGAN
Process multiple images at once
"""

import os
import cv2
import glob
from enlighten_inference import EnlightenOnnxModel
import argparse
from tqdm import tqdm

def process_batch(input_dir, output_dir, file_pattern="*.jpg"):
    """Process all images in a directory"""
    
    # Initialize model
    print("Loading EnlightenGAN model...")
    model = EnlightenOnnxModel(providers=["CPUExecutionProvider"])
    print("Model loaded successfully!")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    image_files = glob.glob(os.path.join(input_dir, file_pattern))
    image_files.extend(glob.glob(os.path.join(input_dir, "*.png")))
    image_files.extend(glob.glob(os.path.join(input_dir, "*.jpeg")))
    image_files.extend(glob.glob(os.path.join(input_dir, "*.bmp")))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load {image_path}")
                continue
            
            # Run inference
            enhanced_img = model.predict(img)
            
            # Save result
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_enhanced{ext}")
            
            cv2.imwrite(output_path, enhanced_img)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    print(f"Batch processing complete! Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process images with EnlightenGAN')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing images')
    parser.add_argument('--output', '-o', required=True, help='Output directory for enhanced images')
    parser.add_argument('--pattern', '-p', default='*.jpg', help='File pattern to match (default: *.jpg)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input directory {args.input} does not exist!")
        exit(1)
    
    process_batch(args.input, args.output, args.pattern)
