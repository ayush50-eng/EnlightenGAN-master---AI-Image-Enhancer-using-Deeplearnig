#!/usr/bin/env python3
"""
Improved EnlightenGAN Web Interface
Better enhancement with multiple techniques and adaptive processing
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from enlighten_inference import EnlightenOnnxModel
import base64
from io import BytesIO
from PIL import Image, ImageEnhance
import argparse

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the EnlightenGAN model
print("Loading EnlightenGAN model...")
model = EnlightenOnnxModel(providers=["CPUExecutionProvider"])
print("Model loaded successfully!")

# Create uploads and results directories
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_array):
    """Convert numpy array to base64 string for web display"""
    # Convert BGR to RGB for web display
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_array
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Convert to base64
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG', quality=95)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{img_str}"

def analyze_image_darkness(img):
    """Analyze how dark an image is"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 30:
        return "very_dark"
    elif mean_brightness < 60:
        return "dark"
    elif mean_brightness < 100:
        return "moderate"
    else:
        return "bright"

def enhance_with_adaptive_clahe(img, darkness_level):
    """Adaptive CLAHE based on image darkness"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Adjust CLAHE parameters based on darkness
    if darkness_level == "very_dark":
        clip_limit = 4.0
        tile_size = (4, 4)
    elif darkness_level == "dark":
        clip_limit = 3.0
        tile_size = (6, 6)
    elif darkness_level == "moderate":
        clip_limit = 2.0
        tile_size = (8, 8)
    else:
        clip_limit = 1.5
        tile_size = (8, 8)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l = clahe.apply(l)
    
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def enhance_with_adaptive_gamma(img, darkness_level):
    """Adaptive gamma correction based on image darkness"""
    if darkness_level == "very_dark":
        gamma = 0.5
    elif darkness_level == "dark":
        gamma = 0.6
    elif darkness_level == "moderate":
        gamma = 0.7
    else:
        gamma = 0.8
    
    enhanced = np.power(img / 255.0, gamma) * 255.0
    return np.uint8(enhanced)

def enhance_with_histogram_equalization(img):
    """Histogram equalization in LAB color space"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    l = cv2.equalizeHist(l)
    
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def enhance_with_retinex(img):
    """Simple Retinex algorithm for illumination correction"""
    # Convert to float
    img_float = img.astype(np.float32) / 255.0
    
    # Apply Gaussian blur to estimate illumination
    illumination = cv2.GaussianBlur(img_float, (0, 0), 15)
    
    # Avoid division by zero
    illumination = np.maximum(illumination, 0.01)
    
    # Reflectance = Image / Illumination
    reflectance = img_float / illumination
    
    # Normalize and convert back
    enhanced = np.clip(reflectance * 255.0, 0, 255).astype(np.uint8)
    
    return enhanced

def enhance_with_enlightengan_improved(img):
    """Improved EnlightenGAN processing with preprocessing"""
    try:
        # Analyze image darkness
        darkness_level = analyze_image_darkness(img)
        
        # Preprocess based on darkness
        if darkness_level in ["very_dark", "dark"]:
            # Apply CLAHE preprocessing for very dark images
            preprocessed = enhance_with_adaptive_clahe(img, darkness_level)
        else:
            preprocessed = img
        
        # Run EnlightenGAN
        enhanced = model.predict(preprocessed)
        
        # Post-process based on result
        result_darkness = analyze_image_darkness(enhanced)
        
        if result_darkness == "dark":
            # Apply additional gamma correction
            enhanced = enhance_with_adaptive_gamma(enhanced, "dark")
        
        return enhanced
        
    except Exception as e:
        print(f"Error in EnlightenGAN processing: {e}")
        return img

def enhance_with_hybrid_method(img):
    """Hybrid method combining multiple techniques"""
    darkness_level = analyze_image_darkness(img)
    
    # Step 1: CLAHE
    clahe_result = enhance_with_adaptive_clahe(img, darkness_level)
    
    # Step 2: Gamma correction
    gamma_result = enhance_with_adaptive_gamma(clahe_result, darkness_level)
    
    # Step 3: Retinex
    retinex_result = enhance_with_retinex(gamma_result)
    
    # Step 4: Final adjustment
    final_result = cv2.convertScaleAbs(retinex_result, alpha=1.1, beta=10)
    
    return final_result

@app.route('/')
def index():
    return render_template('improved_index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read image
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            # Load image
            img = cv2.imread(file_path)
            if img is None:
                return jsonify({'error': 'Could not load image'}), 400
            
            print(f"Processing image: {filename}, shape: {img.shape}")
            
            # Analyze image darkness
            darkness_level = analyze_image_darkness(img)
            print(f"Image darkness level: {darkness_level}")
            
            # Get enhancement method from request
            method = request.form.get('method', 'enlightengan_improved')
            
            # Apply enhancement
            if method == 'enlightengan_improved':
                enhanced_img = enhance_with_enlightengan_improved(img)
            elif method == 'hybrid':
                enhanced_img = enhance_with_hybrid_method(img)
            elif method == 'clahe':
                enhanced_img = enhance_with_adaptive_clahe(img, darkness_level)
            elif method == 'histogram':
                enhanced_img = enhance_with_histogram_equalization(img)
            elif method == 'retinex':
                enhanced_img = enhance_with_retinex(img)
            elif method == 'compare_all':
                # Generate all methods for comparison
                methods_results = {
                    'Original': img,
                    'EnlightenGAN Improved': enhance_with_enlightengan_improved(img),
                    'Hybrid Method': enhance_with_hybrid_method(img),
                    'CLAHE': enhance_with_adaptive_clahe(img, darkness_level),
                    'Histogram Equalization': enhance_with_histogram_equalization(img),
                    'Retinex': enhance_with_retinex(img)
                }
                
                # Convert all to base64
                results_b64 = {}
                for method_name, result_img in methods_results.items():
                    results_b64[method_name] = image_to_base64(result_img)
                
                # Save results
                name, ext = os.path.splitext(filename)
                for method_name, result_img in methods_results.items():
                    safe_name = method_name.replace(' ', '_').replace('(', '').replace(')', '')
                    save_path = os.path.join(RESULTS_FOLDER, f"{name}_{safe_name}{ext}")
                    cv2.imwrite(save_path, result_img)
                
                # Clean up uploaded file
                os.remove(file_path)
                
                return jsonify({
                    'success': True,
                    'results': results_b64,
                    'filename': filename,
                    'original_shape': img.shape,
                    'darkness_level': darkness_level,
                    'method': 'compare_all'
                })
            else:
                enhanced_img = enhance_with_enlightengan_improved(img)
            
            # Convert images to base64 for web display
            original_b64 = image_to_base64(img)
            enhanced_b64 = image_to_base64(enhanced_img)
            
            # Save results
            name, ext = os.path.splitext(filename)
            original_save_path = os.path.join(RESULTS_FOLDER, f"{name}_original{ext}")
            enhanced_save_path = os.path.join(RESULTS_FOLDER, f"{name}_enhanced{ext}")
            
            cv2.imwrite(original_save_path, img)
            cv2.imwrite(enhanced_save_path, enhanced_img)
            
            # Clean up uploaded file
            os.remove(file_path)
            
            return jsonify({
                'success': True,
                'original_image': original_b64,
                'enhanced_image': enhanced_b64,
                'filename': filename,
                'original_shape': img.shape,
                'enhanced_shape': enhanced_img.shape,
                'method': method,
                'darkness_level': darkness_level
            })
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/results/<filename>')
def uploaded_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == '__main__':
    print("Starting Improved EnlightenGAN Web Interface...")
    print("Open your browser and go to: http://localhost:5002")
    app.run(debug=True, host='0.0.0.0', port=5002)
