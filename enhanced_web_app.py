#!/usr/bin/env python3
"""
Enhanced EnlightenGAN Web Interface
Improved preprocessing and multiple enhancement options
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

def preprocess_image(img, method='auto'):
    """Enhanced preprocessing for better results"""
    
    if method == 'auto':
        # Auto-detect if image is dark
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 50:  # Very dark image
            method = 'aggressive'
        elif mean_brightness < 100:  # Moderately dark
            method = 'moderate'
        else:  # Already bright enough
            method = 'light'
    
    if method == 'aggressive':
        # For very dark images
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Gamma correction
        gamma = 0.7
        img = np.power(img / 255.0, gamma) * 255.0
        img = np.uint8(img)
        
    elif method == 'moderate':
        # For moderately dark images
        # Simple gamma correction
        gamma = 0.8
        img = np.power(img / 255.0, gamma) * 255.0
        img = np.uint8(img)
        
    elif method == 'light':
        # For already bright images
        # Just slight enhancement
        img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    
    return img

def postprocess_image(img, method='auto'):
    """Enhanced postprocessing for better results"""
    
    if method == 'auto':
        # Auto-adjust based on result
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness > 200:  # Too bright
            method = 'reduce'
        elif mean_brightness < 80:  # Still too dark
            method = 'boost'
        else:  # Good range
            method = 'balance'
    
    if method == 'boost':
        # Boost brightness and contrast
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
        
    elif method == 'reduce':
        # Reduce brightness
        img = cv2.convertScaleAbs(img, alpha=0.9, beta=-10)
        
    elif method == 'balance':
        # Balance the image
        img = cv2.convertScaleAbs(img, alpha=1.05, beta=5)
    
    return img

def enhance_with_enlightengan(img):
    """Enhanced EnlightenGAN processing"""
    try:
        # Preprocess
        processed_img = preprocess_image(img, 'auto')
        
        # Run EnlightenGAN
        enhanced_img = model.predict(processed_img)
        
        # Postprocess
        final_img = postprocess_image(enhanced_img, 'auto')
        
        return final_img
    except Exception as e:
        print(f"Error in EnlightenGAN processing: {e}")
        return img

def enhance_with_traditional(img):
    """Traditional image enhancement as fallback"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels and convert back
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Gamma correction
    gamma = 0.7
    enhanced = np.power(enhanced / 255.0, gamma) * 255.0
    enhanced = np.uint8(enhanced)
    
    return enhanced

@app.route('/')
def index():
    return render_template('enhanced_index.html')

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
            
            # Get enhancement method from request
            method = request.form.get('method', 'enlightengan')
            
            # Apply enhancement
            if method == 'enlightengan':
                enhanced_img = enhance_with_enlightengan(img)
            elif method == 'traditional':
                enhanced_img = enhance_with_traditional(img)
            elif method == 'both':
                # Try both methods
                enlightengan_img = enhance_with_enlightengan(img)
                traditional_img = enhance_with_traditional(img)
                
                # Convert images to base64 for web display
                original_b64 = image_to_base64(img)
                enlightengan_b64 = image_to_base64(enlightengan_img)
                traditional_b64 = image_to_base64(traditional_img)
                
                # Save results
                name, ext = os.path.splitext(filename)
                original_save_path = os.path.join(RESULTS_FOLDER, f"{name}_original{ext}")
                enlightengan_save_path = os.path.join(RESULTS_FOLDER, f"{name}_enlightengan{ext}")
                traditional_save_path = os.path.join(RESULTS_FOLDER, f"{name}_traditional{ext}")
                
                cv2.imwrite(original_save_path, img)
                cv2.imwrite(enlightengan_save_path, enlightengan_img)
                cv2.imwrite(traditional_save_path, traditional_img)
                
                # Clean up uploaded file
                os.remove(file_path)
                
                return jsonify({
                    'success': True,
                    'original_image': original_b64,
                    'enhanced_image': enlightengan_b64,
                    'traditional_image': traditional_b64,
                    'filename': filename,
                    'original_shape': img.shape,
                    'enhanced_shape': enlightengan_img.shape,
                    'method': 'both'
                })
            else:
                enhanced_img = enhance_with_enlightengan(img)
            
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
                'method': method
            })
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/results/<filename>')
def uploaded_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == '__main__':
    print("Starting Enhanced EnlightenGAN Web Interface...")
    print("Open your browser and go to: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
