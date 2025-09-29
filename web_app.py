#!/usr/bin/env python3
"""
EnlightenGAN Web Interface
A Flask web application for low-light image enhancement
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from enlighten_inference import EnlightenOnnxModel
import base64
from io import BytesIO
from PIL import Image

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
    pil_image.save(buffer, format='JPEG', quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{img_str}"

@app.route('/')
def index():
    return render_template('index.html')

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
            
            # Run EnlightenGAN inference
            enhanced_img = model.predict(img)
            
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
                'enhanced_shape': enhanced_img.shape
            })
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/results/<filename>')
def uploaded_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == '__main__':
    print("Starting EnlightenGAN Web Interface...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
