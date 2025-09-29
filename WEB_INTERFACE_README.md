# 🌅 EnlightenGAN Web Interface

A beautiful, user-friendly web interface for EnlightenGAN low-light image enhancement.

## 🚀 Quick Start

### 1. Start the Web Interface
```bash
python web_app.py
```

### 2. Open Your Browser
Go to: **http://localhost:5000**

### 3. Upload and Enhance Images
- Drag & drop images or click to browse
- Wait 2-3 seconds for AI enhancement
- Download the enhanced results

## ✨ Features

### 🎨 **Beautiful Web Interface**
- Modern, responsive design
- Drag & drop file upload
- Real-time image preview
- Side-by-side comparison
- Download enhanced images

### ⚡ **Fast Processing**
- **Single image**: 1-3 seconds
- **Batch processing**: Available via command line
- **CPU optimized**: Works without GPU

### 📱 **Mobile Friendly**
- Responsive design
- Touch-friendly interface
- Works on phones and tablets

## 🛠️ Usage Options

### Option 1: Web Interface (Recommended)
```bash
python web_app.py
```
- **Time**: 1-3 seconds per image
- **Best for**: Single images, easy to use
- **Access**: http://localhost:5000

### Option 2: Command Line (Single Image)
```bash
python test_inference.py
```
- **Time**: 1-3 seconds per image
- **Best for**: Quick testing

### Option 3: Batch Processing (Multiple Images)
```bash
python batch_process.py --input ./test_dataset/testA/data/DICM --output ./batch_results
```
- **Time**: 5-15 minutes for hundreds of images
- **Best for**: Processing many images at once

### Option 4: Original Training/Inference
```bash
python train.py --dataroot . --name enlightening --model cycle_gan --which_direction AtoB --loadSize 512 --fineSize 256 --batchSize 1 --niter 100 --niter_decay 100 --gpu_ids -1
```
- **Time**: 8-24 hours for full training
- **Best for**: Training from scratch

## 📁 File Structure

```
EnlightenGAN-master/
├── web_app.py              # 🌐 Web interface (MAIN)
├── batch_process.py        # 📦 Batch processing
├── test_inference.py       # 🧪 Single image test
├── templates/
│   └── index.html          # 🎨 Web interface HTML
├── uploads/                # 📤 Temporary uploads
├── results/                # 📥 Enhanced images
└── enlighten_inference/    # 🤖 AI model
```

## 🎯 Supported Formats

- **Input**: JPG, JPEG, PNG, BMP, TIFF
- **Output**: JPG (optimized for web)
- **Max Size**: 16MB per image
- **Resolution**: Any size (automatically handled)

## 🔧 Technical Details

### **Model**: EnlightenGAN ONNX
- **Type**: Pre-trained neural network
- **Purpose**: Low-light image enhancement
- **Speed**: 1-3 seconds per image (CPU)
- **Quality**: Professional-grade results

### **Web Framework**: Flask
- **Backend**: Python Flask
- **Frontend**: HTML5 + CSS3 + JavaScript
- **Image Processing**: OpenCV + PIL
- **AI Inference**: ONNX Runtime

## 🌟 Example Results

The web interface will show:
- **Original**: Your dark/low-light image
- **Enhanced**: AI-enhanced bright image
- **Comparison**: Side-by-side view
- **Download**: Save enhanced image

## 🚨 Troubleshooting

### **Web Interface Not Loading**
```bash
# Check if port 5000 is in use
netstat -an | findstr :5000

# If busy, change port in web_app.py
app.run(port=5001)  # Use different port
```

### **Model Loading Issues**
```bash
# Reinstall dependencies
pip install onnxruntime opencv-python flask pillow
```

### **Image Upload Problems**
- Check file size (< 16MB)
- Ensure image format is supported
- Try different browser if issues persist

## 📊 Performance

| Method | Time per Image | Best For |
|--------|----------------|----------|
| Web Interface | 1-3 seconds | Single images |
| Batch Processing | 1-3 seconds | Multiple images |
| Training | 8-24 hours | Custom models |

## 🎉 Success!

Your EnlightenGAN web interface is now running! 

**🌐 Open**: http://localhost:5000
**⏱️ Processing**: 1-3 seconds per image
**✨ Quality**: Professional AI enhancement

Enjoy enhancing your low-light images! 📸✨
