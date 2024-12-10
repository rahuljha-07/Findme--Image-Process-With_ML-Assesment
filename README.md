# **Image Detection Project**

## **Overview**
This project uses a pre-trained Faster R-CNN model from PyTorch and OpenCV to perform object detection on shelf images. It processes input images, detects objects, and overlays bounding boxes, labels, and confidence scores on the detected objects.

---

## **Project Structure**
```
image_detection_project/
├── images/                # Input images (e.g., 001.jpg, 002.jpg)
├── output/                # Processed images with bounding boxes
├── src/                   # Source code
│   ├── model_loader.py    # Script to load the Faster R-CNN model
│   ├── image_processor.py # Handles image preprocessing and annotation
│   ├── detect_objects.py  # Main script to process images and run detection
├── requirements.txt       # Required Python packages
├── README.md              # Project documentation
```

---

## **Setup Instructions**

### **1. Prerequisites**
Ensure you have the following installed:
- Python 3.8 or later
- pip (Python package manager)

### **2. Install Dependencies**
Run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

---

## **How to Run**

### **1. Add Images**
Place all input images (e.g., `001.jpg`, `002.jpg`) in the `images/` folder.

### **2. Run the Script**
Navigate to the `src` folder and run the main script:
```bash
cd src
python detect_objects.py
```

### **3. View Output**
Processed images with bounding boxes will be saved in the `output/` folder. Each output file retains the same name as the input image (e.g., `001.jpg`).

---

## **How It Works**

### **1. Model Loading**
   - The `model_loader.py` script loads a pre-trained Faster R-CNN model from PyTorch’s torchvision library.

### **2. Image Preprocessing**
   - The `image_processor.py` script reads, validates, and preprocesses images from the `images/` folder for object detection.
   - It also handles drawing bounding boxes, labels, and confidence scores on the images.

### **3. Object Detection**
   - The `detect_objects.py` script orchestrates the entire workflow:
     - Loads the pre-trained model.
     - Processes each image from the `images/` folder.
     - Runs the model to detect objects.
     - Saves annotated images to the `output/` folder.

---
