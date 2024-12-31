# FaceID

A Flask-based web application for face detection and analysis. The application can detect faces in uploaded images and provide detailed analysis of facial features.

## Features

- Face detection using HOG/CNN models
- Upload and process images
- Visual display of detected faces
- Face location and confidence information
- Support for various image formats (JPG, JPEG, PNG)

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd FaceID
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Upload an image and see the face detection results

## Requirements

- Python 3.6+
- OpenCV
- face_recognition
- Flask
- NumPy

## Directory Structure

- `/templates` - HTML templates
- `/static` - Static files (CSS, processed images)
- `/uploads` - Temporary storage for uploaded images
