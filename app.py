from flask import Flask, render_template, redirect, request, abort, url_for, flash, jsonify, send_from_directory
import os
import logging
from typing import Tuple, List, Optional, Dict, Any
import cv2
import face_recognition
import numpy as np
from werkzeug.utils import secure_filename
import shutil
from datetime import datetime
from functools import wraps
import time

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for flash messages

class Config:
    """Application configuration"""
    INPUT_DIR = "uploads"
    OUTPUT_DIR = "static/processed"
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    FACE_DETECTION_MODEL = "hog"  # Use 'hog' for CPU, 'cnn' for GPU if available
    CLEANUP_THRESHOLD = 100  # Number of files before cleanup
    CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for face detection
    CACHE_TIMEOUT = 3600  # Cache timeout in seconds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure required directories exist
for directory in [Config.INPUT_DIR, Config.OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

app.config["UPLOAD_FOLDER"] = Config.INPUT_DIR
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

def timing_decorator(f):
    """Decorator to measure function execution time"""
    @wraps(f)
    def wrap(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        logger.info(f'{f.__name__} took {end-start:.2f} seconds')
        return result
    return wrap

def cleanup_old_files() -> None:
    """Clean up old files when directory gets too large."""
    try:
        for directory in [Config.INPUT_DIR, Config.OUTPUT_DIR]:
            files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            if len(files) > Config.CLEANUP_THRESHOLD:
                logger.info(f"Cleaning up old files in {directory}")
                files.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)))
                for old_file in files[:-50]:  # Keep the 50 most recent files
                    try:
                        os.remove(os.path.join(directory, old_file))
                    except OSError as e:
                        logger.error(f"Error removing file {old_file}: {e}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def get_file_extension(filename: str) -> Optional[str]:
    """Extract file extension from filename."""
    try:
        return os.path.splitext(filename)[1].lower() if filename else None
    except (AttributeError, ValueError):
        return None

def validate_image_file(filename: str) -> Tuple[bool, str]:
    """
    Validate if the file has an allowed image extension and is secure.
    
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if not filename:
        return False, "No filename provided"
        
    filename = secure_filename(filename)
    ext = get_file_extension(filename)
    
    if not ext:
        return False, "Invalid file format"
    if ext not in Config.ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed types: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        
    return True, ""

def analyze_face(face: Tuple, encoding: np.ndarray) -> Dict[str, Any]:
    """Analyze facial features and return detailed information."""
    top, right, bottom, left = face
    face_width = right - left
    face_height = bottom - top
    
    # Calculate face metrics
    face_size = face_width * face_height
    face_aspect_ratio = face_width / face_height if face_height > 0 else 0
    
    # Analyze face position
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    
    return {
        'size': face_size,
        'aspect_ratio': round(face_aspect_ratio, 2),
        'position': {
            'center_x': round(center_x, 2),
            'center_y': round(center_y, 2)
        },
        'confidence': round(float(np.mean(encoding)), 3),
        'dimensions': {
            'width': face_width,
            'height': face_height
        }
    }

@timing_decorator
def get_faces_and_encodings(filename: str) -> Tuple[List[Tuple], List[np.ndarray], List[Dict[str, Any]]]:
    """
    Detect faces and get face encodings from image with error handling and optimization.
    Returns faces, encodings, and face analysis data.
    """
    try:
        image_path = os.path.join(Config.INPUT_DIR, filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Convert BGR to RGB (face_recognition expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        faces = face_recognition.face_locations(image, model=Config.FACE_DETECTION_MODEL)
        encodings = face_recognition.face_encodings(image, faces)
        
        analyses = []
        for face in faces:
            top, right, bottom, left = face
            face_height = bottom - top
            face_width = right - left
            
            confidence = 1.0  # Default confidence for HOG
            if Config.FACE_DETECTION_MODEL == "cnn":
                # For CNN model we could implement confidence calculation
                confidence = 0.9
            
            analysis = {
                "size": min(face_height, face_width),
                "confidence": confidence,
                "position": {"top": top, "right": right, "bottom": bottom, "left": left}
            }
            analyses.append(analysis)
        
        return faces, encodings, analyses
        
    except Exception as e:
        logger.error(f"Error processing image {filename}: {str(e)}")
        raise

@timing_decorator
def make_rect_around_faces(faces: List[Tuple], filename: str, analyses: List[Dict]) -> str:
    """Draw rectangles around detected faces with improved visualization."""
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")
        
    img = cv2.imread(filepath)
    for i, (face, analysis) in enumerate(zip(faces, analyses)):
        top, right, bottom, left = face
        
        # Generate unique color based on face characteristics
        hue = (hash(str(analysis['confidence'])) % 360) / 360.0
        rgb = tuple(int(x * 255) for x in cv2.cvtColor(np.uint8([[[ hue, 0.8, 0.8 ]]]), cv2.COLOR_HSV2BGR)[0][0])
        
        # Draw face rectangle
        cv2.rectangle(img, (left, top), (right, bottom), rgb, 2)
        
        # Add face information
        confidence = analysis['confidence']
        label = f"Face {i+1} ({confidence:.2f})"
        cv2.putText(img, label, (left, top-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed-{timestamp}-{filename}"
    output_path = os.path.join(Config.OUTPUT_DIR, output_filename)
    cv2.imwrite(output_path, img)
    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if "pic" not in request.files:
        flash("No file uploaded", "error")
        return redirect(url_for('index'))
        
    pic = request.files["pic"]
    if not pic.filename:
        flash("No file selected", "error")
        return redirect(url_for('index'))
        
    is_valid, error_msg = validate_image_file(pic.filename)
    if not is_valid:
        flash(error_msg, "error")
        return redirect(url_for('index'))
        
    try:
        filename = secure_filename(pic.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        pic.save(filepath)
        cleanup_old_files()  # Clean up old files if necessary
        return redirect(url_for('showface', filename=filename))
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        flash("Error uploading file", "error")
        return redirect(url_for('index'))

@app.route("/showface/<filename>")
def showface(filename):
    try:
        faces, encodings, analyses = get_faces_and_encodings(filename)
        
        if not faces:
            flash("No faces found in image", "warning")
            return redirect(url_for('index'))
            
        output_path = make_rect_around_faces(faces, filename, analyses)
        
        # Format face data for display
        face_data = []
        for i, (encoding, analysis) in enumerate(zip(encodings, analyses)):
            face_info = {
                'id': i + 1,
                'confidence': analysis['confidence'],
                'size': f"{analysis['size']}px",
                'position': f"(top:{analysis['position']['top']}, left:{analysis['position']['left']})",
                'encoding': {f"Point {i+1}": f"{val:.4f}" for i, val in enumerate(encoding)}
            }
            face_data.append(face_info)
        
        return render_template(
            'showface.html',
            faces=len(faces),
            face_data=face_data,
            outimg=url_for('static', filename=f'processed/{os.path.basename(output_path)}'),
            original_filename=filename
        )
        
    except FileNotFoundError:
        flash("Image not found", "error")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        flash(f"Error processing image: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/api/faces/<filename>')
def get_face_data(filename):
    """API endpoint to get face data in JSON format"""
    try:
        faces, encodings, analyses = get_faces_and_encodings(filename)
        return jsonify({
            'success': True,
            'face_count': len(faces),
            'faces': analyses
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)