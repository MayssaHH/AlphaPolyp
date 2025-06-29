import os
import uuid
import numpy as np
import requests
import cv2
import time 
from flask import Flask, render_template, request, jsonify, url_for, redirect, flash, send_from_directory
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'alphapolyp_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Get model API URL from environment variable or use default
MODEL_API_URL = os.environ.get('MODEL_API_URL', 'http://model:5001')

# Create upload and result directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def visualize_results(image_path, segmentation, volume, dimensions, subject_name):
    """Create visualization of the prediction results"""
    # Read original image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Resize to match the model input size
    image = cv2.resize(image, (352, 352))
    
    # Create a copy of the original image
    original = image.copy()
    
    # Convert segmentation to numpy array if it's a list
    if isinstance(segmentation, list):
        segmentation = np.array(segmentation)
    
    # Resize segmentation to match image dimensions
    segmentation = cv2.resize(segmentation, (image.shape[1], image.shape[0]))
    
    # Create segmentation overlay with red color
    segmentation = (segmentation * 255).astype(np.uint8)
    
    # Create red overlay (BGR format)
    red_overlay = np.zeros_like(image)
    red_overlay[:, :, 2] = segmentation  # Red channel
    
    # Blend original image with red overlay
    overlay = cv2.addWeighted(image, 0.7, red_overlay, 0.3, 0)
    
    
    # Create a side-by-side display
    combined = np.hstack((original, overlay))
    
    return combined

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        base_name, extension = os.path.splitext(filename)
        unique_filename = f"{base_name}_{uuid.uuid4().hex[:8]}{extension}"
        
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Process the image
        try:
            start_time = time.time()  # Start the timer
            # Send the image to the model API
            with open(file_path, 'rb') as f:
                files = {'file': (unique_filename, f, 'image/jpeg')}
                print("About to send request to model API...", flush=True)
                try:
                    response = requests.post(f"{MODEL_API_URL}/upload", files=files)
                    print("Model API status code:", response.status_code, flush=True)
                    print("Model API response text:", response.text, flush=True)
                except Exception as e:
                    print("Exception during model API request:", str(e), flush=True)
                if response.status_code == 200:
                    result_data = response.json()
                    
                    # Extract data from the response
                    volume = result_data.get('volume', 0)
                    print("Volume:", volume)
                    dimensions = result_data.get('dimensions', [0, 0, 0])
                    segmentation = np.array(result_data.get('segmentation', []))
                    processing_time = time.time() - start_time  # Calculate processing time
                    
                    # Create visualization
                    subject_name = base_name
                    result_img = visualize_results(file_path, segmentation, volume, dimensions, subject_name)
                    
                    # Save result image
                    result_filename = f"{base_name}_result{extension}"
                    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                    cv2.imwrite(result_path, result_img)
                    
                    # Return results
                    return render_template('result.html', 
                                          original_image=url_for('static', filename=f'uploads/{unique_filename}'),
                                          result_image=url_for('static', filename=f'results/{result_filename}'),
                                          filename=base_name,
                                          volume=volume,
                                          dimensions=dimensions,
                                          processing_time=processing_time,
                                        has_polyp=np.any(segmentation > 0.5))
                else:
                    flash(f'Error from model API: {response.text}')
                    return redirect(url_for('index'))
                
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.')
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
