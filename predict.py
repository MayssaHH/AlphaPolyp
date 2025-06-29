import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers import AdamW
import cv2
import argparse
from model_architecture.model import create_model
from model_architecture.DiceLoss import dice_metric_loss

"""
The predictions were saved to the google drive
"""
PREDICTIONS_DIR = os.path.join('AlphaPolyp', 'data', 'predictions')

def preprocess_image(image_path, img_size=352):
    """Load and preprocess an image for prediction."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, (img_size, img_size))
    
    # Normalize to [0,1]
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def load_scaler(scaler_path='regression_scaler.npy'):
    """Load the regression scaler parameters."""
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    
    scaler_params = np.load(scaler_path, allow_pickle=True).item()
    return scaler_params

def denormalize_regression(prediction, scaler_params):
    """Denormalize regression predictions using the saved scaler parameters."""
    mean = scaler_params['mean']
    scale = scaler_params['scale']
    return prediction * scale + mean

def get_subject_name(image_path):
    """Extract subject name from image path."""
    base_name = os.path.basename(image_path)
    subject_name = os.path.splitext(base_name)[0]
    return subject_name

def ensure_predictions_dir():
    """Ensure the predictions directory exists."""
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

def visualize_results(image, segmentation, volume, dimensions, subject_name):
    """Visualize the prediction results."""
    original = image.copy()
    segmentation = (segmentation * 255).astype(np.uint8)
    red_overlay = np.zeros_like(image)
    red_overlay[:, :, 2] = segmentation 
    overlay = cv2.addWeighted(image, 0.7, red_overlay, 0.3, 0)
    
    # Add text with subject name and predictions
    # Subject name
    cv2.putText(overlay, f"Subject: {subject_name}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Volume and dimensions
    text = f"Volume: {volume:.2f} | Dims: {dimensions[0]:.2f}, {dimensions[1]:.2f}, {dimensions[2]:.2f}"
    cv2.putText(overlay, text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Create a side-by-side display
    combined = np.hstack((original, overlay))
    
    return combined

def main():
    parser = argparse.ArgumentParser(description='Run polyp detection and measurement on an image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='alphapolyp_optimized_model.h5', 
                        help='Path to trained model')
    args = parser.parse_args()
    
    ensure_predictions_dir()

    subject_name = get_subject_name(args.image)
    
    output_filename = f"{subject_name}_pred.jpg"
    output_path = os.path.join(PREDICTIONS_DIR, output_filename)
    
    
    custom_objects = {
        'AdamW': AdamW,
        'dice_metric_loss': dice_metric_loss
    }
    
    if not os.path.exists(args.model):
        print(f"Model file {args.model} not found. Creating new model...")
        model = create_model(352, 352, 3, 1, 17)
    else:
        print(f"Loading model from {args.model}")
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = load_model(args.model)
    
    try:
        scaler_params = load_scaler()
        print("Loaded regression scaler parameters")
    except FileNotFoundError:
        print("Warning: Regression scaler not found. Predictions will be in normalized form.")
        scaler_params = None
    
    print(f"Processing image: {args.image}")
    img = preprocess_image(args.image)
    
    print("Running prediction...")
    segmentation, regression = model.predict(img)
    
    segmentation = segmentation[0, :, :, 0]  
   
    if scaler_params is not None:
        regression = denormalize_regression(regression, scaler_params)
    
    volume = regression[0, 0]  
    dimensions = regression[0, 1:4]  
    
    original_img = cv2.imread(args.image)
    original_img = cv2.resize(original_img, (352, 352))
    
    result = visualize_results(original_img, segmentation, volume, dimensions, subject_name)
    
    cv2.imwrite(output_path, result)
    print(f"Results saved to {output_path}")
    print(f"Subject: {subject_name}")
    print(f"Predicted Volume: {volume:.2f}")
    print(f"Predicted Dimensions (x,y,z): {dimensions[0]:.2f}, {dimensions[1]:.2f}, {dimensions[2]:.2f}")

if __name__ == "__main__":
    main() 