import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers import AdamW
import cv2
import argparse
import pickle
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

def load_regression_stats(stats_path='regression_stats.pkl'):
    """
    Load the regression statistics used during training.
    These contain min, max, mean, std values for denormalization.
    """
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Regression stats file not found at {stats_path}")
    
    with open(stats_path, 'rb') as f:
        reg_stats = pickle.load(f)
    
    print("Loaded regression statistics:")
    print(f"  Volume: min={reg_stats['min'][0]:.2f}, max={reg_stats['max'][0]:.2f}")
    print(f"  X-dim:  min={reg_stats['min'][1]:.2f}, max={reg_stats['max'][1]:.2f}")
    print(f"  Y-dim:  min={reg_stats['min'][2]:.2f}, max={reg_stats['max'][2]:.2f}")
    print(f"  Z-dim:  min={reg_stats['min'][3]:.2f}, max={reg_stats['max'][3]:.2f}")
    
    return reg_stats

def denormalize_regression(prediction, reg_stats):
    """
    Denormalize regression predictions using min-max scaling.
    This reverses the normalization used during training.
    """
    reg_min = reg_stats['min']
    reg_max = reg_stats['max']
    reg_range = reg_max - reg_min
    
    # Reverse the min-max normalization: y_norm = (y - min) / (max - min)
    # So: y = y_norm * (max - min) + min
    denormalized = prediction * reg_range + reg_min
    
    return denormalized

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
    parser.add_argument('--stats', type=str, default='regression_stats.pkl',
                        help='Path to regression statistics file')
    args = parser.parse_args()
    
    ensure_predictions_dir()

    subject_name = get_subject_name(args.image)
    output_filename = f"{subject_name}_pred.jpg"
    output_path = os.path.join(PREDICTIONS_DIR, output_filename)
    
    # Custom objects for model loading
    custom_objects = {
        'AdamW': AdamW,
        'dice_metric_loss': dice_metric_loss
    }
    
    # Load model
    if not os.path.exists(args.model):
        print(f"Model file {args.model} not found. Creating new model...")
        model = create_model(out_classes=1, starting_filters=17)
    else:
        print(f"Loading model from {args.model}")
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = load_model(args.model)
    
    try:
        reg_stats = load_regression_stats(args.stats)
        print("Successfully loaded regression statistics for denormalization")
    except FileNotFoundError:
        print(f"Warning: Regression stats file not found at {args.stats}")
        print("Predictions will be in normalized form (not denormalized)")
        reg_stats = None
    
    print(f"Processing image: {args.image}")
    img = preprocess_image(args.image, img_size=352)
    
    print("Running prediction...")
    segmentation, regression = model.predict(img, verbose=0)
    
    # Extract results
    segmentation = segmentation[0, :, :, 0]  # Remove batch and channel dimensions
    
    # Denormalize regression if stats are available
    if reg_stats is not None:
        regression = denormalize_regression(regression, reg_stats)
        print("Regression predictions denormalized to original scale")
    else:
        print("Regression predictions in normalized scale")
    
    # Extract volume and dimensions
    regression[0,0] = np.expm1(regression[0,0])
    volume = regression[0, 0]  # First value is volume
    dimensions = regression[0, 1:4]  # Next 3 values are x, y, z dimensions
    
    original_img = cv2.imread(args.image)
    original_img = cv2.resize(original_img, (352, 352))
    
    result = visualize_results(original_img, segmentation, volume, dimensions, subject_name)
    
    cv2.imwrite(output_path, result)
    print(f"Results saved to {output_path}")
    print(f"Subject: {subject_name}")
    print(f"Predicted Volume: {volume:.2f}")
    print(f"Predicted Dimensions (x,y,z): {dimensions[0]:.2f}, {dimensions[1]:.2f}, {dimensions[2]:.2f}")
    
    # Print segmentation confidence
    seg_confidence = np.mean(segmentation)
    print(f"Segmentation confidence: {seg_confidence:.3f}")

if __name__ == "__main__":
    main() 