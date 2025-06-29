"""
test.py  ──  Evaluate the RAPUNet+regression checkpoint on real data.

* computes Dice, mIoU, Precision, Recall, Accuracy on the mask
* optionally computes MSE / MAE on (volume,x,y,z) if you supply a CSV
"""

import os, csv
import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import (
    f1_score, jaccard_score,
    precision_score, recall_score, accuracy_score
)

from ModelArchitecture.DiceLoss import dice_metric_loss
from CustomLayers.ImageLoader2D import load_images_masks_from_drive


img_size   = 352                         
model_path = 'alphapolyp_optimized_model.h5'           
stats_path = 'regression_stats.pkl'     

real_base  = '/content/drive/MyPolypData/real_eval'  # ← change to your folder
img_dir    = os.path.join(real_base, 'images')
mask_dir   = os.path.join(real_base, 'masks')

csv_labels = None                         # e.g. '/content/real_eval_labels.csv'
batch_size = 4


def load_regression_stats(stats_path):
    """Load regression statistics for denormalization"""
    if not os.path.exists(stats_path):
        print(f"Warning: Regression stats not found at {stats_path}")
        return None
    
    with open(stats_path, 'rb') as f:
        reg_stats = pickle.load(f)
    print("Loaded regression statistics for denormalization")
    return reg_stats

def denormalize_regression(prediction, reg_stats):
    """Denormalize regression predictions using min-max scaling"""
    reg_min = reg_stats['min']
    reg_max = reg_stats['max']
    reg_range = reg_max - reg_min
    return prediction * reg_range + reg_min

# Load model
model = tf.keras.models.load_model(
    model_path,
    custom_objects={'dice_metric_loss': dice_metric_loss}
)

# Load regression statistics
reg_stats = load_regression_stats(stats_path)

# load real images & masks
x_real, y_real = load_images_masks_from_drive(img_dir, mask_dir, img_size)
print(f'> Loaded {len(x_real)} real images for evaluation.')

# predict
seg_pred, reg_pred = model.predict(x_real, batch_size=batch_size, verbose=1)

# Denormalize regression predictions if stats are available
if reg_stats is not None:
    reg_pred = denormalize_regression(reg_pred, reg_stats)
    reg_pred[0,0] = np.expm1(reg_pred[0,0])
    print("Regression predictions denormalized to original scale")
else:
    print("Regression predictions in normalized scale")

# segmentation metrics
y_true = (y_real.flatten() > 0.5)
y_pred = (seg_pred.flatten() > 0.5)

dice      = f1_score      (y_true, y_pred)
miou      = jaccard_score (y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall    = recall_score  (y_true, y_pred)
accuracy  = accuracy_score(y_true, y_pred)

print('\nSegmentation performance:')
print(f'  Dice       : {dice:.4f}')
print(f'  mIoU       : {miou:.4f}')
print(f'  Precision  : {precision:.4f}')
print(f'  Recall     : {recall:.4f}')
print(f'  Accuracy   : {accuracy:.4f}')

# regression metrics
if csv_labels and os.path.exists(csv_labels):
    gt_map = {}
    with open(csv_labels) as f:
        for row in csv.DictReader(f):
            gt_map[row['filename']] = [
                float(row['volume']), float(row['x']),
                float(row['y']),      float(row['z'])
            ]

    file_list = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    y_reg_true = np.array(
        [gt_map.get(f, [0, 0, 0, 0]) for f in file_list], dtype=np.float32
    )

    mse = np.mean(np.square(y_reg_true - reg_pred))
    mae = np.mean(np.abs  (y_reg_true - reg_pred))

    print('\nRegression performance:')
    print(f'  MSE (volume,x,y,z): {mse:.4f}')
    print(f'  MAE (volume,x,y,z): {mae:.4f}')
    
    # Individual metrics for each dimension
    print('\nIndividual regression metrics:')
    metrics_names = ['Volume', 'X-dim', 'Y-dim', 'Z-dim']
    for i, name in enumerate(metrics_names):
        mse_i = np.mean(np.square(y_reg_true[:, i] - reg_pred[:, i]))
        mae_i = np.mean(np.abs(y_reg_true[:, i] - reg_pred[:, i]))
        print(f'  {name}: MSE={mse_i:.4f}, MAE={mae_i:.4f}')
else:
    print('\nNo CSV with ground-truth regression labels → skipped MSE/MAE.')
