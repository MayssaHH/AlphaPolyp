import os, csv, gc, datetime
import numpy as np
import albumentations as albu
import pickle
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
import tensorflow as tf

from model_architecture.DiceLoss import dice_metric_loss
from model_architecture.model   import create_model
from model_architecture.ImageLoader2D  import load_images_masks_from_drive

drive_base     = 'G:\My Drive\extracted_folder\synth-colon'
real_img_dir  = os.path.join(drive_base, 'cyclegan_images')
real_mask_dir = os.path.join(drive_base, 'masks')
synth_img_dir  = os.path.join(drive_base, 'images')
synth_mask_dir = os.path.join(drive_base, 'masks')
csv_labels     = 'best_labels.csv'    


img_size       = 352
filters        = 17
batch_size     = 8
seed           = 58800

def validate_paths():
    """Validate that all required paths exist"""
    paths_to_check = [
        (real_img_dir, "Real images directory"),
        (real_mask_dir, "Real masks directory"), 
        (synth_img_dir, "Synthetic images directory"),
        (synth_mask_dir, "Synthetic masks directory"),
        (csv_labels, "CSV labels file")
    ]
    
    missing_paths = []
    for path, description in paths_to_check:
        if not os.path.exists(path):
            missing_paths.append(f"{description}: {path}")
    
    if missing_paths:
        print("ERROR: Missing required paths:")
        for path in missing_paths:
            print(f"  - {path}")
        print("\nPlease ensure all paths exist before running training.")
        return False
    return True

def load_label_map(csv_file):
    """Load label map with error handling"""
    label_map = {}
    try:
        with open(csv_file, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                try:
                    label_map[row['Filename']] = [
                        float(row['logVolume']),
                        float(row['x']),
                        float(row['y']),
                        float(row['z'])
                    ]
                except (KeyError, ValueError) as e:
                    print(f"Warning: Skipping invalid row in CSV: {e}")
                    continue
    except FileNotFoundError:
        print(f"ERROR: CSV file not found: {csv_file}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to read CSV file: {e}")
        return None
    
    print(f"Loaded {len(label_map)} labels from CSV")
    return label_map

def get_reg_labels(file_list):
    """Return an [N,4] array of [vol,x,y,z] for each filename."""
    regs = []
    missing_labels = []
    for fname in file_list:
        base    = os.path.splitext(fname)[0]        
        csv_key = f"{base}_labeled.obj"             
        if label_map is not None and csv_key in label_map:
            regs.append(label_map[csv_key])
        else:
            missing_labels.append(fname)
    
    if missing_labels:
        print(f"Warning: {len(missing_labels)} files missing labels")
    
    return np.array(regs, dtype=np.float32)

def filter_labeled(img_dir):
    """
    Return sorted list of filenames in img_dir whose
    base+'_labeled.obj' is in label_map.
    """
    out = []
    if not os.path.exists(img_dir):
        print(f"ERROR: Image directory does not exist: {img_dir}")
        return out
        
    for fn in os.listdir(img_dir):
        if not fn.lower().endswith(('.jpg','.png','.jpeg')): 
            continue
        base = os.path.splitext(fn)[0]             
        key  = f"{base}_labeled.obj"              
        if key in label_keys:
            out.append(fn)
    return sorted(out)

def interleave(X1, M1, R1, X2, M2, R2):
    """
    Interleave between real and synthetic images
    """
    Xc, Mc, Rc = [], [], []
    n = min(len(X1), len(X2))
    for i in range(n):
        Xc.append(X1[i]); Mc.append(M1[i]); Rc.append(R1[i])
        Xc.append(X2[i]); Mc.append(M2[i]); Rc.append(R2[i])
    
    if len(X1) > n:
        Xc.extend(X1[n:]); Mc.extend(M1[n:]); Rc.extend(R1[n:])
    if len(X2) > n:
        Xc.extend(X2[n:]); Mc.extend(M2[n:]); Rc.extend(R2[n:])
    return np.array(Xc), np.array(Mc), np.array(Rc)

def compute_regression_statistics(reg_data):
    """Compute statistics for regression data normalization"""
    stats = {
        'min': np.min(reg_data, axis=0),
        'max': np.max(reg_data, axis=0),
        'mean': np.mean(reg_data, axis=0),
        'std': np.std(reg_data, axis=0),
        'range': np.max(reg_data, axis=0) - np.min(reg_data, axis=0)
    }
    
    print("Regression data statistics:")
    print(f"  Volume: min={stats['min'][0]:.2f}, max={stats['max'][0]:.2f}, mean={stats['mean'][0]:.2f}")
    print(f"  X-dim:  min={stats['min'][1]:.2f}, max={stats['max'][1]:.2f}, mean={stats['mean'][1]:.2f}")
    print(f"  Y-dim:  min={stats['min'][2]:.2f}, max={stats['max'][2]:.2f}, mean={stats['mean'][2]:.2f}")
    print(f"  Z-dim:  min={stats['min'][3]:.2f}, max={stats['max'][3]:.2f}, mean={stats['mean'][3]:.2f}")
    
    return stats

def create_minmax_normalized_mse_loss(reg_stats):
    """
    Create MSE loss normalized using min-max scaling from training data statistics
    """
    reg_min, reg_max = reg_stats['min'], reg_stats['max']
    reg_range = reg_max - reg_min + 1e-8

    def minmax_normalized_mse_loss(y_true, y_pred):
        y_true_norm = (y_true - reg_min) / reg_range
        y_pred_norm = (y_pred - reg_min) / reg_range
        
        return tf.keras.losses.mean_squared_error(y_true_norm, y_pred_norm)
    
    return minmax_normalized_mse_loss

def augment_batch(X, M, R):
    """
    Apply albumentations to images & masks
    Leave regression labels unchanged.
    """
    Xa, Ma, Ra = [], [], []
    for img, msk, reg in zip(X, M, R):
        try:
            a = aug(image=(img*255).astype(np.uint8),
                    mask =(msk*255).astype(np.uint8))
            Xa.append(a['image'] / 255.0)
            Ma.append(a['mask']  / 255.0)
            Ra.append(reg)
        except Exception as e:
            print(f"Warning: Augmentation failed for sample: {e}")
            # Use original data if augmentation fails
            Xa.append(img)
            Ma.append(msk)
            Ra.append(reg)
    return ( np.array(Xa, dtype=np.float32),
             np.expand_dims(np.array(Ma, dtype=np.float32), -1),
             np.array(Ra, dtype=np.float32) )

def freeze_layers_except_regression(model):
    """Freeze all layers except regression output"""
    for layer in model.layers:
        if 'regression_output' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

def unfreeze_all_layers(model):
    """Unfreeze all layers"""
    for layer in model.layers:
        layer.trainable = True


"""
How many epochs for each phase 
epochs_phase1  for regression head only (segmentation head is frozen)
epochs_phase2  to fine-tune all (segmentation head is unfrozen)
"""

epochs_phase1  = 10  
epochs_phase2  = 15  

"""
This is the path to the pretrained 
RAPUNet checkpoint (segmentation only)
"""

pretrained_ckpt = 'rapunet_pretrained.h5'

log_root       = './logs'
os.makedirs(log_root, exist_ok=True)

label_map = load_label_map(csv_labels)
if label_map is None:
    exit(1)

label_keys = set(label_map.keys())

if not validate_paths():
    exit(1)

real_files  = filter_labeled(real_img_dir)
synth_files = filter_labeled(synth_img_dir)

if not real_files:
    print(f"ERROR: No labeled real files found in {real_img_dir}")
    exit(1)
if not synth_files:
    print(f"ERROR: No labeled synthetic files found in {synth_img_dir}")
    exit(1)

print(f"Found {len(real_files)} real files and {len(synth_files)} synthetic files")

try:
    X_real,  Y_real_mask  = load_images_masks_from_drive(real_img_dir,  real_mask_dir,  img_size)
    X_synth, Y_synth_mask = load_images_masks_from_drive(synth_img_dir, synth_mask_dir, img_size)
except Exception as e:
    print(f"ERROR: Failed to load images/masks: {e}")
    exit(1)

Y_real_reg  = get_reg_labels(real_files)
Y_synth_reg = get_reg_labels(synth_files)

if len(X_real) != len(Y_real_reg):
    print(f"ERROR: Mismatch between real images ({len(X_real)}) and regression labels ({len(Y_real_reg)})")
    exit(1)
if len(X_synth) != len(Y_synth_reg):
    print(f"ERROR: Mismatch between synthetic images ({len(X_synth)}) and regression labels ({len(Y_synth_reg)})")
    exit(1)

X_all, Y_all_mask, Y_all_reg = interleave(
    X_real,  Y_real_mask,  Y_real_reg,
    X_synth, Y_synth_mask, Y_synth_reg
)

if len(X_all) < 10:
    print(f"ERROR: Insufficient data ({len(X_all)} samples). Need at least 10 samples.")
    exit(1)

x_train, x_val, y_train_mask, y_val_mask, y_train_reg, y_val_reg = \
    train_test_split(
        X_all, Y_all_mask, Y_all_reg,
        test_size=0.1, shuffle=True, random_state=seed
    )

print(f"Training set: {len(x_train)} samples, Validation set: {len(x_val)} samples")

reg_stats = compute_regression_statistics(y_train_reg)

# Save regression statistics for prediction
pickle.dump(reg_stats, open('regression_stats.pkl', 'wb'))
print("Saved regression statistics to regression_stats.pkl")

normalized_regression_loss = create_minmax_normalized_mse_loss(reg_stats)

aug = albu.Compose([
    albu.HorizontalFlip(),
    albu.VerticalFlip(),
    albu.ColorJitter(brightness=(0.6,1.6),contrast=0.2,saturation=0.1,hue=0.01,always_apply=True),
    albu.Affine(scale=(0.5,1.5),translate_percent=(-0.125,0.125),rotate=(-180,180),shear=(-22.5,22),always_apply=True),
])

try:
    model = create_model(out_classes=1, starting_filters=filters, reg_mean_norm=reg_stats['mean'])
    print("Model created successfully")
except Exception as e:
    print(f"ERROR: Failed to create model: {e}")
    exit(1)

if os.path.exists(pretrained_ckpt):
    try:
        model.load_weights(pretrained_ckpt, by_name=True, skip_mismatch=True)
        print('Loaded pretrained RAPUNet weights.')
    except Exception as e:
        print(f"Warning: Failed to load pretrained weights: {e}")

run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
callbacks = [
    CSVLogger    (f'{log_root}/train_{run_id}.csv'),
    TensorBoard  (log_dir=f'{log_root}/tb_{run_id}'),
    ModelCheckpoint('alphapolyp_optimized_model.h5',
                    monitor='val_segmentation_output_loss',
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_segmentation_output_loss',
                      factor=0.5, patience=8, verbose=1)
]

""" 
PHASE 1: TRAIN REGRESSION HEAD ONLY 
Freeze everything except regression head
"""
freeze_layers_except_regression(model)

model.compile(
    optimizer=AdamW(1e-4, weight_decay=1e-6),
    loss={'segmentation_output': dice_metric_loss,
          'regression_output'  : normalized_regression_loss},  # Use normalized loss
    loss_weights={'segmentation_output':1.0,
                  'regression_output'  :1.0}
)

print('Phase1: training regression head only')
print('Using normalized regression loss to balance with segmentation loss')
for epoch in range(epochs_phase1):
    try:
        Xa, Ma, Ra = augment_batch(x_train, y_train_mask, y_train_reg)
        model.fit(Xa,
                  {'segmentation_output': Ma,
                   'regression_output'  : Ra},
                  validation_data=(x_val,
                                   {'segmentation_output': y_val_mask,
                                    'regression_output'  : y_val_reg}),
                  epochs=1, batch_size=batch_size,
                  callbacks=callbacks, verbose=1)
        gc.collect()
    except Exception as e:
        print(f"ERROR: Training failed at epoch {epoch}: {e}")
        break

"""
PHASE 2: FINE-TUNE ENTIRE NETWORK 
Unfreeze all layers
"""

unfreeze_all_layers(model)

model.compile(
    optimizer=AdamW(1e-5, weight_decay=1e-6),
    loss={'segmentation_output': dice_metric_loss,
          'regression_output'  : normalized_regression_loss},  
    loss_weights={'segmentation_output':1.0,
                  'regression_output'  :1.0}
)

print('Phase2: fine-tuning all layers')
print('Using normalized regression loss to balance with segmentation loss')
for epoch in range(epochs_phase2):
    try:
        Xa, Ma, Ra = augment_batch(x_train, y_train_mask, y_train_reg)
        model.fit(Xa,
                  {'segmentation_output': Ma,
                   'regression_output'  : Ra},
                  validation_data=(x_val,
                                   {'segmentation_output': y_val_mask,
                                    'regression_output'  : y_val_reg}),
                  epochs=1, batch_size=batch_size,
                  callbacks=callbacks, verbose=1)
        gc.collect()
    except Exception as e:
        print(f"ERROR: Training failed at epoch {epoch}: {e}")
        break

print('Training complete — alphapolyp_optimized_model.h5 saved.')
print('Note: Regression outputs are in original scale. Use reg_stats for denormalization if needed.')
