import os, csv, gc, datetime
import numpy as np
import albumentations as albu
import pickle
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
import tensorflow as tf
import argparse

from model_architecture.DiceLoss import dice_metric_loss
from model_architecture.model import create_model
from model_architecture.DataGenerator import LargeDatasetGenerator, create_minmax_normalized_mse_loss

# Argument parsing
parser = argparse.ArgumentParser(description='Train AlphaPolyp model on large dataset (groups)')
parser.add_argument('--root', type=str, required=True, help='Root path to data (drive_base)')
parser.add_argument('--csv', type=str, required=True, help='CSV file with labels')
parser.add_argument('--stats', type=str, required=True, help='Path to global regression stats')
args = parser.parse_args()

# Configuration
drive_base = args.root
real_img_dir = os.path.join(drive_base, 'cyclegan_images')
real_mask_dir = os.path.join(drive_base, 'masks')
synth_img_dir = os.path.join(drive_base, 'images')
synth_mask_dir = os.path.join(drive_base, 'masks')
csv_labels = args.csv
stats_path = args.stats

img_size = 384
filters = 17
batch_size = 8
group_size = 40  # Number of images per group
seed = 58800

# Training phases
epochs_phase1 = 10  # Regression head only
epochs_phase2 = 15  # Fine-tune all layers

# Pretrained checkpoint
pretrained_ckpt = 'rapunet_pretrained.h5'

# Logging
log_root = './logs'
os.makedirs(log_root, exist_ok=True)

def validate_paths():
    """Validate that all required paths exist"""
    paths_to_check = [
        (real_img_dir, "Real images directory"),
        (real_mask_dir, "Real masks directory"), 
        (synth_img_dir, "Synthetic images directory"),
        (synth_mask_dir, "Synthetic masks directory"),
        (csv_labels, "CSV labels file"),
        (stats_path, "Global regression stats file")
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

def augment_batch(X, M, R):
    """
    Apply albumentations to images & masks
    Leave regression labels unchanged.
    """
    Xa, Ma, Ra = [], [], []
    for img, msk, reg in zip(X, M, R):
        try:
            a = aug(image=(img*255).astype(np.uint8),
                    mask=(msk*255).astype(np.uint8))
            Xa.append(a['image'] / 255.0)
            Ma.append(a['mask'] / 255.0)
            Ra.append(reg)
        except Exception as e:
            print(f"Warning: Augmentation failed for sample: {e}")
            # Use original data if augmentation fails
            Xa.append(img)
            Ma.append(msk)
            Ra.append(reg)
    return (np.array(Xa, dtype=np.float32),
            np.expand_dims(np.array(Ma, dtype=np.float32), -1),
            np.array(Ra, dtype=np.float32))

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

def train_on_group(model, data_generator, group_idx, phase, epochs, callbacks):
    """Train model on a specific group"""
    print(f"\n=== Training on Group {group_idx} (Phase {phase}) ===")
    
    # Get group info
    group_info = data_generator.get_group_info(group_idx)
    print(f"Group {group_idx}: {group_info['total_samples']} samples "
          f"({group_info['real_samples']} real, {group_info['synthetic_samples']} synthetic)")
    
    # Load and split data for this group
    x_train, y_train_mask, y_train_reg, x_val, y_val_mask, y_val_reg = \
        data_generator.get_group_train_val_split(group_idx)
    
    if len(x_train) < 10:
        print(f"Warning: Group {group_idx} has insufficient training data ({len(x_train)} samples)")
        return False
    
    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")
    
    # Train for specified epochs
    for epoch in range(epochs):
        try:
            print(f"Epoch {epoch + 1}/{epochs}")
            
            # Apply augmentation
            Xa, Ma, Ra = augment_batch(x_train, y_train_mask, y_train_reg)
            
            # Train
            history = model.fit(
                Xa,
                {'segmentation_output': Ma,
                 'regression_output': Ra},
                validation_data=(x_val,
                               {'segmentation_output': y_val_mask,
                                'regression_output': y_val_reg}),
                epochs=1, batch_size=batch_size,
                callbacks=callbacks, verbose=1
            )
            
            # Clean up memory
            del Xa, Ma, Ra
            gc.collect()
            
        except Exception as e:
            print(f"ERROR: Training failed at epoch {epoch + 1}: {e}")
            return False
    
    return True

def main():
    """Main training function"""
    print("=== AlphaPolyp Large Dataset Training ===")
    
    # Validate paths
    if not validate_paths():
        exit(1)
    
    # Load global regression statistics
    with open(stats_path, 'rb') as f:
        global_reg_stats = pickle.load(f)
    print(f"Loaded global regression statistics from {stats_path}")
    
    # Initialize data generator
    try:
        print("Initializing data generator...")
        data_generator = LargeDatasetGenerator(
            real_img_dir=real_img_dir,
            real_mask_dir=real_mask_dir,
            synth_img_dir=synth_img_dir,
            synth_mask_dir=synth_mask_dir,
            csv_labels=csv_labels,
            img_size=img_size,
            group_size=group_size,
            batch_size=batch_size,
            seed=seed
        )
        print(f"Data generator initialized with {data_generator.get_num_groups()} groups")
    except Exception as e:
        print(f"ERROR: Failed to initialize data generator: {e}")
        exit(1)
    
    # Create model
    try:
        model = create_model(img_height=img_size, img_width=img_size, input_channels=3, out_classes=1, starting_filters=filters, 
                           reg_mean_norm=global_reg_stats['mean'])
        print("Model created successfully")
    except Exception as e:
        print(f"ERROR: Failed to create model: {e}")
        exit(1)
    
    # Load pretrained weights if available
    if os.path.exists(pretrained_ckpt):
        try:
            model.load_weights(pretrained_ckpt, by_name=True, skip_mismatch=True)
            print('Loaded pretrained RAPUNet weights.')
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")
    
    # Create normalized loss function
    normalized_regression_loss = create_minmax_normalized_mse_loss(global_reg_stats)
    
    # Setup callbacks
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    callbacks = [
        CSVLogger(f'{log_root}/train_{run_id}.csv'),
        TensorBoard(log_dir=f'{log_root}/tb_{run_id}'),
        ModelCheckpoint('alphapolyp_optimized_model.h5',
                       monitor='val_segmentation_output_loss',
                       save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_segmentation_output_loss',
                         factor=0.5, patience=8, verbose=1)
    ]
    
    # Training loop over groups
    num_groups = data_generator.get_num_groups()
    
    print(f"\n=== Starting Training on {num_groups} Groups ===")
    print(f"Phase 1: {epochs_phase1} epochs per group (regression head only)")
    print(f"Phase 2: {epochs_phase2} epochs per group (all layers)")
    
    # PHASE 1: Train regression head only
    print("\n=== PHASE 1: Training Regression Head Only ===")
    freeze_layers_except_regression(model)
    
    model.compile(
        optimizer=AdamW(1e-4, weight_decay=1e-6),
        loss={'segmentation_output': dice_metric_loss,
              'regression_output': normalized_regression_loss},
        loss_weights={'segmentation_output': 1.0,
                     'regression_output': 1.0}
    )
    
    for group_idx in range(num_groups):
        success = train_on_group(model, data_generator, group_idx, 1, epochs_phase1, callbacks)
        if not success:
            print(f"Warning: Failed to train on group {group_idx}, continuing...")
    
    # PHASE 2: Fine-tune all layers
    print("\n=== PHASE 2: Fine-tuning All Layers ===")
    unfreeze_all_layers(model)
    
    model.compile(
        optimizer=AdamW(1e-5, weight_decay=1e-6),
        loss={'segmentation_output': dice_metric_loss,
              'regression_output': normalized_regression_loss},
        loss_weights={'segmentation_output': 1.0,
                     'regression_output': 1.0}
    )
    
    for group_idx in range(num_groups):
        success = train_on_group(model, data_generator, group_idx, 2, epochs_phase2, callbacks)
        if not success:
            print(f"Warning: Failed to train on group {group_idx}, continuing...")
    
    print("\n=== Training Complete ===")
    print("Final model saved as: alphapolyp_optimized_model.h5")
    print("Global regression statistics saved as: global_regression_stats.pkl")
    print("Note: Regression outputs are in original scale. Use global_regression_stats.pkl for denormalization.")

if __name__ == "__main__":
    # Define augmentation pipeline
    aug = albu.Compose([
        albu.HorizontalFlip(),
        albu.VerticalFlip(),
        albu.ColorJitter(brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01, always_apply=True),
        albu.Affine(scale=(0.5, 1.5), translate_percent=(-0.125, 0.125), rotate=(-180, 180), shear=(-22.5, 22), always_apply=True),
    ])
    
    main() 