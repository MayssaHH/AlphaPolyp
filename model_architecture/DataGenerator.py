import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pickle
import random
from sklearn.model_selection import train_test_split


class LargeDatasetGenerator:
    """
    Generator for handling large datasets by splitting into groups
    and loading data on-demand to prevent memory issues.
    """
    
    def __init__(self, real_img_dir, real_mask_dir, synth_img_dir, synth_mask_dir, 
                 csv_labels, img_size=352, group_size=2500, batch_size=8, seed=58800):
        self.real_img_dir = real_img_dir
        self.real_mask_dir = real_mask_dir
        self.synth_img_dir = synth_img_dir
        self.synth_mask_dir = synth_mask_dir
        self.csv_labels = csv_labels
        self.img_size = img_size
        self.group_size = group_size
        self.batch_size = batch_size
        self.seed = seed
        
        # Load label map
        self.label_map = self._load_label_map()
        if self.label_map is None:
            raise ValueError("Failed to load label map")
        
        # Get file lists
        self.real_files = self._filter_labeled(real_img_dir)
        self.synth_files = self._filter_labeled(synth_img_dir)
        
        if not self.real_files or not self.synth_files:
            raise ValueError("No labeled files found")
        
        print(f"Found {len(self.real_files)} real files and {len(self.synth_files)} synthetic files")
        
        # Create groups
        self._create_groups()
        
        
    def _load_label_map(self):
        """Load label map from CSV"""
        label_map = {}
        try:
            import csv
            with open(self.csv_labels, newline='', encoding='utf-8-sig') as f:
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
        except Exception as e:
            print(f"ERROR: Failed to read CSV file: {e}")
            return None
        
        print(f"Loaded {len(label_map)} labels from CSV")
        return label_map
    
    def _filter_labeled(self, img_dir):
        """Return sorted list of filenames that have labels"""
        out = []
        if not os.path.exists(img_dir):
            print(f"ERROR: Image directory does not exist: {img_dir}")
            return out
            
        for fn in os.listdir(img_dir):
            if not fn.lower().endswith(('.jpg','.png','.jpeg')): 
                continue
            base = os.path.splitext(fn)[0]
            base_no_zeros = base.lstrip('0')  # Remove leading zeros
            if base_no_zeros in self.label_map:
                out.append(fn)
        return sorted(out)

        
    
    def _get_reg_labels(self, file_list):
        """Return regression labels for file list"""
        regs = []
        for fname in file_list:
            base = os.path.splitext(fname)[0]
            base_no_zeros = base.lstrip('0')  # Remove leading zeros
            if self.label_map is not None and base_no_zeros in self.label_map:
                regs.append(self.label_map[base_no_zeros])
        return np.array(regs, dtype=np.float32)
    
    def _create_groups(self):
        """Split data into groups"""
        # Interleave real and synthetic files
        all_files = []
        n = min(len(self.real_files), len(self.synth_files))
        for i in range(n):
            all_files.append(('real', self.real_files[i]))
            all_files.append(('synth', self.synth_files[i]))
        
        # Add remaining files
        if len(self.real_files) > n:
            for i in range(n, len(self.real_files)):
                all_files.append(('real', self.real_files[i]))
        if len(self.synth_files) > n:
            for i in range(n, len(self.synth_files)):
                all_files.append(('synth', self.synth_files[i]))
        
        # Shuffle and split into groups
        random.seed(self.seed)
        random.shuffle(all_files)
        
        self.groups = []
        for i in range(0, len(all_files), self.group_size):
            group = all_files[i:i + self.group_size]
            self.groups.append(group)
        
        print(f"Created {len(self.groups)} groups of size ~{self.group_size}")
    
    
    def load_group_data(self, group_idx):
        """Load data for a specific group"""
        if group_idx >= len(self.groups):
            raise ValueError(f"Group index {group_idx} out of range")
        
        group = self.groups[group_idx]
        real_files = [f[1] for f in group if f[0] == 'real']
        synth_files = [f[1] for f in group if f[0] == 'synth']
        
        # Load images and masks
        X_real, Y_real_mask = self._load_images_masks(real_files, self.real_img_dir, self.real_mask_dir)
        X_synth, Y_synth_mask = self._load_images_masks(synth_files, self.synth_img_dir, self.synth_mask_dir)
        
        # Load regression labels
        Y_real_reg = self._get_reg_labels(real_files) if real_files else np.array([], dtype=np.float32)
        Y_synth_reg = self._get_reg_labels(synth_files) if synth_files else np.array([], dtype=np.float32)
        
        # Combine data
        X_all = []
        Y_all_mask = []
        Y_all_reg = []
        
        if len(X_real) > 0:
            X_all.extend(X_real)
            Y_all_mask.extend(Y_real_mask)
            Y_all_reg.extend(Y_real_reg)
        
        if len(X_synth) > 0:
            X_all.extend(X_synth)
            Y_all_mask.extend(Y_synth_mask)
            Y_all_reg.extend(Y_synth_reg)
        
        return (np.array(X_all, dtype=np.float32),
                np.array(Y_all_mask, dtype=np.float32),
                np.array(Y_all_reg, dtype=np.float32))
    
    def _load_images_masks(self, file_list, img_dir, mask_dir):
        """Load images and masks for a list of files"""
        if not file_list:
            return [], []
        
        X = []
        Y = []
        
        for fname in file_list:
            img_path = os.path.join(img_dir, fname)
            mask_name = os.path.splitext(fname)[0] + '.png'
            mask_path = os.path.join(mask_dir, mask_name)
            
            if not os.path.exists(mask_path):
                print(f"Warning: mask missing for {fname}")
                continue
            
            # Load image
            img = tf.io.read_file(img_path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.resize(img, [self.img_size, self.img_size])
            img = tf.cast(img, tf.float32) / 255.0
            
            # Load mask
            msk = tf.io.read_file(mask_path)
            msk = tf.image.decode_image(msk, channels=1)
            msk = tf.image.resize(msk, [self.img_size, self.img_size])
            msk = tf.cast(msk, tf.float32)
            msk = tf.where(msk > 0.5, 1.0, 0.0)
            
            X.append(img.numpy())
            Y.append(msk.numpy())
        
        return X, Y
    
    def get_group_train_val_split(self, group_idx, val_split=0.1):
        """Get train/val split for a specific group"""
        X, Y_mask, Y_reg = self.load_group_data(group_idx)
        
        if len(X) < 10:
            print(f"Warning: Group {group_idx} has only {len(X)} samples")
            return X, Y_mask, Y_reg, np.array([]), np.array([]), np.array([])
        
        # Split the data
        x_train, x_val, y_train_mask, y_val_mask, y_train_reg, y_val_reg = \
            train_test_split(
                X, Y_mask, Y_reg,
                test_size=val_split, shuffle=True, random_state=self.seed
            )
        
        return x_train, y_train_mask, y_train_reg, x_val, y_val_mask, y_val_reg
    
    def get_num_groups(self):
        """Get total number of groups"""
        return len(self.groups)
    
    def get_group_info(self, group_idx):
        """Get information about a specific group"""
        if group_idx >= len(self.groups):
            return None
        
        group = self.groups[group_idx]
        real_count = len([f for f in group if f[0] == 'real'])
        synth_count = len([f for f in group if f[0] == 'synth'])
        
        return {
            'total_samples': len(group),
            'real_samples': real_count,
            'synthetic_samples': synth_count,
            'files': group
        }


def create_minmax_normalized_mse_loss(reg_stats):
    """
    Create MSE loss normalized using min-max scaling from statistics
    """
    reg_min, reg_max = reg_stats['min'], reg_stats['max']
    reg_range = reg_max - reg_min + 1e-8

    def minmax_normalized_mse_loss(y_true, y_pred):
        y_true_norm = (y_true - reg_min) / reg_range
        y_pred_norm = (y_pred - reg_min) / reg_range
        
        return tf.keras.losses.mean_squared_error(y_true_norm, y_pred_norm)
    
    return minmax_normalized_mse_loss 