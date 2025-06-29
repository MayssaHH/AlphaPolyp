import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def load_images_masks_from_drive(img_dir, mask_dir, img_size):
    """
    Load all images and masks from directories on Google Drive.
    Returns:
      X: np.array of shape [N, img_size, img_size, 3]
      Y: np.array of shape [N, img_size, img_size, 1]
    """
    file_list = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    X = []
    Y = []
    for fname in tqdm(file_list):
        img_path = os.path.join(img_dir, fname)
        mask_name = os.path.splitext(fname)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            print(f"Warning: mask missing for {fname}")
            continue
        # load image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 255.0
        # load mask
        msk = tf.io.read_file(mask_path)
        msk = tf.image.decode_image(msk, channels=1)
        msk = tf.image.resize(msk, [img_size, img_size])
        msk = tf.cast(msk, tf.float32)
        msk = tf.where(msk>0.5, 1.0, 0.0)
        X.append(img.numpy())
        Y.append(msk.numpy())
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return X, Y
