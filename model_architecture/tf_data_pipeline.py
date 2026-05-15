"""
tf_data_pipeline.py  —  GPU-friendly tf.data input pipeline for AlphaPolyp.

Replaces the old numpy / albumentations augment_batch approach:
  - Parallel file I/O, decode, and resize  (num_parallel_calls=AUTOTUNE)
  - Synchronised image + mask augmentation via stateless TF ops and tfa.image
  - Prefetch keeps the GPU fed while the CPU prepares the next batch
  - Optional RAM or disk cache after decode (avoids re-reading each epoch)

Public API
----------
build_dataset(img_paths, mask_paths, reg_labels, *, img_size, batch_size,
              training, seed, cache) -> tf.data.Dataset
"""

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

# Module-level constant updated by build_dataset so the map functions
# (which are traced as tf.functions) see the current value.
_IMG_SIZE: int = 352


# ─────────────────────────────────────────────────────────────────────────────
# I/O — decode + resize + normalise
# ─────────────────────────────────────────────────────────────────────────────

def _load(img_path: tf.Tensor,
          mask_path: tf.Tensor,
          reg_label: tf.Tensor):
    """
    Load one (image, mask, regression_label) triplet from disk.

    Returns
    -------
    image  : (IMG_SIZE, IMG_SIZE, 3)  float32  in [0, 1]
    mask   : (IMG_SIZE, IMG_SIZE, 1)  float32  binary {0, 1}
    reg    : (4,)                     float32  [logVol, x, y, z]
    """
    # ── image ────────────────────────────────────────────────────────────────
    raw = tf.io.read_file(img_path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])          # fix rank so downstream ops are happy
    img = tf.image.resize(img, [_IMG_SIZE, _IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0

    # ── mask ─────────────────────────────────────────────────────────────────
    raw_msk = tf.io.read_file(mask_path)
    msk = tf.image.decode_image(raw_msk, channels=1, expand_animations=False)
    msk.set_shape([None, None, 1])
    # Nearest-neighbour resize keeps the mask binary (no interpolation artefacts)
    msk = tf.cast(msk, tf.float32)
    msk = tf.image.resize(msk, [_IMG_SIZE, _IMG_SIZE], method='nearest')
    msk = tf.cast(msk > 127.5, tf.float32)

    return img, msk, reg_label


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation — all transforms applied synchronously to image AND mask
# ─────────────────────────────────────────────────────────────────────────────

def _augment(img: tf.Tensor,
             msk: tf.Tensor,
             reg: tf.Tensor):
    """
    Randomised augmentation for one (image, mask) pair.

    Transforms applied (same random parameters to image and mask):
      • Horizontal flip
      • Vertical flip
      • Rotation  ±180°
      • Translation  ±12.5 % of image size
      • Colour jitter  (brightness / contrast / saturation / hue)  — image only

    Masks use NEAREST interpolation throughout so they stay binary.
    A final cast re-binarises any floating-point residue.
    """
    # ── random flips (stateless → same decision for image and mask) ──────────
    seed_h = tf.random.uniform([2], 0, tf.int32.max, dtype=tf.int32)
    seed_v = tf.random.uniform([2], 0, tf.int32.max, dtype=tf.int32)

    img = tf.image.stateless_random_flip_left_right(img, seed=seed_h)
    msk = tf.image.stateless_random_flip_left_right(msk, seed=seed_h)
    img = tf.image.stateless_random_flip_up_down(img, seed=seed_v)
    msk = tf.image.stateless_random_flip_up_down(msk, seed=seed_v)

    # ── rotation (±180°) ─────────────────────────────────────────────────────
    angle = tf.random.uniform((), -np.pi, np.pi)
    img = tfa.image.rotate(img, angle,
                           interpolation='BILINEAR',
                           fill_mode='constant', fill_value=0.0)
    msk = tfa.image.rotate(msk, angle,
                           interpolation='NEAREST',
                           fill_mode='constant', fill_value=0.0)

    # ── translation (±12.5 % of image dimensions) ────────────────────────────
    hw = tf.cast(_IMG_SIZE, tf.float32)
    dx = tf.random.uniform((), -0.125 * hw, 0.125 * hw)
    dy = tf.random.uniform((), -0.125 * hw, 0.125 * hw)
    translations = tf.stack([dx, dy])       # shape (2,) → [x_shift, y_shift]
    img = tfa.image.translate(img, translations,
                              interpolation='BILINEAR',
                              fill_mode='constant', fill_value=0.0)
    msk = tfa.image.translate(msk, translations,
                              interpolation='NEAREST',
                              fill_mode='constant', fill_value=0.0)

    # ── colour jitter (image only) ────────────────────────────────────────────
    img = tf.image.random_brightness(img, max_delta=0.4)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
    img = tf.image.random_hue(img, max_delta=0.01)
    img = tf.clip_by_value(img, 0.0, 1.0)

    # ── re-binarise mask: spatial interpolation may leave continuous values ───
    msk = tf.cast(msk > 0.5, tf.float32)

    return img, msk, reg


# ─────────────────────────────────────────────────────────────────────────────
# Output formatter — packs targets into the dict Keras expects
# ─────────────────────────────────────────────────────────────────────────────

def _to_keras_format(img, msk, reg):
    return img, {
        'segmentation_output': msk,
        'regression_output':   reg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(img_paths,
                  mask_paths,
                  reg_labels,
                  *,
                  img_size:    int  = 352,
                  batch_size:  int  = 8,
                  training:    bool = True,
                  seed:        int  = 58800,
                  cache              = False) -> tf.data.Dataset:
    """
    Build a prefetch-ready tf.data.Dataset for model.fit / model.evaluate.

    Parameters
    ----------
    img_paths   : list[str]      Absolute paths to input images.
    mask_paths  : list[str]      Absolute paths to corresponding binary masks.
    reg_labels  : array-like     Shape (N, 4) — [logVolume, x, y, z].
    img_size    : int            Resize target (must be 352 for CAFormerS18).
    batch_size  : int            Mini-batch size.
    training    : bool           True → shuffle + augment; False → deterministic.
    seed        : int            RNG seed for shuffle.
    cache       : bool | str     False  → no cache (re-reads disk every epoch).
                                 True   → RAM cache (after decode; good for
                                          datasets that fit in system RAM).
                                 str    → disk-cache path (e.g. '/tmp/ap_cache').

    Returns
    -------
    tf.data.Dataset yielding:
        (image_batch, {'segmentation_output': mask_batch,
                       'regression_output':   reg_batch})
    """
    global _IMG_SIZE
    _IMG_SIZE = img_size

    n = len(img_paths)
    assert n > 0,                            "img_paths is empty"
    assert n == len(mask_paths),             f"len mismatch: {n} imgs vs {len(mask_paths)} masks"
    assert n == len(reg_labels),             f"len mismatch: {n} imgs vs {len(reg_labels)} labels"

    reg_arr = np.asarray(reg_labels, dtype=np.float32)
    assert reg_arr.shape == (n, 4),          f"reg_labels must be (N, 4), got {reg_arr.shape}"

    ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(img_paths,   dtype=tf.string),
        tf.constant(mask_paths,  dtype=tf.string),
        tf.constant(reg_arr,     dtype=tf.float32),
    ))

    if training:
        ds = ds.shuffle(buffer_size=n, seed=seed, reshuffle_each_iteration=True)

    # Parallel decode + resize
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    # Optional cache — snapshots decoded images so disk I/O only happens once.
    # Place AFTER decode so we cache float32 tensors, not raw bytes.
    # Place BEFORE augment so each epoch still sees fresh random transforms.
    if cache is True:
        ds = ds.cache()
    elif isinstance(cache, str):
        ds = ds.cache(cache)

    # Augmentation (training only)
    if training:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    # Reformat output for Keras multi-output model
    ds = ds.map(_to_keras_format, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch + prefetch
    # drop_remainder=True keeps training batch sizes uniform (avoids retracing)
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
