"""
sanity_check.py — Pre-experiment validation for AlphaPolyp.

Usage:
    python sanity_check.py --root /path/to/data --csv /path/to/labels.csv [--stats global_regression_stats.pkl]

Exit code 0 = all checks passed.  Non-zero = at least one check failed.
"""

import sys
import os
import argparse
import traceback

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_results = []

def check(name, fn):
    """Run fn(); record PASS or FAIL with message."""
    try:
        msg = fn()
        _results.append(("PASS", name, msg or ""))
        return True
    except Exception as e:
        _results.append(("FAIL", name, str(e)))
        return False


def print_summary():
    print("\n" + "=" * 70)
    print(f"{'CHECK':<50} {'STATUS'}")
    print("=" * 70)
    passed = failed = 0
    for status, name, msg in _results:
        tag = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {name:<48} {tag}")
        if status == "FAIL":
            print(f"       → {msg}")
            failed += 1
        else:
            passed += 1
    print("=" * 70)
    print(f"  {passed} passed, {failed} failed")
    print("=" * 70)
    return failed == 0


# ──────────────────────────────────────────────────────────────────────────────
# Check functions
# ──────────────────────────────────────────────────────────────────────────────

def check_python_version():
    v = sys.version_info
    assert v >= (3, 8), f"Python >= 3.8 required, got {v.major}.{v.minor}"
    return f"Python {v.major}.{v.minor}.{v.micro}"


def check_tensorflow():
    import tensorflow as tf
    ver = tf.__version__
    assert ver.startswith("2."), f"TensorFlow 2.x required, got {ver}"
    return f"TensorFlow {ver}"


def check_gpu():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    assert gpus, "No GPU found. Training will be very slow on CPU."
    return f"{len(gpus)} GPU(s): {[g.name for g in gpus]}"


def check_mixed_precision_safe():
    import tensorflow as tf
    # Mixed precision is safe on Ampere+ (compute capability >= 8.0).
    # We just report status; user decides whether to enable it.
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return "skipped (no GPU)"
    return "GPU present; enable mixed precision manually if compute cap >= 8.0"


def check_imports():
    import numpy          # noqa
    import cv2            # noqa
    import albumentations # noqa
    import sklearn        # noqa
    import tqdm           # noqa
    from keras_cv_attention_models import caformer  # noqa
    from tensorflow_addons.optimizers import AdamW  # noqa
    return "all required packages importable"


def check_model_architecture_imports():
    from model_architecture.LossFunctions    import dice_metric_loss, normalized_mse_loss  # noqa
    from model_architecture.DiceLoss         import dice_metric_loss as _dl  # backward-compat shim  # noqa
    from model_architecture.ImageLoader2D    import load_images_masks_from_drive  # noqa
    from model_architecture.model            import create_model  # noqa
    from model_architecture.RAPU_blocks      import resnet_block, RAPU, convf_bn_act, SBA  # noqa
    from model_architecture.tf_data_pipeline import build_dataset  # noqa
    return "all model_architecture modules importable (incl. tf_data_pipeline)"


def check_data_dirs(real_img_dir, real_mask_dir, synth_img_dir, synth_mask_dir):
    missing = []
    for path, label in [
        (real_img_dir,  "real_img_dir"),
        (real_mask_dir, "real_mask_dir"),
        (synth_img_dir, "synth_img_dir"),
        (synth_mask_dir,"synth_mask_dir"),
    ]:
        if not os.path.isdir(path):
            missing.append(f"{label}: {path}")
    assert not missing, "Missing directories:\n  " + "\n  ".join(missing)
    return "all data directories exist"


def check_csv_schema(csv_path):
    import csv as _csv
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
    required_cols = {"Filename", "logVolume", "x", "y", "z"}
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = _csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = required_cols - cols
        assert not missing, f"CSV missing columns: {missing}. Found: {cols}"
        rows = list(reader)
    assert rows, "CSV has no data rows"
    # Spot-check first row for parseable floats
    row = rows[0]
    for col in ["logVolume", "x", "y", "z"]:
        float(row[col])
    return f"{len(rows)} rows, columns OK"


def check_image_mask_pairing(img_dir, mask_dir, sample_limit=20):
    """Verify that masks exist for (up to sample_limit) images."""
    files = sorted([f for f in os.listdir(img_dir)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))])[:sample_limit]
    assert files, f"No image files found in {img_dir}"
    missing = []
    for fname in files:
        mask_name = os.path.splitext(fname)[0] + '.png'
        if not os.path.exists(os.path.join(mask_dir, mask_name)):
            missing.append(fname)
    assert not missing, f"{len(missing)} images missing masks: {missing[:5]}..."
    return f"checked {len(files)} image/mask pairs — all matched"


def check_csv_filename_key_pairing(img_dir, csv_path, sample_limit=20):
    """
    Verify that at least some image filenames map to CSV keys.
    The expected key format is: {base}_labeled.obj
    e.g., image 'polyp_001.jpg' → key 'polyp_001_labeled.obj'
    """
    import csv as _csv
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        label_keys = {row['Filename'] for row in _csv.DictReader(f)}

    files = sorted([f for f in os.listdir(img_dir)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))])[:sample_limit]
    matched = []
    for fname in files:
        key = os.path.splitext(fname)[0] + '_labeled.obj'
        if key in label_keys:
            matched.append(fname)
    frac = len(matched) / max(len(files), 1)
    assert frac > 0, (
        f"ZERO images in {img_dir} matched any CSV key.\n"
        f"  Example image base: {os.path.splitext(files[0])[0] if files else 'N/A'}\n"
        f"  Expected key format: <base>_labeled.obj\n"
        f"  Example CSV keys: {list(label_keys)[:3]}"
    )
    return f"{len(matched)}/{len(files)} sampled images have CSV labels"


def check_regression_stats(stats_path):
    import pickle
    assert os.path.exists(stats_path), f"Stats file not found: {stats_path}"
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    required_keys = {'min', 'max', 'mean', 'std', 'range'}
    missing = required_keys - set(stats.keys())
    assert not missing, f"Stats dict missing keys: {missing}"
    for k in required_keys:
        assert len(stats[k]) == 4, f"stats['{k}'] should have 4 elements (logVol,x,y,z)"
    return "regression stats file valid (4-element arrays for min/max/mean/std/range)"


def check_model_build():
    from model_architecture.model import create_model
    model = create_model(img_height=352, img_width=352, input_channels=3,
                         out_classes=1, starting_filters=17)
    assert len(model.outputs) == 2, "Model should have 2 outputs"
    seg_shape = tuple(model.outputs[0].shape[1:])
    reg_shape = tuple(model.outputs[1].shape[1:])
    assert seg_shape == (352, 352, 1), f"Segmentation output shape wrong: {seg_shape}"
    assert reg_shape == (4,), f"Regression output shape wrong: {reg_shape}"
    n_params = model.count_params()
    return f"model built OK — seg {seg_shape}, reg {reg_shape}, {n_params:,} params"


def check_loss_functions():
    import tensorflow as tf
    import numpy as np
    from model_architecture.LossFunctions import dice_metric_loss, normalized_mse_loss

    # Dice loss on a perfect prediction
    y = tf.ones([2, 10, 10, 1], dtype=tf.float32)
    loss_perfect = float(dice_metric_loss(y, y))
    assert loss_perfect < 1e-4, f"Dice loss on perfect pred should be ~0, got {loss_perfect}"

    # Dice loss on all-wrong prediction
    loss_wrong = float(dice_metric_loss(y, tf.zeros_like(y)))
    assert loss_wrong > 0.9, f"Dice loss on all-wrong pred should be ~1, got {loss_wrong}"

    # Normalised MSE loss — identical predictions → 0
    stats = {'min': np.zeros(4), 'max': np.ones(4) * 100, 'range': np.ones(4) * 100}
    mse_fn = normalized_mse_loss(stats)
    y_reg = tf.constant([[1.0, 2.0, 3.0, 4.0]])
    loss_mse = float(mse_fn(y_reg, y_reg))
    assert loss_mse < 1e-6, f"Normalised MSE on identical tensors should be 0, got {loss_mse}"

    return "dice_metric_loss and normalized_mse_loss behave correctly"


def check_forward_pass(img_dir, mask_dir):
    """Load two real images, run a forward pass, validate output shapes and ranges."""
    import numpy as np
    import tensorflow as tf
    from model_architecture.ImageLoader2D import load_images_masks_from_drive
    from model_architecture.model import create_model

    X, Y = load_images_masks_from_drive(img_dir, mask_dir, img_size=352)
    assert len(X) >= 1, f"No images loaded from {img_dir}"
    batch = X[:2]   # small batch

    model = create_model(img_height=352, img_width=352, input_channels=3,
                         out_classes=1, starting_filters=17)
    seg_pred, reg_pred = model.predict(batch, verbose=0)

    assert seg_pred.shape == (len(batch), 352, 352, 1), f"Seg shape: {seg_pred.shape}"
    assert reg_pred.shape == (len(batch), 4), f"Reg shape: {reg_pred.shape}"
    assert np.all(seg_pred >= 0) and np.all(seg_pred <= 1), \
        f"Segmentation output outside [0,1]: min={seg_pred.min()}, max={seg_pred.max()}"
    assert not np.any(np.isnan(reg_pred)), "NaN in regression output"
    return (f"forward pass OK — seg {seg_pred.shape}, reg {reg_pred.shape}, "
            f"seg range [{seg_pred.min():.3f}, {seg_pred.max():.3f}]")


def check_mini_train_step(img_dir, mask_dir, csv_path):
    """One mini-batch gradient update: verify loss goes through without NaN/error."""
    import csv as _csv
    import numpy as np
    import tensorflow as tf
    from tensorflow_addons.optimizers import AdamW
    from model_architecture.ImageLoader2D import load_images_masks_from_drive
    from model_architecture.LossFunctions import dice_metric_loss, normalized_mse_loss
    from model_architecture.model import create_model

    X, Y = load_images_masks_from_drive(img_dir, mask_dir, img_size=352)
    assert len(X) >= 2, "Need at least 2 samples for mini-train check"
    batch_x = X[:2]
    batch_m = Y[:2]

    # Use dummy regression labels (zeros)
    batch_r = np.zeros((2, 4), dtype=np.float32)

    stats = {
        'min': np.zeros(4, dtype=np.float32),
        'max': np.ones(4, dtype=np.float32) * 10,
        'range': np.ones(4, dtype=np.float32) * 10,
    }
    mse_fn = normalized_mse_loss(stats)

    model = create_model(img_height=352, img_width=352, input_channels=3,
                         out_classes=1, starting_filters=17)
    model.compile(
        optimizer=AdamW(1e-4, weight_decay=1e-6),
        loss={'segmentation_output': dice_metric_loss,
              'regression_output': mse_fn},
        loss_weights={'segmentation_output': 1.0, 'regression_output': 1.0},
    )

    history = model.fit(
        batch_x,
        {'segmentation_output': batch_m, 'regression_output': batch_r},
        epochs=1, batch_size=2, verbose=0
    )
    total_loss = history.history['loss'][0]
    assert not np.isnan(total_loss), f"NaN loss after one step: {total_loss}"
    assert total_loss < 1e6, f"Loss exploded: {total_loss}"
    return f"mini train step OK — loss = {total_loss:.4f}"


def check_tf_data_pipeline(img_dir, mask_dir):
    """
    Build a small tf.data dataset from real images and verify:
    - output shapes are correct for the multi-output Keras model
    - image values are in [0, 1]
    - mask values are binary {0, 1}
    - regression labels have correct shape
    """
    import numpy as np
    import tensorflow as tf
    from model_architecture.tf_data_pipeline import build_dataset

    # Collect up to 4 image/mask pairs to test
    files = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg')))[:4]
    assert files, f"No images in {img_dir}"

    img_paths, mask_paths = [], []
    for fname in files:
        base = os.path.splitext(fname)[0]
        mask_path = os.path.join(mask_dir, base + '.png')
        if os.path.exists(mask_path):
            img_paths.append(os.path.join(img_dir, fname))
            mask_paths.append(mask_path)

    assert len(img_paths) >= 2, f"Need >=2 matched pairs; found {len(img_paths)}"
    dummy_regs = np.zeros((len(img_paths), 4), dtype=np.float32)

    ds = build_dataset(
        img_paths, mask_paths, dummy_regs,
        img_size=352, batch_size=2,
        training=True, seed=42, cache=False,
    )

    # Grab one batch
    batch = next(iter(ds))
    imgs, targets = batch
    masks = targets['segmentation_output']
    regs  = targets['regression_output']

    assert imgs.shape  == (2, 352, 352, 3), f"Image batch shape: {imgs.shape}"
    assert masks.shape == (2, 352, 352, 1), f"Mask batch shape:  {masks.shape}"
    assert regs.shape  == (2, 4),           f"Reg batch shape:   {regs.shape}"

    img_np  = imgs.numpy()
    mask_np = masks.numpy()
    assert img_np.min()  >= 0.0 and img_np.max()  <= 1.0, \
        f"Image values out of [0,1]: [{img_np.min():.3f}, {img_np.max():.3f}]"
    unique_mask = set(np.unique(mask_np).tolist())
    assert unique_mask <= {0.0, 1.0}, \
        f"Mask not binary after augmentation: {unique_mask}"

    return (f"tf.data pipeline OK — imgs{imgs.shape} masks{masks.shape} regs{regs.shape}, "
            f"img range [{img_np.min():.3f},{img_np.max():.3f}], mask values {unique_mask}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Pre-experiment sanity checks for AlphaPolyp')
    parser.add_argument('--root',  type=str, required=True,
                        help='Root data directory (same as --root in train.py)')
    parser.add_argument('--stats', type=str, default=None,
                        help='Path to regression_stats.pkl (optional; checked if provided)')
    args = parser.parse_args()

    root = args.root

    # Derived paths matching train.py layout
    real_train_img  = os.path.join(root, 'cyclegan_images', 'train')
    synth_train_img = os.path.join(root, 'images',          'train')
    train_mask_dir  = os.path.join(root, 'masks',           'train')
    real_val_img    = os.path.join(root, 'cyclegan_images', 'test')
    synth_val_img   = os.path.join(root, 'images',          'test')
    val_mask_dir    = os.path.join(root, 'masks',           'test')
    train_csv       = os.path.join(root, 'train_labels.csv')
    val_csv         = os.path.join(root, 'test_labels.csv')

    print("=" * 70)
    print("  AlphaPolyp Sanity Check")
    print("=" * 70)

    # ── Environment ──────────────────────────────────────────────────────────
    check("Python version >= 3.8",                 check_python_version)
    check("TensorFlow 2.x importable",             check_tensorflow)
    check("GPU detected",                          check_gpu)
    check("Mixed precision note",                  check_mixed_precision_safe)
    check("All required packages importable",      check_imports)
    check("model_architecture modules importable", check_model_architecture_imports)

    # ── Data ─────────────────────────────────────────────────────────────────
    check("Data directories exist",
          lambda: check_data_dirs(real_train_img, train_mask_dir,
                                  synth_train_img, train_mask_dir))
    check("train_labels.csv schema",
          lambda: check_csv_schema(train_csv))
    check("test_labels.csv schema",
          lambda: check_csv_schema(val_csv))
    check("Image/mask pairing (real train)",
          lambda: check_image_mask_pairing(real_train_img, train_mask_dir))
    check("Image/mask pairing (synth train)",
          lambda: check_image_mask_pairing(synth_train_img, train_mask_dir))
    check("CSV keys match real-train images",
          lambda: check_csv_filename_key_pairing(real_train_img, train_csv))
    check("CSV keys match synth-train images",
          lambda: check_csv_filename_key_pairing(synth_train_img, train_csv))
    if args.stats:
        check("Regression stats file valid",
              lambda: check_regression_stats(args.stats))

    # ── Model ────────────────────────────────────────────────────────────────
    check("Model builds (output shapes correct)",  check_model_build)
    check("Loss functions numerically correct",    check_loss_functions)
    check("tf.data pipeline (shapes + binary mask + value range)",
          lambda: check_tf_data_pipeline(real_train_img, train_mask_dir))
    check("Forward pass (shape + value range)",
          lambda: check_forward_pass(real_train_img, train_mask_dir))
    check("Mini train step (gradient update, no NaN)",
          lambda: check_mini_train_step(real_train_img, train_mask_dir, train_csv))

    ok = print_summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
