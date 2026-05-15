"""
train_groups.py  —  AlphaPolyp large-dataset training script (tf.data pipeline).

With the tf.data pipeline, the old "group" concept (manually splitting the
dataset into RAM-sized chunks) is no longer needed: tf.data streams and
prefetches batches automatically.  This script is therefore a thin wrapper
around train.py's logic that accepts the same arguments and data layout.

Expected data layout  (identical to train.py)
---------------------------------------------
<root>/
  cyclegan_images/{train,test}/
  images/{train,test}/
  masks/{train,test}/
  train_labels.csv    (columns: Filename, logVolume, x, y, z)
  test_labels.csv

Usage
-----
python train_groups.py --root /path/to/data [--pretrained rapunet_pretrained.h5]
"""

import os
import csv
import time
import pickle
import datetime
import argparse
import numpy as np
import tensorflow as tf
from keras.callbacks import (
    CSVLogger, ModelCheckpoint, TensorBoard,
    ReduceLROnPlateau, EarlyStopping, TerminateOnNaN,
    BackupAndRestore, LambdaCallback,
)
from tensorflow_addons.optimizers import AdamW

from model_architecture.LossFunctions    import dice_metric_loss, normalized_mse_loss
from model_architecture.model            import create_model
from model_architecture.tf_data_pipeline import build_dataset
from model_architecture.training_report  import generate_report

START_TIME = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Train AlphaPolyp — large dataset path')
parser.add_argument('--root',       type=str, required=True,
                    help='Root data directory')
parser.add_argument('--pretrained', type=str, default='rapunet_pretrained.h5',
                    help='Path to pretrained RAPUNet checkpoint (optional)')
parser.add_argument('--cache', type=str, default='',
                    help='tf.data cache: empty=off, "ram"=RAM, or a directory path for disk cache')
args = parser.parse_args()

root = args.root


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

REAL_TRAIN_IMG  = os.path.join(root, 'cyclegan_images', 'train')
SYNTH_TRAIN_IMG = os.path.join(root, 'images',          'train')
TRAIN_MASK_DIR  = os.path.join(root, 'masks',           'train')
REAL_VAL_IMG    = os.path.join(root, 'cyclegan_images', 'test')
SYNTH_VAL_IMG   = os.path.join(root, 'images',          'test')
VAL_MASK_DIR    = os.path.join(root, 'masks',           'test')
TRAIN_CSV       = os.path.join(root, 'train_labels.csv')
VAL_CSV         = os.path.join(root, 'test_labels.csv')

for path, label in [
    (REAL_TRAIN_IMG,  'cyclegan_images/train'),
    (SYNTH_TRAIN_IMG, 'images/train'),
    (TRAIN_MASK_DIR,  'masks/train'),
    (REAL_VAL_IMG,    'cyclegan_images/test'),
    (SYNTH_VAL_IMG,   'images/test'),
    (VAL_MASK_DIR,    'masks/test'),
    (TRAIN_CSV,       'train_labels.csv'),
    (VAL_CSV,         'test_labels.csv'),
]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required path missing — {label}: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE   = 352
FILTERS    = 17
BATCH_SIZE = 8
SEED       = 58800
EPOCHS_P1  = 10
EPOCHS_P2  = 50
LOG_ROOT   = './logs'
os.makedirs(LOG_ROOT, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (shared with train.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_label_map(csv_path):
    label_map = {}
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            try:
                label_map[row['Filename']] = [
                    float(row['logVolume']), float(row['x']),
                    float(row['y']),         float(row['z']),
                ]
            except (KeyError, ValueError) as e:
                print(f"  Warning: skipping row: {e}")
    return label_map


def build_paths(img_dir, mask_dir, label_map):
    img_paths, mask_paths, reg_labels = [], [], []
    skipped = 0
    files = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg')))
    for fname in files:
        base  = os.path.splitext(fname)[0]
        label = None
        for key in (base, base + '_labeled.obj', fname):
            if key in label_map:
                label = label_map[key]
                break
        if label is None:
            skipped += 1
            continue
        mask_path = os.path.join(mask_dir, base + '.png')
        if not os.path.exists(mask_path):
            skipped += 1
            continue
        img_paths.append(os.path.join(img_dir, fname))
        mask_paths.append(mask_path)
        reg_labels.append(label)
    print(f"  {os.path.relpath(img_dir, root)}: {len(img_paths)} matched, {skipped} skipped")
    return img_paths, mask_paths, reg_labels


# ─────────────────────────────────────────────────────────────────────────────
# Labels + path lists
# ─────────────────────────────────────────────────────────────────────────────

print("Loading label maps...")
train_lm = load_label_map(TRAIN_CSV)
val_lm   = load_label_map(VAL_CSV)
print(f"  train: {len(train_lm)} entries  |  val: {len(val_lm)} entries")

print("Building path lists...")
ri_t, rm_t, rr_t = build_paths(REAL_TRAIN_IMG,  TRAIN_MASK_DIR, train_lm)
si_t, sm_t, sr_t = build_paths(SYNTH_TRAIN_IMG, TRAIN_MASK_DIR, train_lm)
ri_v, rm_v, rr_v = build_paths(REAL_VAL_IMG,    VAL_MASK_DIR,   val_lm)
si_v, sm_v, sr_v = build_paths(SYNTH_VAL_IMG,   VAL_MASK_DIR,   val_lm)

train_imgs  = ri_t + si_t;  train_masks = rm_t + sm_t;  train_regs  = rr_t + sr_t
val_imgs    = ri_v + si_v;  val_masks   = rm_v + sm_v;  val_regs    = rr_v + sr_v

if not train_imgs:
    raise RuntimeError("No training samples matched. Check CSV Filename column.")
if not val_imgs:
    raise RuntimeError("No validation samples found.")

print(f"\nDataset: {len(train_imgs)} train  |  {len(val_imgs)} val")


# ─────────────────────────────────────────────────────────────────────────────
# Regression stats (training set only)
# ─────────────────────────────────────────────────────────────────────────────

train_reg_arr = np.array(train_regs, dtype=np.float32)
reg_stats = {
    'min':   train_reg_arr.min(axis=0),
    'max':   train_reg_arr.max(axis=0),
    'mean':  train_reg_arr.mean(axis=0),
    'std':   train_reg_arr.std(axis=0),
    'range': train_reg_arr.max(axis=0) - train_reg_arr.min(axis=0),
}
print("\nRegression stats (training set):")
for i, name in enumerate(['logVolume', 'x', 'y', 'z']):
    print(f"  {name:12s}: min={reg_stats['min'][i]:.3f}  max={reg_stats['max'][i]:.3f}  "
          f"mean={reg_stats['mean'][i]:.3f}")

pickle.dump(reg_stats, open('regression_stats.pkl', 'wb'))
print("Saved regression_stats.pkl")

mse_loss = normalized_mse_loss(reg_stats)


# ─────────────────────────────────────────────────────────────────────────────
# Resolve cache argument
# ─────────────────────────────────────────────────────────────────────────────

cache_arg: bool | str = False
if args.cache == 'ram':
    cache_arg = True
elif args.cache:
    os.makedirs(args.cache, exist_ok=True)
    cache_arg = args.cache


# ─────────────────────────────────────────────────────────────────────────────
# tf.data datasets
# ─────────────────────────────────────────────────────────────────────────────

print("\nBuilding tf.data datasets...")
train_ds = build_dataset(
    train_imgs, train_masks, train_regs,
    img_size=IMG_SIZE, batch_size=BATCH_SIZE,
    training=True, seed=SEED, cache=cache_arg,
)
val_ds = build_dataset(
    val_imgs, val_masks, val_regs,
    img_size=IMG_SIZE, batch_size=BATCH_SIZE,
    training=False, seed=SEED, cache=False,
)
print("Datasets ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

model = create_model(
    img_height=IMG_SIZE, img_width=IMG_SIZE, input_channels=3,
    out_classes=1, starting_filters=FILTERS,
    bias=reg_stats['mean'],
)
print(f"\nModel: {model.count_params():,} parameters")

if os.path.exists(args.pretrained):
    model.load_weights(args.pretrained, by_name=True, skip_mismatch=True)
    print(f"Loaded pretrained weights: {args.pretrained}")

def _freeze_except_regression(m):
    for layer in m.layers:
        layer.trainable = ('regression_output' in layer.name)

def _unfreeze_all(m):
    for layer in m.layers:
        layer.trainable = True


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def _lr_log(epoch, logs):
    try:
        logs['lr'] = float(model.optimizer.learning_rate)
    except Exception:
        pass

phase1_callbacks = [
    CSVLogger(f'{LOG_ROOT}/phase1_{run_id}.csv'),
    TensorBoard(log_dir=f'{LOG_ROOT}/tb_phase1_{run_id}'),
    ModelCheckpoint(
        f'{LOG_ROOT}/phase1_best_reg_{run_id}.weights.h5',
        monitor='val_regression_output_loss', save_best_only=True,
        save_weights_only=True, verbose=1,
    ),
    ReduceLROnPlateau(
        monitor='val_regression_output_loss', factor=0.5,
        patience=4, min_lr=1e-7, verbose=1,
    ),
    TerminateOnNaN(),
    BackupAndRestore(backup_dir=os.path.join(LOG_ROOT, 'backup_phase1')),
    LambdaCallback(on_epoch_end=_lr_log),
]

phase2_callbacks = [
    CSVLogger(f'{LOG_ROOT}/phase2_{run_id}.csv'),
    TensorBoard(log_dir=f'{LOG_ROOT}/tb_phase2_{run_id}'),
    ModelCheckpoint(
        f'{LOG_ROOT}/phase2_best_reg_{run_id}.weights.h5',
        monitor='val_regression_output_loss', save_best_only=True,
        save_weights_only=True, verbose=1,
    ),
    ModelCheckpoint(
        f'{LOG_ROOT}/phase2_best_seg_{run_id}.weights.h5',
        monitor='val_segmentation_output_loss', mode='min',
        save_best_only=True, save_weights_only=True, verbose=1,
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=6,
        cooldown=2, min_lr=1e-7, verbose=1,
    ),
    EarlyStopping(
        monitor='val_loss', patience=16,
        restore_best_weights=True, verbose=1,
    ),
    TerminateOnNaN(),
    BackupAndRestore(backup_dir=os.path.join(LOG_ROOT, 'backup_phase2')),
    LambdaCallback(on_epoch_end=_lr_log),
]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — regression head warm-up
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f" Phase 1 — {EPOCHS_P1} epochs  (regression head only)")
print(f"{'='*60}")

_freeze_except_regression(model)
model.compile(
    optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-6),
    loss={'segmentation_output': dice_metric_loss,
          'regression_output':   mse_loss},
    loss_weights={'segmentation_output': 0.0,
                  'regression_output':   1.0},
)
model.fit(train_ds, validation_data=val_ds,
          epochs=EPOCHS_P1, callbacks=phase1_callbacks)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — full fine-tune
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f" Phase 2 — up to {EPOCHS_P2} epochs  (all layers, EarlyStopping active)")
print(f"{'='*60}")

_unfreeze_all(model)
model.compile(
    optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-6),
    loss={'segmentation_output': dice_metric_loss,
          'regression_output':   mse_loss},
    loss_weights={'segmentation_output': 1.0,
                  'regression_output':   1.0},
)
model.fit(train_ds, validation_data=val_ds,
          epochs=EPOCHS_P2, callbacks=phase2_callbacks)

# Save final best weights (EarlyStopping has restored them already)
final_ckpt = os.path.join(LOG_ROOT, f'final_best_{run_id}.weights.h5')
model.save_weights(final_ckpt)

report_path = generate_report(
    run_id        = run_id,
    log_root      = LOG_ROOT,
    reg_stats     = reg_stats,
    n_train       = len(train_imgs),
    n_val         = len(val_imgs),
    n_real_train  = len(ri_t),
    n_synth_train = len(si_t),
    n_real_val    = len(ri_v),
    n_synth_val   = len(si_v),
    img_size      = IMG_SIZE,
    batch_size    = BATCH_SIZE,
    filters       = FILTERS,
    epochs_p1     = EPOCHS_P1,
    epochs_p2     = EPOCHS_P2,
    n_params      = model.count_params(),
    pretrained    = args.pretrained,
    pretrained_loaded = os.path.exists(args.pretrained),
    start_time    = START_TIME,
)

print(f"\n{'='*60}")
print(f" Training complete.")
print(f"{'='*60}")
print(f"  Report        : {report_path}")
print(f"  Final weights : {final_ckpt}")
print(f"  Stats         : regression_stats.pkl")
print(f"  TensorBoard   : tensorboard --logdir {LOG_ROOT}")
