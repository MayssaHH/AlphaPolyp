"""
training_report.py  —  End-of-training markdown report generator.

Called by train.py and train_groups.py after both phases complete.
Reads the CSVLogger output files produced during training and emits a
self-contained .md file in the logs directory.
"""

import os
import csv
import datetime
from typing import Optional


def _read_csv_log(path: str) -> list[dict]:
    """Return list of dicts from a Keras CSVLogger output file."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append({k: _try_float(v) for k, v in row.items()})
    return rows


def _try_float(v: str):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _best_row(rows: list[dict], monitor: str, mode: str = 'min') -> Optional[dict]:
    """Return the epoch row where `monitor` is best (min or max)."""
    valid = [r for r in rows if monitor in r and r[monitor] is not None]
    if not valid:
        return None
    return min(valid, key=lambda r: r[monitor]) if mode == 'min' \
        else max(valid, key=lambda r: r[monitor])


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _phase_table(rows: list[dict], columns: list[str]) -> str:
    """Build a markdown table from a subset of CSVLogger columns."""
    present = [c for c in columns if any(c in r for r in rows)]
    if not present:
        return "_No data available._\n"

    header = "| " + " | ".join(present) + " |"
    sep    = "|" + "|".join([":---:" for _ in present]) + "|"
    lines  = [header, sep]
    for r in rows:
        cells = []
        for c in present:
            v = r.get(c, "")
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    run_id:        str,
    log_root:      str,
    reg_stats:     dict,
    n_train:       int,
    n_val:         int,
    n_real_train:  int,
    n_synth_train: int,
    n_real_val:    int,
    n_synth_val:   int,
    img_size:      int,
    batch_size:    int,
    filters:       int,
    epochs_p1:     int,
    epochs_p2:     int,
    n_params:      int,
    pretrained:    str,
    pretrained_loaded: bool,
    start_time:    float,        # time.time() captured at script start
) -> str:
    """
    Generate a markdown training report and write it to
    `{log_root}/training_report_{run_id}.md`.

    Returns the absolute path of the written report.
    """
    import time
    elapsed   = time.time() - start_time
    generated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    phase1_csv = os.path.join(log_root, f'phase1_{run_id}.csv')
    phase2_csv = os.path.join(log_root, f'phase2_{run_id}.csv')
    p1_rows    = _read_csv_log(phase1_csv)
    p2_rows    = _read_csv_log(phase2_csv)

    # ── best metrics ─────────────────────────────────────────────────────────
    p1_best_reg = _best_row(p1_rows, 'val_regression_output_loss')
    p2_best_tot = _best_row(p2_rows, 'val_loss')
    p2_best_seg = _best_row(p2_rows, 'val_segmentation_output_loss')
    p2_best_reg = _best_row(p2_rows, 'val_regression_output_loss')

    def _epoch(row):
        if row is None:
            return 'N/A'
        e = row.get('epoch', '')
        return str(int(e) + 1) if isinstance(e, float) else str(e)

    def _val(row, key):
        if row is None or key not in row:
            return 'N/A'
        return f"{row[key]:.6f}"

    # ── checkpoint paths ──────────────────────────────────────────────────────
    ckpt_p1_reg = os.path.join(log_root, f'phase1_best_reg_{run_id}.weights.h5')
    ckpt_p2_reg = os.path.join(log_root, f'phase2_best_reg_{run_id}.weights.h5')
    ckpt_p2_seg = os.path.join(log_root, f'phase2_best_seg_{run_id}.weights.h5')
    ckpt_final  = os.path.join(log_root, f'final_best_{run_id}.weights.h5')

    pretrained_note = (
        f"`{pretrained}` (loaded successfully)" if pretrained_loaded
        else f"`{pretrained}` (**not found** — trained from scratch)"
    )

    # ── regression stats table ────────────────────────────────────────────────
    stat_rows = ""
    for i, label in enumerate(['logVolume', 'x', 'y', 'z']):
        stat_rows += (
            f"| {label} "
            f"| {reg_stats['min'][i]:.4f} "
            f"| {reg_stats['max'][i]:.4f} "
            f"| {reg_stats['mean'][i]:.4f} "
            f"| {reg_stats['std'][i]:.4f} |\n"
        )

    # ── phase tables ─────────────────────────────────────────────────────────
    p1_cols = ['epoch', 'regression_output_loss', 'val_regression_output_loss', 'lr']
    p2_cols = ['epoch', 'loss', 'val_loss',
               'segmentation_output_loss', 'val_segmentation_output_loss',
               'regression_output_loss',  'val_regression_output_loss', 'lr']

    p1_table = _phase_table(p1_rows, p1_cols)
    p2_table = _phase_table(p2_rows, p2_cols)

    p2_stopped = len(p2_rows)
    p2_note = (
        f"Completed {p2_stopped} of {epochs_p2} max epochs"
        + (f" — **early stopped** (best at epoch {_epoch(p2_best_tot)})"
           if p2_stopped < epochs_p2 else " — ran to full budget")
    )

    # ── assemble report ───────────────────────────────────────────────────────
    report = f"""# AlphaPolyp Training Report

**Run ID:** `{run_id}`
**Generated:** {generated}
**Total training time:** {_fmt_duration(elapsed)}

---

## 1. Configuration

| Parameter | Value |
|:----------|:------|
| Architecture | CAFormerS18 backbone + RAPUNet decoder + Regression head |
| Input size | {img_size} × {img_size} × 3 |
| Starting filters | {filters} |
| Total parameters | {n_params:,} |
| Batch size | {batch_size} |
| Phase 1 optimizer | AdamW  lr=1e-4  weight_decay=1e-6 |
| Phase 2 optimizer | AdamW  lr=1e-5  weight_decay=1e-6 |
| Phase 1 epochs | {epochs_p1} |
| Phase 2 max epochs | {epochs_p2} (EarlyStopping patience=16) |
| Random seed | 58800 |
| Pretrained checkpoint | {pretrained_note} |

---

## 2. Dataset

| Split | Real (cyclegan) | Synthetic | Total |
|:------|----------------:|----------:|------:|
| Train | {n_real_train} | {n_synth_train} | **{n_train}** |
| Val   | {n_real_val}   | {n_synth_val}   | **{n_val}**   |

### Regression Label Statistics (Training Set)

| Label | Min | Max | Mean | Std |
|:------|----:|----:|-----:|----:|
{stat_rows}
> Volume is stored as **log₁p(mm³)**; x / y / z are in mm.

---

## 3. Phase 1 — Regression Head Warm-up ({len(p1_rows)} epochs)

> Segmentation layers frozen. Only the Dense regression head updated.

**Best `val_regression_output_loss`:** {_val(p1_best_reg, 'val_regression_output_loss')} (epoch {_epoch(p1_best_reg)})

{p1_table}

---

## 4. Phase 2 — Full Fine-tune

> {p2_note}

| Metric | Best Value | Epoch |
|:-------|:----------:|:-----:|
| `val_loss` (combined) | {_val(p2_best_tot, 'val_loss')} | {_epoch(p2_best_tot)} |
| `val_segmentation_output_loss` | {_val(p2_best_seg, 'val_segmentation_output_loss')} | {_epoch(p2_best_seg)} |
| `val_regression_output_loss` | {_val(p2_best_reg, 'val_regression_output_loss')} | {_epoch(p2_best_reg)} |

{p2_table}

---

## 5. Saved Checkpoints

| File | Monitors | Best Value |
|:-----|:---------|:----------:|
| `{os.path.basename(ckpt_p1_reg)}` | `val_regression_output_loss` (Phase 1) | {_val(p1_best_reg, 'val_regression_output_loss')} |
| `{os.path.basename(ckpt_p2_reg)}` | `val_regression_output_loss` (Phase 2) | {_val(p2_best_reg, 'val_regression_output_loss')} |
| `{os.path.basename(ckpt_p2_seg)}` | `val_segmentation_output_loss` (Phase 2) | {_val(p2_best_seg, 'val_segmentation_output_loss')} |
| `{os.path.basename(ckpt_final)}` | Final best weights (EarlyStopping restored) | — |

> All files are in `{log_root}/`.
> Load with:
> ```python
> from model_architecture.model import create_model
> model = create_model(img_height=352, img_width=352, input_channels=3,
>                      out_classes=1, starting_filters={filters})
> model.load_weights("path/to/file.weights.h5")
> ```

---

## 6. TensorBoard

```bash
tensorboard --logdir {log_root}
# then open http://localhost:6006
```

---

## 7. Next Steps

```bash
# Evaluate on test set
python test.py \\
    --model_path {log_root}/final_best_{run_id}.weights.h5 \\
    --data_path  /path/to/data/test \\
    --csv        /path/to/data/test_labels.csv

# Single-image inference
python predict.py \\
    --image /path/to/image.jpg \\
    --model {log_root}/final_best_{run_id}.weights.h5 \\
    --stats regression_stats.pkl
```
"""

    report_path = os.path.join(log_root, f'training_report_{run_id}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return report_path
