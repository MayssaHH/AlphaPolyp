#!/usr/bin/env bash
# setup_runpod.sh  —  One-shot environment + data setup for AlphaPolyp on RunPod.
#
# Run this ONCE after the pod starts (or after attaching a fresh network volume).
# The Network Volume should be mounted at /workspace.
#
# Usage:
#   bash setup_runpod.sh
#
# What it does:
#   1. Installs Python dependencies
#   2. Installs & configures rclone for Google Drive
#   3. Mounts Google Drive at /workspace/gdrive
#   4. Symlinks the dataset into the expected layout under /workspace/data/
#   5. Runs a quick smoke-test (python sanity_check.py --root ...)

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE=/workspace
REPO_DIR=${WORKSPACE}/AlphaPolyp
DATA_DIR=${WORKSPACE}/data
GDRIVE_MOUNT=${WORKSPACE}/gdrive

# Google Drive folder name (inside the shared drive)
# The public link is:
# https://drive.google.com/drive/folders/1NARp2kYtM5-aD8gHWHypCNpK98rDFR7-
GDRIVE_REMOTE="gdrive"
GDRIVE_SUBFOLDER="AlphaPolyp_data/extracted_folder/syth-colon"

echo "============================================================"
echo "  AlphaPolyp RunPod Setup"
echo "============================================================"

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/5] Installing system packages..."
apt-get update -qq
apt-get install -y -qq fuse3 unzip curl wget git

# ── 2. Python dependencies ────────────────────────────────────────────────────
echo "[2/5] Installing Python dependencies..."
pip install --quiet --upgrade pip
# blinker 1.4 ships as a distutils package on some base images; force-upgrade it
# so Flask's blinker>=1.6.2 requirement resolves cleanly.
pip install --quiet --ignore-installed blinker
pip install --quiet \
    "numpy>=1.22,<1.24" \
    "opencv-python-headless>=4.5.0" \
    "tensorflow==2.13.0" \
    "tensorflow-addons==0.22.0" \
    "albumentations==1.3.0" \
    "scikit-learn==1.3.0" \
    "tqdm==4.66.2" \
    "Pillow==10.3.0" \
    "flask==2.3.3" \
    "werkzeug>=2.3.0" \
    "keras" \
    "keras-cv-attention-models==1.3.17" \
    "requests>=2.0.0" \
    "rclone-python"   # Python wrapper (optional; we use the CLI directly below)

# ── 3. Install rclone (CLI) ───────────────────────────────────────────────────
echo "[3/5] Installing rclone..."
if ! command -v rclone &> /dev/null; then
    curl -s https://rclone.org/install.sh | bash
fi
rclone version

# ── 4. Configure rclone for Google Drive ─────────────────────────────────────
echo "[4/5] Configuring rclone for Google Drive..."
echo ""
echo "  ┌─────────────────────────────────────────────────────────┐"
echo "  │  INTERACTIVE STEP — follow the prompts below            │"
echo "  │                                                          │"
echo "  │  When asked:                                             │"
echo "  │    name>             type: gdrive                        │"
echo "  │    Storage type:     choose Google Drive (number)        │"
echo "  │    client_id:        leave blank (press Enter)           │"
echo "  │    client_secret:    leave blank (press Enter)           │"
echo "  │    scope:            type: 1  (full access)              │"
echo "  │    Use auto config?  type: n                             │"
echo "  │                                                          │"
echo "  │  You will get a URL.  Open it on your LOCAL machine,     │"
echo "  │  log in with Google, copy the verification code and      │"
echo "  │  paste it back here.                                     │"
echo "  └─────────────────────────────────────────────────────────┘"
echo ""

if ! rclone listremotes | grep -q "^${GDRIVE_REMOTE}:"; then
    rclone config
else
    echo "  rclone remote '${GDRIVE_REMOTE}' already configured — skipping."
fi

# ── 5. Mount Google Drive + create data symlinks ──────────────────────────────
echo "[5/5] Mounting Google Drive and linking data..."

mkdir -p "${GDRIVE_MOUNT}"

# Mount in background (read-only, no-modtime for speed)
if ! mountpoint -q "${GDRIVE_MOUNT}"; then
    rclone mount "${GDRIVE_REMOTE}:" "${GDRIVE_MOUNT}" \
        --vfs-cache-mode full \
        --vfs-cache-max-size 20G \
        --vfs-read-chunk-size 32M \
        --transfers 8 \
        --dir-cache-time 48h \
        --read-only \
        --daemon
    echo "  Google Drive mounted at ${GDRIVE_MOUNT}"
    sleep 3   # give FUSE time to settle
else
    echo "  ${GDRIVE_MOUNT} already mounted."
fi

# Link the dataset subtree into /workspace/data/
SRC="${GDRIVE_MOUNT}/${GDRIVE_SUBFOLDER}"

if [ ! -d "${SRC}" ]; then
    echo "ERROR: Could not find data at ${SRC}"
    echo "  Check that GDRIVE_SUBFOLDER matches the folder structure on your Drive."
    echo "  Run: rclone ls ${GDRIVE_REMOTE}: | head -30"
    exit 1
fi

ln -sfn "${SRC}" "${DATA_DIR}"
echo "  Dataset linked: ${DATA_DIR} -> ${SRC}"

# ── Verify repo is present ────────────────────────────────────────────────────
if [ ! -d "${REPO_DIR}" ]; then
    echo ""
    echo "Cloning AlphaPolyp repository..."
    git clone https://github.com/LineIntegralx/AlphaPolyp.git "${REPO_DIR}"
fi

cd "${REPO_DIR}"

echo ""
echo "============================================================"
echo "  Setup complete. Running sanity check..."
echo "============================================================"
echo ""

python sanity_check.py --root "${DATA_DIR}"

echo ""
echo "============================================================"
echo "  Ready to train. Run:"
echo "    cd ${REPO_DIR}"
echo "    bash run_training.sh"
echo "============================================================"
