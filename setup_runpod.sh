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

# ── 5. Copy dataset from Google Drive to network volume ───────────────────────
echo "[5/5] Copying dataset from Google Drive to ${DATA_DIR}..."
echo "  (FUSE mount is not used — data is copied directly for reliability)"

mkdir -p "${DATA_DIR}"

# Skip if the key subdirectories already exist (re-run safe)
if [ -d "${DATA_DIR}/images" ] && [ -d "${DATA_DIR}/masks" ]; then
    echo "  Dataset already present at ${DATA_DIR} — skipping copy."
else
    rclone copy "${GDRIVE_REMOTE}:${GDRIVE_SUBFOLDER}" "${DATA_DIR}" \
        --progress \
        --transfers 8 \
        --checkers 16 \
        --drive-chunk-size 64M
    echo "  Dataset copied to ${DATA_DIR}"
fi

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
