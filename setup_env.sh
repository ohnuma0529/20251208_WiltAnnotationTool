#!/bin/bash
set -e

echo "Starting Environment Setup for WiltAnnotationTool..."

# Function to check for command existence
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 could not be found. Please install it."
        exit 1
    fi
}

check_command wget
check_command git

# 0. Check CUDA (Basic check)
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA Driver detected."
    nvidia-smi
else
    echo "Warning: nvidia-smi not found. Ensure NVIDIA drivers are installed for GPU support."
fi

# 1. Create/Update Conda Environment
echo "Creating/Updating conda environment 'WiltAnnotation'..."
source ~/anaconda3/etc/profile.d/conda.sh
if conda info --envs | grep -q "WiltAnnotation"; then
    echo "Environment 'WiltAnnotation' already exists."
    conda activate WiltAnnotation
else
    conda create -n WiltAnnotation python=3.10 -y
    conda activate WiltAnnotation
fi

# 2. Install PyTorch (CUDA 12.1)
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install Core Dependencies
echo "Installing Core Dependencies..."
# Ensure backend/requirements.txt or direct install usually works. 
# Using explicit list from previous file plus additions.
pip install fastapi uvicorn[standard] opencv-python supervision matplotlib scipy click
pip install notebook ipywidgets hydra-core>=1.1.0

# 4. Install SAM 2
echo "Installing SAM 2..."
pip install git+https://github.com/facebookresearch/sam2.git

# 5. Install CoTracker3
echo "Installing CoTracker3..."
pip install git+https://github.com/facebookresearch/co-tracker.git

# 6. Download Checkpoints
echo "Downloading Checkpoints..."
mkdir -p checkpoints

SAM2_CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
SAM2_CHECKPOINT_PATH="checkpoints/sam2_hiera_large.pt"

if [ ! -f "$SAM2_CHECKPOINT_PATH" ]; then
    echo "Downloading SAM 2 Large Checkpoint..."
    wget -O "$SAM2_CHECKPOINT_PATH" "$SAM2_CHECKPOINT_URL"
else
    echo "SAM 2 Checkpoint already exists."
fi

# CoTracker checkpoints are handled by torch.hub automatically in the code, 
# but we can pre-download if needed. 
# For now, relying on torch.hub as per code.

echo "Environment Setup Complete!"
echo "To activate: conda activate WiltAnnotation"
