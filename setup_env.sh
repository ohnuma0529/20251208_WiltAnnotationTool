#!/bin/bash
set -e

# Create Conda Environment
echo "Creating conda environment 'WiltAnnotation'..."
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n WiltAnnotation python=3.10 -y
conda activate WiltAnnotation

# Install PyTorch (CUDA 12.1)
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install Core Dependencies
echo "Installing Core Dependencies..."
pip install fastapi uvicorn[standard] opencv-python supervision matplotlib scipy click
pip install notebook ipywidgets

# Install SAM 2
echo "Installing SAM 2..."
# Cloning SAM 2 repository and installing (assuming standard install)
pip install git+https://github.com/facebookresearch/sam2.git

# Install CoTracker3
echo "Installing CoTracker3..."
pip install git+https://github.com/facebookresearch/co-tracker.git
# Note: CoTracker might require additional dependencies or specific hydra-core versions, 
# ensuring basic hydra is installed
pip install hydra-core>=1.1.0

echo "Environment Setup Complete!"
