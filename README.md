# WiltAnnotationTool

Video Annotation Tool powered by SAM 2 and CoTracker.

## Features
- **Semi-Automatic Annotation**: Uses SAM 2 for mask generation and CoTracker for tracking points over time.
- **Multi-GPU Support**: Automatically detects multiple GPUs (e.g., Dual RTX A6000) and splits models for efficiency (SAM 2 on GPU 0, CoTracker on GPU 1). Default to Single GPU (RTX 3090) otherwise.

## Setup

### Prerequisites
- Linux
- NVIDIA Driver (CUDA 12.1+ recommended)
- Conda (Anaconda/Miniconda)

### Automatic Setup
Run the setup script to create the environment, install dependencies, and download checkpoints.

```bash
bash setup_env.sh
```

This will:
1. Create a Conda environment `WiltAnnotation`.
2. Install PyTorch and dependencies.
3. Clone SAM 2 and CoTracker repositories.
4. Download the necessary SAM 2 checkpoint.

### Manual Activation
After setup, activate the environment:

```bash
conda activate WiltAnnotation
```

## Running the Application

### Start Backend
```bash
./start_backend.sh
```

### Start Frontend
```bash
./start_frontend.sh
```
