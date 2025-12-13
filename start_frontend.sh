#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate WiltAnnotation

# Check if node is installed, if not, try to install it or warn
if ! command -v npm &> /dev/null
then
    echo "npm could not be found. Please ensure nodejs is installed in the conda environment."
    echo "Run: conda install -c conda-forge nodejs=18 -y"
    exit 1
fi

cd frontend

if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

echo "Starting Frontend..."
npm run dev
