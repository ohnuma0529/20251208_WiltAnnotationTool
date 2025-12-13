#!/bin/bash
# Preload all images to cache

# Ensure we are in the project root
echo "Starting Cache Preload..."
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the python script
/home/happyai2023/anaconda3/envs/WiltAnnotation/bin/python backend/preload_cache.py

echo "Done."
