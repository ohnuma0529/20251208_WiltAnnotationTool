#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate WiltAnnotation
cd backend
# Run uvicorn from the parent directory of backend?
# No, we are in backend dir, but package structure is backend.main
# Better to run from root

cd ..
export PYTHONPATH=$PYTHONPATH:$(pwd)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
