#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate WiltAnnotation
cd backend
# Run uvicorn from the parent directory of backend?
# No, we are in backend dir, but package structure is backend.main
# Better to run from root

cd ..
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Pre-flight Check
python3 scripts/verify_deployment.py
if [ $? -ne 0 ]; then
    echo "Deployment verification failed. Aborting startup."
    exit 1
fi

nohup env PYTHONUNBUFFERED=1 uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload >> backend.log 2>&1 &
echo "Backend started in background. Logs in backend.log"
