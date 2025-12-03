#!/bin/bash
# Run SAFER v3.0 in Kaggle Docker Container with GPU Support

# Navigate to the working directory inside container
docker run --gpus all -it --rm \
  -p 8888:8888 \
  -v $(pwd):/kaggle/working \
  kaggle-image \
  bash -c "cd /kaggle/working && bash"

# Alternative: Direct training without interactive shell
# docker run --gpus all -it --rm \
#   -p 8888:8888 \
#   -v $(pwd):/kaggle/working \
#   kaggle-image \
#   bash -c "cd /kaggle/working && pip install -e . && python -m safer_v3.scripts.train_mamba --data_dir CMAPSSData --dataset FD001 --epochs 50 --batch_size 64"
