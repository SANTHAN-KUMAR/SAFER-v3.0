#!/bin/bash
# One-command training in your Kaggle Docker container

docker run --gpus all -it --rm \
  -p 8888:8888 \
  -v $(pwd):/kaggle/working \
  kaggle-image \
  bash -c "cd /kaggle/working && pip install -e . && python -m safer_v3.scripts.train_mamba --data_dir CMAPSSData --dataset FD001 --epochs 50 --batch_size 64 --device cuda"
