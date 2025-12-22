# Running SAFER v3.0 with Kaggle Docker Image

## Prerequisites
- Docker installed on your Fedora PC
- NVIDIA Docker runtime installed (`nvidia-docker2`)
- Kaggle GPU Docker image available

## Quick Start

### 1. Transfer Project to Linux PC

On your Windows machine:
```bash
# Compress the project (excluding outputs/large files)
tar -czf safer_v3.tar.gz --exclude='outputs' --exclude='__pycache__' --exclude='*.pyc' .
```

Transfer to your Linux PC via:
- USB drive
- Network share (scp/rsync)
- Git push/pull

On your Linux PC:
```bash
cd /path/to/your/workspace
tar -xzf safer_v3.tar.gz
```

### 2. Run with Kaggle Docker Image

```bash
# Navigate to project directory
cd /path/to/SAFER_v3.0

# Option A: Run interactive container (recommended for first time)
docker run --gpus all -it --rm \
  -v "$(pwd):/workspace" \
  -w /workspace \
  gcr.io/kaggle-gpu-images/python:latest \
  bash

# Inside container, install SAFER
pip install -e .

# Run training
python -m safer_v3.scripts.train_mamba \
  --data_dir CMAPSSData \
  --dataset FD001 \
  --epochs 50 \
  --batch_size 32 \
  --device cuda
```

```bash
# Option B: Direct training command (one-liner)
docker run --gpus all -it --rm \
  -v "$(pwd):/workspace" \
  -w /workspace \
  gcr.io/kaggle-gpu-images/python:latest \
  bash -c "pip install -e . && python -m safer_v3.scripts.train_mamba --data_dir CMAPSSData --dataset FD001 --epochs 50 --batch_size 32"
```

### 3. Verify GPU Access

Inside the container:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Expected output:
```
CUDA available: True
CUDA version: 12.2
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

## Docker Image Alternatives

If you already have a different Kaggle image:

```bash
# List available images
docker images | grep kaggle

# Use your specific image
docker run --gpus all -it --rm \
  -v "$(pwd):/workspace" \
  -w /workspace \
  YOUR_KAGGLE_IMAGE:TAG \
  bash
```

## Training Commands

### Quick Test (5 epochs, ~2-3 minutes on RTX 4060)
```bash
python -m safer_v3.scripts.train_mamba \
  --data_dir CMAPSSData \
  --dataset FD001 \
  --epochs 5 \
  --batch_size 32 \
  --device cuda
```

### Full Training (50 epochs, ~20-30 minutes on RTX 4060)
```bash
python -m safer_v3.scripts.train_mamba \
  --data_dir CMAPSSData \
  --dataset FD001 \
  --epochs 50 \
  --batch_size 64 \
  --lr 0.001 \
  --device cuda
```

### Train All Models (Mamba + Baselines)
```bash
# Train Mamba
python -m safer_v3.scripts.train_mamba \
  --data_dir CMAPSSData \
  --dataset FD001 \
  --epochs 50 \
  --batch_size 64 \
  --device cuda

# Train baselines for comparison
python -m safer_v3.scripts.train_baselines \
  --data_dir CMAPSSData \
  --dataset FD001 \
  --model all \
  --epochs 50 \
  --device cuda
```

## Advanced: Docker Compose (Optional)

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  safer-training:
    image: gcr.io/kaggle-gpu-images/python:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./:/workspace
    working_dir: /workspace
    command: bash -c "pip install -e . && python -m safer_v3.scripts.train_mamba --data_dir CMAPSSData --dataset FD001 --epochs 50 --batch_size 32"
```

Run:
```bash
docker-compose up
```

## Monitoring Training

### Option 1: Watch outputs directory
```bash
# In another terminal on Linux PC
watch -n 5 ls -lh outputs/
```

### Option 2: Tail log files
```bash
tail -f outputs/mamba_fd001_*/train.log
```

### Option 3: Check GPU utilization
```bash
# In another terminal
watch -n 1 nvidia-smi
```

## Troubleshooting

### GPU not detected
```bash
# Test NVIDIA Docker
docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# If fails, install nvidia-container-toolkit:
sudo dnf install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Permission issues
```bash
# Run with user permissions
docker run --gpus all -it --rm \
  -v "$(pwd):/workspace" \
  -w /workspace \
  -u $(id -u):$(id -g) \
  gcr.io/kaggle-gpu-images/python:latest \
  bash
```

### Out of memory
Reduce batch size:
```bash
python -m safer_v3.scripts.train_mamba \
  --batch_size 16  # or 8 for very large models
```

## Post-Training

Results will be saved to:
- `outputs/mamba_fd001_<timestamp>/`
  - `best_model.pt` - Best checkpoint
  - `train.log` - Training logs
  - `args.json` - Training arguments
  - `metrics.json` - Final metrics

Transfer back to Windows:
```bash
# On Linux PC
tar -czf safer_outputs.tar.gz outputs/

# Copy to Windows or access via network share
```

## Performance Expectations (RTX 4060 Laptop GPU)

| Dataset | Epochs | Batch Size | Training Time | Expected RMSE |
|---------|--------|------------|---------------|---------------|
| FD001   | 5      | 32         | ~2-3 min      | ~20-25        |
| FD001   | 50     | 32         | ~20-30 min    | ~12-15        |
| FD001   | 100    | 64         | ~40-50 min    | ~11-13        |

## Alternative: Jupyter Notebook in Container

```bash
docker run --gpus all -it --rm \
  -v "$(pwd):/workspace" \
  -w /workspace \
  -p 8888:8888 \
  gcr.io/kaggle-gpu-images/python:latest \
  bash -c "pip install -e . && jupyter notebook --ip=0.0.0.0 --allow-root"
```

Then access: `http://localhost:8888` (copy token from terminal)

---

**Quick Reference**: The fastest way is Option B above - one command to install and train!
