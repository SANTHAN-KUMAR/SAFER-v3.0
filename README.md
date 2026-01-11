# SAFER v3.0

**Scalable Aerospace Failure Estimation Runtime**

A tri-partite architecture for aerospace turbofan engine prognostics combining deep learning, physics-informed monitoring, and runtime safety assurance.

## Overview

SAFER v3.0 implements a novel approach to Remaining Useful Life (RUL) prediction for turbofan engines:
```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SAFER v3.0 Architecture                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │    Mamba     │    │   LPV-SINDy  │    │   Baseline   │               │
│  │  Predictor   │    │   Physics    │    │  Predictor   │               │
│  │   (DAL E)    │    │   Monitor    │    │   (DAL C)    │               │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘               │
│         │                   │                   │                       │
│         └───────────────────┼───────────────────┘                       │
│                             ▼                                           │
│                   ┌─────────────────┐                                   │
│                   │ Simplex Decision│                                  │
│                   │     Module      │                                  │
│                   └────────┬────────┘                                   │
│                            ▼                                            │
│                   ┌─────────────────┐                                   │
│                   │  Conformal UQ   │                                   │
│                   │  Alert Manager  │                                   │
│                   └─────────────────┘                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

<img width="2816" height="1536" alt="workflow diagram" src="https://github.com/user-attachments/assets/83a17330-5295-4e44-8379-bf7acc837453" />


### Key Components

1. **Mamba RUL Predictor** (DAL E): State-space model for high-accuracy prediction
2. **LPV-SINDy Physics Monitor** (DAL C): Data-driven physics model for anomaly detection
3. **Baseline Predictor** (DAL C): Conservative LSTM/Transformer for safety fallback
4. **Simplex Decision Module**: Runtime arbitration with safety guarantees
5. **Conformal Prediction**: Distribution-free uncertainty quantification
6. **Shared Memory Fabric**: Lock-free inter-process communication

## Features

- **Pure PyTorch Implementation**: No CUDA kernels or Triton dependencies
- **ONNX Export**: Deploy models in production environments
- **Ensemble Support**: Multiple model training for uncertainty estimation
- **Conformal Prediction**: Calibrated prediction intervals with coverage guarantees
- **Physics-Informed Monitoring**: LPV-SINDy for interpretable anomaly detection
- **Safety Assurance**: Simplex architecture for runtime switching
- **Lock-Free Communication**: High-performance shared memory transport



## Mamba model training block :

<img width="2816" height="1536" alt="mamba model" src="https://github.com/user-attachments/assets/0f9cfc54-b00c-4060-b0e8-755e32a9d4e9" />


In the **SAFER v3.0** repository, training the Mamba model involves a **Selective State Space Model (SSM)** architecture designed for time-series Remaining Useful Life (RUL) prediction.

### 1. Training Summary
Training is performed using a supervised learning approach on the **C-MAPSS** dataset. The model learns to map a sequence of sensor readings (typically 14 prognostic sensors) to a scalar RUL value. 

*   **Parallelization:** Unlike standard RNNs, Mamba is trained efficiently using a **parallel associative scan**. This allows the model to process long sequences in $O(L)$ time while maintaining the recurrent state benefits.
*   **Loss Function:** The repository uses **Mean Squared Error (MSE)** loss:
    $$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$
    where $\hat{y}$ is the predicted RUL and $y$ is the ground truth.
*   **Optimization:** Training utilizes the **Adam optimizer** with features like **Early Stopping** (based on validation RMSE) and **Learning Rate Scheduling** (ReduceLROnPlateau).
*   **Selective Mechanism:** The "selection" happens during the forward pass where the model parameters ($\Delta, B, C$) are computed as functions of the input $x_t$, allowing the model to "focus" on or "forget" specific parts of the sequence.

---

### 2. Core Mamba Equations
The training and inference are governed by the following mathematical transformations (implemented in [`safer_v3/core/ssm_ops.py`](https://github.com/SANTHAN-KUMAR/SAFER-v3.0/blob/main/safer_v3/core/ssm_ops.py)):

#### A. Continuous-Time SSM
The underlying system is modeled as a linear differential equation:
$$h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t)$$
$$y(t) = \mathbf{C}h(t)$$

#### B. Discretization (Zero-Order Hold)
To process discrete time steps $\Delta$, the continuous parameters $(\mathbf{A, B})$ are discretized into $(\mathbf{\bar{A}, \bar{B}})$:
$$\mathbf{\bar{A}} = \exp(\Delta \mathbf{A})$$
$$\mathbf{\bar{B}} = (\Delta \mathbf{A})^{-1}(\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot (\Delta \mathbf{B})$$
*Note: For diagonal $\mathbf{A}$, this is simplified in code to $\mathbf{\bar{B}} \approx \Delta \mathbf{B}$.*

#### C. Selective Mechanism (Input-Dependent)
The core "Selective" part of Mamba makes $\Delta, \mathbf{B},$ and $\mathbf{C}$ functions of the input $x_t$:
$$\Delta_t = \text{Softplus}(\text{Linear}_\Delta(x_t))$$
$$\mathbf{B}_t = \text{Linear}_B(x_t)$$
$$\mathbf{C}_t = \text{Linear}_C(x_t)$$

#### D. Recurrent Computation
During inference (or during the scan in training), the state is updated as:
$$h_t = \mathbf{\bar{A}}_t h_{t-1} + \mathbf{\bar{B}}_t x_t$$
$$y_t = \mathbf{C}_t h_t$$

### 3. Architecture Block
As seen in [`safer_v3/core/mamba.py`](https://github.com/SANTHAN-KUMAR/SAFER-v3.0/blob/main/safer_v3/core/mamba.py), the training loop passes data through a `MambaBlock`:
1.  **Normalization:** `RMSNorm` is applied to the input: $y = \frac{x}{\text{RMS}(x)} \cdot \gamma$
2.  **SSM Operation:** The selective SSM processes the normalized sequence.
3.  **Residual Connection:** $x_{out} = x_{in} + \text{SSM}(\text{RMSNorm}(x_{in}))$



## Physics model workflow :

<img width="2816" height="1536" alt="physics model img" src="https://github.com/user-attachments/assets/47b89f93-bc37-417d-9d25-20187723b793" />

In **SAFER v3.0**, the physics model (called **LPV-SINDy**) is trained to "discover" the laws of math and physics that govern an engine. It doesn't just memorize patterns; it creates an actual mathematical formula for the engine's behavior.

Here is the simple summary of how it is trained and used:

### 1. How it is Trained (The Discovery Phase)

The goal of training is to find the simplest equation that describes how sensor values (like Pressure or Temperature) change over time.

*   **The "Math Menu" (Library):** The model is given a "menu" of possible mathematical terms, such as $x$ (linear), $x^2$ (curves), and $x \cdot y$ (interactions).
*   **The "Cleaning" (Integration):** Because sensor data is noisy, the model doesn't look at instant changes. It looks at the **total change** over a small window of time (integration). This "smooths out" the jitters.
*   **The "Pruning" (Sparse Regression):** The model tries to fit the data using all the terms in the menu. If a term (like $x^3$) isn't helpful, its coefficient is set to **zero** and it's thrown away. This leaves only the "true" physics.

**The Training Equation:**
$$\Delta x \approx \left( \int \Theta(x) dt \right) \cdot \xi$$
*   $\Delta x$: The actual change in the sensor.
*   $\int \Theta(x) dt$: The "smoothed" menu of math functions.
*   $\xi$ (Xi): The **Coefficients** the model is trying to learn (e.g., "how much does fuel affect heat?").

---

### 2. How it is Used (The Monitoring Phase)

Once the model has learned the coefficients ($\xi$), it uses them as a "Virtual Engine" to double-check the real engine.

1.  **Prediction:** The model looks at current sensors and uses its learned equation to calculate what the *next* sensor reading **should** be.
2.  **Comparison:** It calculates the **Residual** (the gap between the math and reality).
3.  **Alert:** If the gap is small, the engine is following the "laws of physics." If the gap is large, something is physically wrong (like a broken blade or a leaking pipe).

**The Residual Equation:**
$$\text{Residual} = | \text{Real Sensor Change} - \text{Predicted Math Change} |$$

---

### Simple Example: A Fuel Leak
*   **Learned Law:** The model knows that if you increase **Fuel Flow** by 1 unit, **Temperature** should rise by 10 degrees.
*   **The Situation:** A fuel line develops a leak. The sensor shows high **Fuel Flow**, but the **Temperature** stays low because the fuel is spilling out instead of burning.
*   **The Physics Check:** The model calculates: *"Math says temperature should be +10, but reality says +0."*
*   **The Action:** The **Residual** is 10. This exceeds the safety threshold, and the system triggers an **Alert**, even if a standard AI hasn't seen this specific leak before.

### Why this is better?
By training the model this way, SAFER v3.0 doesn't just say "this looks weird." It says **"this violates the law of thermal dynamics,"** making the engine much safer and easier for engineers to trust.


## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/safer-v3.git
cd safer-v3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install SAFER package
pip install -e .
```

## Quick Start

### Training Mamba Predictor

```bash
python -m safer_v3.scripts.train_mamba \
    --data_dir CMAPSSData \
    --dataset FD001 \
    --epochs 100 \
    --batch_size 64 \
    --export_onnx
```

### Training Baseline Models

```bash
# Train all baselines for comparison
python -m safer_v3.scripts.train_baselines \
    --data_dir CMAPSSData \
    --dataset FD001 \
    --model all

# Train specific baseline
python -m safer_v3.scripts.train_baselines \
    --model lstm \
    --dataset FD001
```

### Using SAFER in Code

```python
from safer_v3.core.mamba import MambaRULPredictor, MambaConfig
from safer_v3.physics.lpv_sindy import LPVSINDyMonitor
from safer_v3.decision.simplex import SimplexDecisionModule
from safer_v3.decision.conformal import SplitConformalPredictor

# Create Mamba predictor
config = MambaConfig(d_input=14, d_model=64, d_state=16, n_layers=4)
mamba = MambaRULPredictor(config)

# Load trained weights
mamba.load_state_dict(torch.load('outputs/mamba_model.pt'))

# Create physics monitor
physics = LPVSINDyMonitor(n_states=14, window_size=5)

# Create decision module
simplex = SimplexDecisionModule()

# Create conformal predictor (after calibration)
conformal = SplitConformalPredictor(coverage=0.9)
conformal.calibrate(y_cal, y_pred_cal)

# Inference
with torch.no_grad():
    mamba_rul = mamba(sensor_sequence)
    baseline_rul = baseline(sensor_sequence)
    physics_residual = physics.detect_anomaly(sensor_sequence)
    
    interval = conformal.predict(mamba_rul.item())
    
    result = simplex.decide(
        complex_rul=mamba_rul.item(),
        baseline_rul=baseline_rul.item(),
        rul_lower=interval.lower,
        rul_upper=interval.upper,
        physics_residual=physics_residual,
    )
    
print(f"RUL: {result.rul:.1f} [{result.rul_lower:.1f}, {result.rul_upper:.1f}]")
print(f"Mode: {result.state.name}")
```

## Project Structure

```
safer_v3/
├── __init__.py
├── utils/
│   ├── config.py          # Configuration dataclasses
│   ├── logging_config.py  # Logging setup
│   └── metrics.py         # RUL metrics and scoring
├── core/
│   ├── ssm_ops.py         # SSM operations (ZOH, parallel scan)
│   ├── mamba.py           # Mamba architecture
│   ├── baselines.py       # LSTM, Transformer, CNN-LSTM
│   └── trainer.py         # Training pipeline
├── physics/
│   ├── library.py         # Function libraries
│   ├── sparse_regression.py  # STLSQ, STRidge, SR3
│   └── lpv_sindy.py       # LPV-SINDy monitor
├── fabric/
│   ├── ring_buffer.py     # Lock-free SPSC buffer
│   ├── shm_transport.py   # Shared memory transport
│   └── process_manager.py # Multi-process management
├── decision/
│   ├── conformal.py       # Conformal prediction
│   ├── alerts.py          # Alert management
│   └── simplex.py         # Simplex decision module
├── simulation/
│   ├── engine_sim.py      # Engine degradation simulation
│   └── data_generator.py  # Synthetic data generation
└── scripts/
    ├── train_mamba.py     # Mamba training script
    └── train_baselines.py # Baseline training script
```

## Data

SAFER v3.0 is designed for the NASA C-MAPSS turbofan engine degradation dataset:

- **FD001**: Single operating condition, single fault mode (100 train, 100 test)
- **FD002**: Six operating conditions, single fault mode
- **FD003**: Single operating condition, two fault modes
- **FD004**: Six operating conditions, two fault modes

Download from: [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

### Sensor Selection

SAFER uses 14 prognostic sensors (indices 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21) from the 21 available sensors based on correlation with degradation.

## Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| RMSE | < 15 cycles | Root mean square error |
| Inference Latency | < 20 ms | End-to-end prediction time |
| Coverage | ≥ 90% | Conformal prediction coverage |
| NASA Score | Minimize | Asymmetric penalty function |

### NASA Scoring Function

```
Score = Σ exp(-d/a₁) - 1  for d < 0 (late prediction)
        Σ exp(d/a₂) - 1   for d ≥ 0 (early prediction)

where d = RUL_pred - RUL_true, a₁ = 13, a₂ = 10
```

## Design Assurance Levels (DAL)

Following DO-178C guidelines:

| Component | DAL | Rationale |
|-----------|-----|-----------|
| Mamba Predictor | E | High performance, monitored |
| Physics Monitor | C | Interpretable, mathematically grounded |
| Baseline Predictor | C | Simple, well-understood |
| Simplex Decision | C | Safety-critical arbiter |

## Configuration

### YAML Configuration

```yaml
# config.yaml
mamba:
  d_input: 14
  d_model: 64
  d_state: 16
  n_layers: 4
  expand: 2
  dropout: 0.1

training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001

physics:
  window_size: 5
  threshold: 0.1
  polynomial_degree: 2

decision:
  physics_threshold: 0.1
  divergence_threshold: 30.0
  uncertainty_threshold: 50.0
```

### Loading Configuration

```python
from safer_v3.utils.config import SAFERConfig

config = SAFERConfig.from_yaml('config.yaml')
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=safer_v3 --cov-report=html

# Run specific test
pytest tests/test_mamba.py -v
```

## License

[License information]

## Citation

If you use SAFER v3.0 in your research, please cite:

```bibtex
@software{safer_v3,
  title = {SAFER v3.0: Scalable Aerospace Failure Estimation Runtime},
  year = {2024},
  description = {Tri-partite architecture for turbofan engine prognostics}
}
```

## Acknowledgments

- NASA Prognostics Center of Excellence for C-MAPSS dataset
- Mamba architecture from "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- SINDy methodology from "Discovering governing equations from data"




