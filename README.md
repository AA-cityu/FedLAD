# FedLAD: A Modular and Adaptive Testbed for Federated Log Anomaly Detection

**FedLAD** is a modular, extensible, and self-adaptive testbed for benchmarking log anomaly detection (LAD) models under federated learning (FL) settings. It simulates realistic deployment environments with simulated clients, data privacy constraints, and adaptive training behaviors. Researchers can easily plug in custom models, datasets, and aggregation strategies to evaluate performance, scalability, and robustness.

## Key Features

- **Modular Design**: Plug-and-play LAD models, datasets, and FL algorithms.
- **Self-Adaptive Training**: Built-in controller supports early stopping and dynamic aggregation strategy switching.
- **Benchmarking Ready**: Integrated logging, metric tracking, and visualization.
- **Realistic Simulation**: Supports both IID and non-IID data splits across multiple virtual clients.

## Supported Models

FedLAD supports commonly used LAD models such as:
- DeepLog (LSTM-based)
- NeuralLog (Transformer-based)
- LogAnomaly (Hybrid)

Users can also easily register and integrate **any custom LAD model** by following the standard interface.

## Supported Datasets

- BGL
- HDFS
- Thunderbird

Users can also register and preprocess **custom datasets** using the parser registry framework.

## System Requirements

FedLAD has been tested on a workstation with 4× NVIDIA GeForce RTX 3090 (24GB each) using CUDA 11.4. The following environment settings are recommended:

- **Operating System**:
  - Ubuntu 20.04 or later (Linux recommended)
  - MacOS and Windows are not officially supported

- **Python Version**:  
  - Python 3.9 or 3.10 (both tested)

- **Hardware Requirements**:
  - CPU: ≥ 4 cores (8 recommended for simulating multiple clients)
  - RAM: ≥ 8 GB (≥ 16 GB recommended for large models)
  - GPU: NVIDIA GPU with CUDA ≥ 11.0 and ≥ 8 GB VRAM

- **Disk Space**:  
  - ≥ 5 GB (for datasets, checkpoints, logs)


## Environment Setup

We recommend using Conda to manage environments for the best compatibility.

```bash
# Step 1: Create and activate a virtual environment (using conda)
conda create -n fedlad python=3.9 -y
conda activate fedlad

# Step 2: Install PyTorch 1.13.0 with CUDA 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

# Step 3: Install other dependencies
cd FedLAD
pip install -r requirement.txt
```

## Quick Start

### Option 1: Auto-generate configuration

Use `auto_config.py` to generate a config file based on your chosen dataset and model. This tool will recommend reasonable defaults and reduce setup complexity.

```bash
# Example: generate config for BGL + LogAnomaly
python auto_config.py --dataset bgl --model loganomaly

# Then run training with the generated config
python src/main_fed.py --config config.yaml
```

### Option 2: Auto-generate configuration

If you prefer not to generate configs, you can use our prepared YAML files:

```bash
python src/main_fed.py --config manual_config.yaml
```

## Extension Guide

This section describes how to extend FedLAD with custom models, datasets, and strategies.

### 1. Add a New LAD Model
1. Implement your model class in `models/your_model.py`
2. Ensure it inherits from `nn.Module` and implements `forward()`
3. Register the model in `model_registry.py`:
   
```python
from models.your_model import YourModel
model_registry["your_model"] = YourModel
```

### 2. Add a New Dataset
1. Register a parser by adding your custom parser to `parser_registry.py` and register it like so:

```python
from parser_registry import register_parser

@register_parser("your_dataset_name")
def parse_your_dataset():
    ...
    return log_vectors, labels
```

2. Once registered, your parser can be retrieved and used by name:

```python
from parser_registry import get_parser

parser = get_parser("your_dataset_name")
```

3. Use Your Dataset
Once registered, the dataset can be used by name in your configuration file:

```python
dataset:
  name: your_dataset_name
```
