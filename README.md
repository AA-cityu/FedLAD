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

## Environment Setup

We recommend using the following environment for best compatibility and performance:

  - Python version: **3.9**
  - PyTorch version: **1.13.1+cu117** (CUDA 11.7)

You can create a clean virtual environment and install the required packages as follows:

```bash
# Step 1: Create and activate a virtual environment (using conda)
conda create -n fedlad python=3.9 -y
conda activate fedlad

# Step 2: Install PyTorch 1.13.0 with CUDA 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

# Step 3: Install other dependencies
pip install -r requirements.txt
```

## Quick Start

### Option 1: Auto-generate configuration

Use `auto_config.py` to generate a config file based on your chosen dataset and model. This tool will recommend reasonable defaults and reduce setup complexity.

```bash
# Example: generate config for BGL + DeepLog
python auto_config.py --dataset bgl --model deeplog

# Then run training with the generated config
python src/main_fed.py --config config.yaml
```

### Option 2: Auto-generate configuration

If you prefer not to generate configs, you can use our prepared YAML files:

```bash
python src/main_fed.py --config manual_config.yaml
