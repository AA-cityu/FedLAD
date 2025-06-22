# FedLAD: A Modular and Adaptive Testbed for Federated Log Anomaly Detection

**FedLAD** is a modular, extensible, and self-adaptive **testbed** for benchmarking log anomaly detection (LAD) models under federated learning (FL) settings. It simulates realistic deployment environments with heterogeneous clients, data privacy constraints, and adaptive training behaviors. Researchers can easily plug in custom models, datasets, and aggregation strategies to evaluate performance, scalability, and robustness.

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

## Quick Start

### Option 1: Auto-generate configuration

Use `auto_config.py` to generate a config file based on your chosen dataset and model. This tool will recommend reasonable defaults and reduce setup complexity.

```bash
# Example: generate config for BGL + DeepLog
python auto_config.py --dataset BGL --model deeplog

# Then run training with the generated config
python src/main_fed.py --config config.py


### Option 2: Auto-generate configuration

If you prefer not to generate configs manually, you can use our prepared YAML files in the config/ folder:

```bash
python src/main_fed.py --config manual_config.py




