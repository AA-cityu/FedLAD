import os
import argparse
import yaml
import torch
import numpy as np
from typing import Dict
from collections import Counter

from dataset.loader import parse_hdfs, parse_bgl_thunder
from dataset.parser_registry import get_parser


def analyze_dataset(dataset: str, device: str, log_path: str = None, label_path: str = None) -> Dict:
    print(f"[INFO] Using device: {device}")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    parser_fn = get_parser(dataset)

    # If the user does not provide a path, then use the default path.
    if log_path is None:
        if dataset.lower() == "hdfs":
            log_path = 'dataset/HDFS/HDFS10k.log'
            label_path = 'dataset/HDFS/HDFS10k_label.csv'
        elif dataset.lower() == "bgl":
            log_path = 'dataset/BGL/BGL10k.log'
        elif dataset.lower() == 'thunderbird':
            log_path = 'dataset/Thunderbird/Thunderbird10k.log'

    # Uniformly call the parser_fn, depending on whether label_path processing is required.
    if dataset.lower() == "hdfs":
        x, y = parser_fn(log_path, device, label_path)
    else:
        x, y = parser_fn(log_path, device)

    y = np.array(y)
    total_samples = len(y)
    class_counter = Counter(y.tolist())

    return {
        "total_samples": total_samples,
        "class_distribution": dict(class_counter),
        "anomaly_ratio": class_counter.get(1, 0) / total_samples
    }


def default_model_params(model: str) -> Dict:
    """
    Provide the default structural parameters of the model.
    """
    if model.lower() == "deeplog":
        return {"input_size": 768, "hidden_size": 64, "dropout": 0.1}
    elif model.lower() == "neurallog":
        return {"embed_dim": 256, "ff_dim": 512, "num_heads": 4, "max_len": 64, "dropout": 0.1}
    elif model.lower() == "loganomaly":
        return {"input_size": 384, "hidden_size": 64}
    else:
        raise ValueError(f"Unsupported model: {model}")


def generate_config(dataset: str,
                    model: str,
                    raw_device: str,
                    stats: Dict,
                    overrides: Dict) -> Dict:
    anomaly_ratio = stats["anomaly_ratio"]
    total_samples = stats["total_samples"]

    client_num = overrides.get("clients", 100 if total_samples >= 50000 else 50)
    rounds = overrides.get("rounds", 50 if anomaly_ratio < 0.05 else 30)
    batch_size = overrides.get("batch_size", 64 if anomaly_ratio > 0.2 else 32)
    iid = overrides.get("iid", not (anomaly_ratio > 0.6 or anomaly_ratio < 0.05))
    print(f'raw_device: {raw_device}')
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    device = 0 if str(raw_device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    print(f'device: {device}')

    config = {
        "device": device,
        "data": {
            "name": dataset,
            "distribution": "iid" if iid else "noniid",
            "val_ratio": 0.1,
            "test_ratio": 0.2
        },
        "model": {
            "name": model,
            model.lower(): default_model_params(model)
        },
        "train": {
            "batch_size": batch_size,
            "local_epochs": 5,
            "lr": 0.001,
            "weight_decay": 1e-6
        },
        "federated": {
            "num_clients": client_num,
            "total_rounds": rounds,
            "aggregation": "fedavg",
            "early_stop_rounds": 20,
            "early_stop_delta": 0.0001,
            "fedprox": {
                "mu": 0.01
            }
        },
        "output": {
            "dir": "results/"
        }
    }
    return config


def save_config(config: Dict, output_path: str):
    with open(output_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Config saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="config.yaml")
    parser.add_argument("--log_path", type=str, default=None, help="Path to the log file (for custom datasets)")
    parser.add_argument("--label_path", type=str, default=None, help="Path to the label file (if required)")

    parser.add_argument("--clients", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--iid", type=lambda x: x.lower() == "true")

    args = parser.parse_args()

    stats = analyze_dataset(args.dataset, args.device, args.log_path, args.label_path)

    overrides = {}
    if args.clients is not None:
        overrides["clients"] = args.clients
    if args.rounds is not None:
        overrides["rounds"] = args.rounds
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.iid is not None:
        overrides["iid"] = args.iid

    config = generate_config(args.dataset, args.model, args.device, stats, overrides)
    save_config(config, args.output)


if __name__ == "__main__":
    main()
