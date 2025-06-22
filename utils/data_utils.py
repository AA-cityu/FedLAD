import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from dataset.loader import parse_bgl_thunder, parse_hdfs
from collections import Counter

def preprocess_and_split(config):
    # Automatic completion of paths and parameter settings (based on dataset name)
    name = config["data"]["name"].lower()
    base_path = f"./dataset"

    if name == "bgl":
        config["data"]["raw_log_path"] = f"{base_path}/BGL/BGL10k.log"
        config["data"]["encoded_file"] = f"{base_path}/BGL/total.pkl"
        config["data"]["window_size"] = 1
        config["data"]["step_size"] = 1

    elif name == "hdfs":
        config["data"]["raw_log_path"] = f"{base_path}/HDFS/HDFS10k.log"
        config["data"]["label_path"] = f"{base_path}/HDFS/HDFS10k_label.csv"
        config["data"]["encoded_file"] = f"{base_path}/HDFS/total.pkl"
        config["data"]["window_size"] = 1
        config["data"]["step_size"] = 1

    elif name == "thunderbird":
        config["data"]["raw_log_path"] = f"{base_path}/Thunderbird/Thunderbird10k.log"
        config["data"]["encoded_file"] = f"{base_path}/Thunderbird/total.pkl"
        config["data"]["window_size"] = 1
        config["data"]["step_size"] = 1

    else:
        raise ValueError(f"[ERROR] Unsupported dataset name: {name}")

    # load configuration parameters
    dataset_name = config["data"]["name"]
    raw_log_path = config["data"]["raw_log_path"]
    label_path = config["data"].get("label_path", None)
    encoded_path = config["data"]["encoded_file"]
    device = config["device"]
    window_size = config["data"].get("window_size", 1)
    step_size = config["data"].get("step_size", 1)
    val_ratio = config["data"]["val_ratio"]
    test_ratio = config["data"]["test_ratio"]

    if os.path.exists(encoded_path):
        print(f"[INFO] Load cache data: {encoded_path}")
        with open(encoded_path, "rb") as f:
            x, y = pickle.load(f)
    else:
        print(f"[INFO] The original log file is being parsed: {raw_log_path}")
        if dataset_name in ["bgl", "thunderbird"]:
            x, y = parse_bgl_thunder(raw_log_path, device, window_size, step_size)
        elif dataset_name == "hdfs":
            if not label_path:
                raise ValueError("In the HDFS mode, the label_path must be provided.")
            x, y = parse_hdfs(raw_log_path, label_path, device)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        os.makedirs(os.path.dirname(encoded_path), exist_ok=True)
        with open(encoded_path, "wb") as f:
            pickle.dump((x, y), f)
        print(f"[INFO] The preprocessed data has been saved to: {encoded_path}")

    # Hierarchical sampling division: train/val/test
    x, y = shuffle(x, y, random_state=2025)
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=float(test_ratio), stratify=y, random_state=2025)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=float(val_ratio) / (1 - float(test_ratio)),
        stratify=y_temp, random_state=2025)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Print data distribution
    print("[INFO] Label distributions:")
    print("Train label dist:", Counter(y_train.numpy()))
    print("Val label dist:  ", Counter(y_val.numpy()))
    print("Test label dist: ", Counter(y_test.numpy()))

    return x_train, y_train, x_val, y_val, x_test, y_test
