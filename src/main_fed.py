import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import argparse
import torch
import numpy as np
import random
import time
import csv

from torch.utils.data import TensorDataset, DataLoader
from client.client import Client
from federated.trainer import FederatedTrainer
from monitor.monitor import TrainingMonitor
from adaptation.controller import AdaptiveController
from evaluation.evaluator import Evaluator
from utils.data_utils import preprocess_and_split
from dataset.split import split_iid, split_noniid
from models import get_model
from utils.output_utils import save_full_report


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device(gpu_id):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device


def main(args):
    # 1. Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. device
    config['device'] = get_device(config['device'])

    # 3. random seed
    set_seed(config.get("seed", 42))

    # 4. model
    model_name = config["model"]["name"]
    model_params = config["model"].get(model_name, {})

    config['model_fn'] = lambda: get_model(model_name)(**model_params)

    # 5. Automatically preprocess the data and save it as a client-side file.
    # Return the total data for splitting into val/test/train.
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_and_split(config)

    # 6. Divide the training data among each client.
    num_clients = config["federated"]["num_clients"]
    if config["data"]["distribution"] == "iid":
        client_splits = split_iid(x_train.numpy(), y_train.numpy(), num_clients)
    else:
        client_splits = split_noniid(x_train.numpy(), y_train.numpy(), num_clients)

    # 7. Create client instances
    clients = []
    batch_size = config["train"]["batch_size"]
    model_fn = config['model_fn']

    for i, (c_x, c_y) in enumerate(client_splits):
        tensor_x = torch.tensor(c_x, dtype=torch.float32)
        tensor_y = torch.tensor(c_y, dtype=torch.long)
        client_dataset = TensorDataset(tensor_x, tensor_y)

        # 8. Pass in client_dataset and model_fn
        clients.append(Client(i, client_dataset, model_fn, config))

    # 9. Build the validation set
    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 10. Initialize the training component
    trainer = FederatedTrainer(config["model_fn"], clients, val_loader, config)
    monitor = TrainingMonitor()
    controller = AdaptiveController(config)

    # 11. FL training
    start_time = time.time()
    early_stop_round = None
    total_rounds = config["federated"]["total_rounds"]
    for r in range(total_rounds):
        print(f"\n Round {r + 1}/{total_rounds}")
        train_loss, val_loss, train_f1, val_f1 = trainer.train_one_round()
        monitor.log_round(r, train_loss, val_loss, train_f1, val_f1)

        if monitor.check_stability():
            new_strategy = controller.adapt_strategy(monitor.metrics["val_loss"])
            trainer.set_strategy(new_strategy)

        if monitor.check_early_stop(
            patience=config["federated"]["early_stop_rounds"],
            delta=config["federated"].get("early_stop_delta", 0.001)
        ):
            print(" Early stopping triggered.")
            early_stop_round = r + 1
            break
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)

    # 12. Save best model
    if trainer.best_model_state is not None:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(trainer.best_model_state,
                   f"results/checkpoints/{config['data']['name']}_{model_name}_best_model.pth")
        print(f" Best model saved to checkpoints with F1: {trainer.best_val_f1:.4f}")
    else:
        print("️ No model was saved. Did training finish early or fail to improve?")

    # 13. Save logs
    os.makedirs("results", exist_ok=True)
    monitor.generate_plots()
    monitor.save_to_csv(f"results/monitors/{config['data']['name']}_{model_name}_training_metrics.csv")

    # 14. Testing set evaluation
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    evaluator = Evaluator(trainer.global_model, test_loader, config["device"])
    y_true, y_pred, y_prob = evaluator.evaluate()

    summary = {
        "early_stop_round": early_stop_round or total_rounds,
        "strategy_switch_count": controller.switch_count,
        "wall_clock_time": elapsed_time
    }

    save_full_report(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        summary_info=summary,
        txt_path="results/performance/final_report.txt",
        csv_path="results/performance/final_metrics.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="manual_config.yaml", help="Path to config file.")
    args = parser.parse_args()
    main(args)
