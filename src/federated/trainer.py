import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import copy
import numpy as np
from src.monitor.monitor import TrainingMonitor
from src.adaptation.controller import AdaptiveController
from strategies import FedAvg, Scaffold, FedAdam, FedProx
from sklearn.metrics import f1_score
from collections import Counter
from sklearn.metrics import classification_report


class FederatedTrainer:
    def __init__(self, model_fn, clients, val_loader, config):
        self.model_fn = model_fn
        self.clients = clients  # list of Client objects
        self.val_loader = val_loader
        self.config = config
        self.strategy_name = config["federated"]["aggregation"]
        self.strategy_map = {
            "fedavg": FedAvg(),
            "scaffold": Scaffold(),
            "fedadam": FedAdam(),
            "fedprox": FedProx()
        }
        self.monitor = TrainingMonitor()
        self.controller = AdaptiveController()
        self.global_model = self.model_fn().to(config["device"])
        self.global_weights = self.global_model.state_dict()
        self.best_val_f1 = -1.0
        self.best_model_state = None

        if self.strategy_name == "scaffold":
            self.strategy_map["scaffold"].initialize_controls(self.global_weights)

    def set_strategy(self, new_strategy):
        if new_strategy != self.strategy_name:
            print(f"[Controller] Strategy changed: {self.strategy_name} → {new_strategy}")
            self.strategy_name = new_strategy
            if new_strategy == "scaffold":
                self.strategy_map["scaffold"].initialize_controls(self.global_weights)

    def validate(self):
        self.global_model.eval()
        correct, total, loss_sum = 0, 0, 0
        y_true, y_pred = [], []
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.config["device"]), y.to(self.config["device"])
                logits = self.global_model(x)
                loss = criterion(logits, y)

                loss_sum += loss.item() * y.size(0)
                preds = torch.argmax(logits, dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        if total == 0 or len(y_true) == 0:
            print("[WARNING] Validation set is empty or corrupted.")
            return 0.0, 0.0, 0.0

        print(" True label distribution:", Counter(y_true))
        print(" Predicted label distribution:", Counter(y_pred))

        acc = correct / total
        f1 = f1_score(y_true, y_pred, average="weighted")  # 或 macro / micro
        return loss_sum / total, acc, f1

    def train_one_round(self):
        """
        Single-round training:
        Aggregation + Update of global model + Evaluation on validation set + Summarization of client results
        return train_loss, val_loss, train_f1, val_f1
        """
        print(f"\n[Trainer] Round using strategy: {self.strategy_name}")
        local_weights, local_controls = [], []

        for client in self.clients:
            client.set_weights(copy.deepcopy(self.global_weights))
            # Pass "global_weights" to the fedprox strategy
            if self.strategy_name == "fedprox":
                w, c = client.train(strategy=self.strategy_name, global_weights=self.global_weights)
            else:
                w, c = client.train(strategy=self.strategy_name)

            local_weights.append(w)
            if c is not None:
                local_controls.append(c)

        # Aggregation
        aggregator = self.strategy_map[self.strategy_name]
        if self.strategy_name == "fedavg":
            self.global_weights = aggregator.aggregate(local_weights)
        elif self.strategy_name == "scaffold":
            self.global_weights = aggregator.aggregate(local_weights, local_controls)
        elif self.strategy_name == "fedadam":
            self.global_weights = aggregator.aggregate(local_weights, self.global_weights)

        self.global_model.load_state_dict(self.global_weights)

        # Average training results of the client side
        train_loss = np.mean([c.last_loss for c in self.clients])
        train_acc = np.mean([c.last_acc for c in self.clients])
        train_f1 = np.mean([c.last_f1 for c in self.clients])

        val_loss, val_acc, val_f1 = self.validate()

        print(f"[Round Summary] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > self.best_val_f1:
            print(f"[Checkpoint] New best model found! Val F1: {val_f1:.4f} (prev: {self.best_val_f1:.4f})")
            self.best_val_f1 = val_f1
            self.best_model_state = copy.deepcopy(self.global_model.state_dict())

        return train_loss, val_loss, train_f1, val_f1
