import torch
import copy
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

class Client:
    def __init__(self, client_id, dataset: TensorDataset, model_fn, config):
        self.client_id = client_id
        self.model = model_fn().to(config["device"])
        self.config = config
        self.train_loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True)
        self.last_loss = None
        self.last_acc = None
        self.last_f1 = None
        self.control = None  # for Scaffold

    def set_weights(self, weights):
        self.model.load_state_dict(copy.deepcopy(weights))

    def _compute_class_weights(self):
        all_labels_flat = [label for _, y in self.train_loader for label in y.numpy()]
        label_counter = Counter(all_labels_flat)
        total = sum(label_counter.values())
        class_weights = []
        for cls in [0, 1]:
            count = label_counter.get(cls, 0)
            weight = total / (count + 1e-6) if count > 0 else 0.0
            class_weights.append(weight)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.config["device"])
        return class_weights / class_weights.sum()

    def _apply_fedprox(self, loss, global_weights, mu):
        prox_reg = 0.0
        for name, param in self.model.named_parameters():
            prox_reg += ((param - global_weights[name].to(self.config["device"])) ** 2).sum()
        return loss + (mu / 2) * prox_reg

    def _apply_scaffold_gradients(self):
        for name, param in self.model.named_parameters():
            if name in self.control:
                param.grad += self.control[name]

    def _initialize_control(self):
        self.control = {k: torch.zeros_like(v).to(self.config["device"]) for k, v in self.model.state_dict().items()}

    def train(self, strategy, global_weights=None):
        self.model.train()
        device = self.config["device"]
        epochs = self.config["train"]["local_epochs"]
        mu = self.config["train"].get("fedprox_mu", 0.01)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["train"]["lr"],
            weight_decay=self.config["train"]["weight_decay"]
        )

        if strategy == "scaffold" and self.control is None:
            self._initialize_control()

        criterion = torch.nn.CrossEntropyLoss(weight=self._compute_class_weights())

        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        for _ in range(epochs):
            for x_batch, y_batch in self.train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = criterion(output, y_batch)

                if strategy == "fedprox" and global_weights is not None:
                    loss = self._apply_fedprox(loss, global_weights, mu)

                loss.backward()

                if strategy == "scaffold":
                    self._apply_scaffold_gradients()

                optimizer.step()

                total_loss += loss.item() * y_batch.size(0)
                preds = torch.argmax(output, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        self.last_loss = total_loss / total
        self.last_acc = correct / total
        self.last_f1 = f1_score(all_labels, all_preds, average="weighted")

        weights = copy.deepcopy(self.model.state_dict())

        if strategy == "scaffold":
            control_delta = {k: weights[k] - self.control[k] for k in weights}
            return weights, control_delta
        else:
            return weights, None
