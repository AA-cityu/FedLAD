# Aggregation strategies for federated learning

import copy
import torch
import numpy as np
from collections import OrderedDict

class FedAvg:
    """
    Standard FedAvg: average model weights from all clients
    """
    def aggregate(self, client_weights):
        avg_weights = copy.deepcopy(client_weights[0])
        for key in avg_weights.keys():
            for i in range(1, len(client_weights)):
                avg_weights[key] += client_weights[i][key]
            avg_weights[key] = avg_weights[key] / len(client_weights)
        return avg_weights


class Scaffold:
    """
    SCAFFOLD algorithm using control variates to correct client drift.
    """
    def __init__(self):
        self.server_control = None
        self.client_controls = {}

    def initialize_controls(self, global_weights):
        self.server_control = {k: torch.zeros_like(v) for k, v in global_weights.items()}

    def aggregate(self, client_weights, client_controls):
        # Simple average as baseline for aggregation
        avg_weights = copy.deepcopy(client_weights[0])
        for key in avg_weights.keys():
            for i in range(1, len(client_weights)):
                avg_weights[key] += client_weights[i][key]
            avg_weights[key] = avg_weights[key] / len(client_weights)

        # Update server control
        avg_control = copy.deepcopy(client_controls[0])
        for key in avg_control.keys():
            for i in range(1, len(client_controls)):
                avg_control[key] += client_controls[i][key]
            avg_control[key] = avg_control[key] / len(client_controls)

        self.server_control = avg_control
        return avg_weights


class FedAdam:
    """
    Server-side Adam optimizer
    """
    def __init__(self, beta1=0.9, beta2=0.99, epsilon=1e-5, lr=1.0):
        self.m = None
        self.v = None
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr
        self.t = 0

    def aggregate(self, client_weights, global_weights):
        self.t += 1
        delta = copy.deepcopy(client_weights[0])
        for key in delta.keys():
            delta[key] -= global_weights[key]

        # Average delta from all clients
        for key in delta.keys():
            for i in range(1, len(client_weights)):
                delta[key] += (client_weights[i][key] - global_weights[key])
            delta[key] = delta[key] / len(client_weights)

        if self.m is None:
            self.m = {k: torch.zeros_like(v) for k, v in delta.items()}
            self.v = {k: torch.zeros_like(v) for k, v in delta.items()}

        new_weights = copy.deepcopy(global_weights)
        for k in delta.keys():
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * delta[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (delta[k] ** 2)

            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            new_weights[k] -= self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

        return new_weights


class FedProx:
    def __init__(self, mu=0.01):
        self.mu = mu  # regularization

    def aggregate(self, local_weights):
        global_weights = OrderedDict()
        for k in local_weights[0].keys():
            global_weights[k] = torch.stack([w[k] for w in local_weights], dim=0).mean(dim=0)
        return global_weights