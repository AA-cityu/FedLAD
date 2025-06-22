# checkpoints/__init__.py

from .deeplog import DeepLog
from .loganomaly import LogAnomaly
from .neurallog import NeuralLog
from .onelog import OneLog

def get_model(name):
    name = name.lower()
    if name == "deeplog":
        return DeepLog
    elif name == "loganomaly":
        return LogAnomaly
    elif name == "neurallog":
        return NeuralLog
    else:
        raise ValueError(f"Unsupported model name: {name}")
