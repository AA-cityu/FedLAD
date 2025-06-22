import numpy as np

class AdaptiveController:
    """
    Self-adaptation module:
    - Decides whether to stop training early (early stopping)
    - Allows adaptation of aggregation strategy based on training signals
    """

    def __init__(self, patience=5, delta=0.001, strategy_list=None):
        self.patience = patience  # Number of rounds to wait before stopping
        self.delta = delta  # Minimum improvement to continue training
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.current_strategy = "fedavg"
        self.strategy_list = strategy_list or ["fedavg", "scaffold", "fedadam"]
        self.strategy_index = 0
        self.history = []
        self.switch_count = 0
        self.early_stop_round = None

    def check_early_stopping(self, current_loss):
        """
        Check whether to stop training early based on validation loss.
        """
        if self.best_loss is None or current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"[Monitor] No significant improvement. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("[✓] Early stopping triggered.")
        return self.early_stop

    def adapt_strategy(self, val_losses):
        """
        If loss curve shows sustained increase or oscillation, adapt aggregation strategy.
        """
        if len(val_losses) < 6:
            return self.current_strategy  # Not enough dataset to adapt

        recent = val_losses[-6:]
        diffs = np.diff(recent)

        if all(d > 0 for d in diffs):  # sustained increase
            self.strategy_index = (self.strategy_index + 1) % len(self.strategy_list)
            self.current_strategy = self.strategy_list[self.strategy_index]
            self.switch_count += 1
            print(f"Loss increasing — switching aggregation strategy to: {self.current_strategy}")

        elif np.sum(np.sign(diffs)[:-1] != np.sign(diffs)[1:]) >= 4:  # oscillation
            self.strategy_index = (self.strategy_index + 1) % len(self.strategy_list)
            self.current_strategy = self.strategy_list[self.strategy_index]
            self.switch_count += 1
            print(f"Loss oscillation — switching aggregation strategy to: {self.current_strategy}")

        return self.current_strategy
