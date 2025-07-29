import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator

class TrainingMonitor:
    """
    Monitor for federated training metrics across rounds.
    Automatically record and save the training/verification loss, F1 value,
    detect convergence/oscillation, and be able to export in CSV format.
    """

    def __init__(self, log_dir="results/logs", plot_dir="results/figures"):
        self.log_dir = log_dir
        self.plot_dir = plot_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        self.metrics = {
            "round": [],
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": []
        }

    def log_round(self, round_num, train_loss, val_loss, train_f1, val_f1):
        """
        Call after each training round: Record the corresponding train/val loss and F1.
        """
        self.metrics["round"].append(round_num)
        self.metrics["train_loss"].append(train_loss)
        self.metrics["val_loss"].append(val_loss)
        self.metrics["train_f1"].append(train_f1)
        self.metrics["val_f1"].append(val_f1)
        self._save_json()

    def _save_json(self):
        """
        Save the metrics as JSON format, which can then be loaded or visualized at any time.
        """
        path = os.path.join(self.log_dir, "training_metrics.json")
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def generate_plots(self):
        """
        Based on the recorded metrics, draw the Loss curve and F1 curve, and save them as PNG.
        """
        df = pd.DataFrame(self.metrics)

        # Loss
        fig1 = plt.figure()
        plt.plot(df["round"], df["train_loss"], label="Train Loss")
        plt.plot(df["round"], df["val_loss"], label="Val Loss")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.title("Loss Over Rounds")
        plt.legend()
        loss_path = os.path.join(self.plot_dir, "loss_curve.png")
        plt.savefig(loss_path)
        plt.close(fig1)

        # F1
        fig2 = plt.figure()
        plt.plot(df["round"], df["train_f1"], label="Train F1")
        plt.plot(df["round"], df["val_f1"], label="Val F1")
        plt.xlabel("Round")
        plt.ylabel("F1 Score")
        plt.title("F1 Score Over Rounds")
        plt.legend()
        f1_path = os.path.join(self.plot_dir, "f1_curve.png")
        plt.savefig(f1_path)
        plt.close(fig2)

        print(f"Plots saved to {self.plot_dir}")

    def save_to_csv(self, csv_path):
        df = pd.DataFrame(self.metrics)
        df.to_csv(csv_path, index=False)
        print(f"Metrics saved to CSV: {csv_path}")

    def detect_convergence_or_instability(self, window=5, threshold=0.01):
        """
        Analyze the val_loss of the recent window to determine whether the training has converged or is oscillating.
        """
        val_loss = self.metrics["val_loss"]
        status = "Training ongoing..."

        if len(val_loss) >= window + 1:
            recent_losses = val_loss[-(window + 1):]
            diffs = np.diff(recent_losses)
            mean_change = np.mean(np.abs(diffs))

            # convergence
            if mean_change < threshold:
                status = f"Converged: avg loss change {mean_change:.4f} < {threshold}"
            # oscillation
            elif np.sum(np.sign(diffs)[:-1] != np.sign(diffs)[1:]) >= window - 1:
                status = f"Oscillation detected in last {window} rounds"
            else:
                status = f"Stable: avg loss change = {mean_change:.4f}"

        print(f"[Monitor] {status}")
        return status

    def check_early_stop(self, patience=5, delta=0.001):
        """
        Detect early stopping:
        If the val_loss does not show a significant decrease in consecutive patience number of rounds, then return True.
        """
        val_loss = self.metrics["val_loss"]
        if len(val_loss) < patience + 1:
            return False

        recent_losses = val_loss[-(patience + 1):]
        best_loss = min(recent_losses[:-1])
        if recent_losses[-1] > best_loss - delta:
            print(f"[Monitor] Early stopping triggered. No improvement over last {patience} rounds.")
            return True
        return False

    def check_stability(self, window=5, threshold=0.01):
        """
        External interface: Check if adaptive strategy switching (convergence/oscillation) is required recently.
        Returning True indicates that "oscillation" or "rise" has been detected and switching is necessary.
        """
        val_loss = self.metrics["val_loss"]
        if len(val_loss) < window + 1:
            return False

        recent_losses = val_loss[-(window + 1):]
        diffs = np.diff(recent_losses)
        mean_change = np.mean(np.abs(diffs))

        if mean_change > threshold or np.sum(np.sign(diffs)[:-1] != np.sign(diffs)[1:]) >= window - 1:
            print(f"[Monitor] Instability detected (mean change {mean_change:.4f} > {threshold}).")
            return True
        return False

