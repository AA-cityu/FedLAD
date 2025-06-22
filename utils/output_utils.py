import csv
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from collections import Counter
import numpy as np


def save_full_report(y_true, y_pred, y_prob=None, summary_info=None,
                     txt_path="final_report.txt", csv_path="final_metrics.csv"):

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    with open(txt_path, "w") as f:
        f.write("=== Final Evaluation Report ===\n\n")

        f.write("True Label Distribution:\n")
        f.write(str(Counter(y_true)) + "\n\n")

        f.write("Predicted Label Distribution:\n")
        f.write(str(Counter(y_pred)) + "\n\n")

        f.write("Final Evaluation Results:\n")
        f.write(f"Accuracy     : {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"Precision    : {precision_score(y_true, y_pred):.4f}\n")
        f.write(f"Recall       : {recall_score(y_true, y_pred):.4f}\n")
        f.write(f"F1 Score     : {f1_score(y_true, y_pred, average='weighted'):.4f}\n")
        if y_prob is not None:
            try:
                f.write(f"ROC-AUC      : {roc_auc_score(y_true, y_prob):.4f}\n")
            except:
                f.write("ROC-AUC      : Not available (only one class present).\n")

        f.write("\nDetailed Report:\n")
        f.write(classification_report(y_true, y_pred, digits=4))

        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_true, y_pred)))

        # Enter the summary information
        if summary_info:
            f.write("\n\n=== Training Summary ===\n")
            for k, v in summary_info.items():
                f.write(f"{k}: {v}\n")

    print(f"Full evaluation report saved to {txt_path}")

    # Save core indicators to CSV
    header = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "early_stop_round", "strategy_switch_count", "wall_clock_time"]
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
        "roc_auc": -1
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except:
            pass

    # Add summary information
    if summary_info:
        metrics.update(summary_info)

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: metrics.get(k, -1) for k in header})

    print(f"Key evaluation metrics saved to {csv_path}")
