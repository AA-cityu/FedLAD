import torch
import numpy as np
from collections import Counter
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class Evaluator:
    def __init__(self, model, test_loader, device):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def evaluate(self):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)

        print(" True label distribution:", Counter(y_true))
        print(" Predicted label distribution:", Counter(y_pred))

        print("\n Final Evaluation Results:")
        print(f"Accuracy     : {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision    : {precision_score(y_true, y_pred):.4f}")
        print(f"Recall       : {recall_score(y_true, y_pred):.4f}")
        print(f"F1 Score     : {f1_score(y_true, y_pred, average='weighted'):.4f}")
        try:
            print(f"ROC-AUC      : {roc_auc_score(y_true, y_prob):.4f}")
        except:
            print("ROC-AUC      : Not available (only one class present).")

        print("\nDetailed Report:")
        print(classification_report(y_true, y_pred, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        return y_true, y_pred, y_prob
