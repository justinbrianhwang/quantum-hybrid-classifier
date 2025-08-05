import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

    def evaluate(self, test_loader):
        self.model.eval()
        all_preds, all_probs, all_targets = [], [], []

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device).float()
                output = self.model(data)

                if output.shape[1] == 1:
                    probs = output.cpu().numpy().flatten()
                    preds = (output > 0.5).float().cpu().numpy().flatten()
                else:
                    probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
                    preds = output.argmax(dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_probs.extend(probs)
                all_targets.extend(target.numpy())

        acc = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
        auc_roc = roc_auc_score(all_targets, all_probs)

        print("=== Model Evaluation ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")

        self._plot_confusion_matrix(all_targets, all_preds)

        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc
        }

    def _plot_confusion_matrix(self, targets, preds):
        cm = confusion_matrix(targets, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        plt.show()

