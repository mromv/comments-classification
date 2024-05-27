import numpy as np
import torch
from datasets import Dataset
from transformers import Trainer

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    classification_report
)


def make_classification_report(trainer: Trainer, eval_dataset: Dataset) -> str:
    out = trainer.predict(eval_dataset)
    labels = [trainer.model.config.id2label[i] for i in range(len(trainer.model.config.id2label))]
    return classification_report(out.label_ids, np.argmax(out.predictions, axis=1), target_names=labels)


def multi_label_metrics(
    predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute multilabel metrics.

    Args:
        predictions (np.ndarray): logits array
        labels (np.ndarray): labels array
        threshold (float, optional): activation threshold. Defaults to 0.5.

    Returns:
        dict[str, float]: metrics dict
    """
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    f1_micro_average = f1_score(y_true=labels, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(labels, y_pred, average="micro")
    accuracy = accuracy_score(labels, y_pred)
    metrics = {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}
    return metrics
