import numpy as np
import torch

from hydra.utils import instantiate
from sklearn.metrics import classification_report
from loss.losses import FocalLoss, DiceLoss
from datasets import Dataset
from transformers import Trainer


class CustomTrainer(Trainer):
    def __init__(self, loss, **kwargs):
        super(CustomTrainer, self).__init__(**kwargs)
        self.loss_func = instantiate(loss)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        if isinstance(self.loss_func, DiceLoss):
            labels = torch.nn.functional.one_hot(labels, num_classes=len(model.config.id2label))

        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = self.loss_func(logits.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss


def predict(logits: torch.Tensor) -> np.ndarray:
    s = torch.nn.Softmax()
    probs = s(torch.tensor(logits))
    return np.argmax(probs, axis=1)


def make_classification_report(trainer: Trainer, eval_dataset: Dataset) -> str:
    out = trainer.predict(eval_dataset)
    labels = [trainer.model.config.id2label[i] for i in range(len(trainer.model.config.id2label))]
    return classification_report(out.label_ids, np.argmax(out.predictions, axis=1), target_names=labels)