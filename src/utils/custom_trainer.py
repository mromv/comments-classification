import numpy as np
import torch

from hydra.utils import instantiate
from transformers import Trainer

from .losses import DiceLoss, FocalLoss


class CustomTrainer(Trainer):
    def __init__(self, loss, **kwargs):
        super(CustomTrainer, self).__init__(**kwargs)
        self.loss_func = instantiate(loss)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        if isinstance(self.loss_func, DiceLoss):
            labels = torch.nn.functional.one_hot(labels, num_classes=len(model.config.id2label))

        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_func(logits.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss
