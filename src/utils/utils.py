import torch
from transformers import AutoModelForSequenceClassification


def freeze_model(model: AutoModelForSequenceClassification, part: float) -> AutoModelForSequenceClassification:
    weights = sum([p.numel() for p in model.parameters()])
    threshold = part * weights

    params = 0
    for w in model.parameters():
        if params < threshold:
            params += w.numel()
            w.requires_grad = False
        else:
            break

    return model
