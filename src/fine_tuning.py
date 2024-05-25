import os
import hydra
import logging
from typing import Any
from hydra.utils import instantiate
from omegaconf import DictConfig

import pandas as pd
import numpy as np
import torch

from load_data import load_dataset

from custom_trainer import (
    CustomTrainer,
    make_classification_report
)

from datasets import (
    load_metric,
    Dataset
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback
)

log = logging.getLogger(__name__)


def make_training_pipeline(
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        train_conf: dict[str, Any],
        loss: torch.nn.Module,
        checkpoints_dir: str,
        eval_metric: str="f1"
) -> Trainer:

    metric = load_metric(eval_metric)

    def compute_metrics(eval_pred: EvalPrediction) -> float:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="weighted")

    args = instantiate(train_conf, output_dir=checkpoints_dir)

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        loss=loss
    )
    return trainer


def freeze_model(model: AutoModelForSequenceClassification, part: float) -> AutoModelForSequenceClassification:
    weights = sum([p.numel() for p in model.parameters()])
    threshold = part * weights

    params = 0
    for w in model.parameters():
        if params <= threshold:
            params += w.numel()
            w.requires_grad = False
        else:
            break

    return model

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg["general"]["device"]) if torch.cuda.is_available() else torch.device("cpu")
    data_path = os.path.join(cfg["general"]["data_dir"], "2_preprocessed_data.csv")
    output_dir = os.path.join(cfg["general"]["output_dir"], cfg["encoder"]["short_name"])
    checkpoints_dir = os.path.join(cfg["general"]["checkpoints"], cfg["encoder"]["short_name"])

    tokenizer = instantiate(cfg["encoder"]["tokenizer"])

    dataset, id2label, label2id = load_dataset(
        data_path=data_path,
        use_col=cfg["data"]["column_use"],
        tokenizer=tokenizer,
        test_size=cfg["general"]["test_size"],
        max_length=cfg["data"]["max_length"],
        random_seed=cfg["general"]["random_state"],
        device=device
    )

    model = instantiate(
        cfg["encoder"]["model"], num_labels=len(id2label), id2label=id2label, label2id=label2id
    )
    model = freeze_model(model, cfg["fine_tuning"]["freeze"])

    trainer = make_training_pipeline(model, tokenizer, dataset, cfg["training"], cfg["loss"], checkpoints_dir)
    trainer.train()
    trainer.save_model(output_dir)

    report = make_classification_report(trainer, dataset["test"])

    # logging
    log.info(f"Fine tuning result: {cfg['encoder']['short_name']} with loss {cfg['loss']['_target_'].split('.')[-1]}")
    [log.info(f"{i + 1} epoch: {epoch}") for i, epoch in enumerate(trainer.state.log_history[:-1])]
    log.info(f"{trainer.state.log_history[-1]}")
    log.info(f"\n{report}")


if __name__ == "__main__":
    main()
