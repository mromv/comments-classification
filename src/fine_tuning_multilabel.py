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
from utils.custom_trainer import CustomTrainer
from utils.metrics import multi_label_metrics
from utils.utils import freeze_model

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback
)

log = logging.getLogger(__name__)


def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    """Metrics computation wrapper.

    Args:
        p (EvalPrediction): hf model output

    Returns:
        dict[str, float]: metrics dict
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


def make_training_pipeline(
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        train_conf: dict[str, Any],
        loss: torch.nn.Module,
        checkpoints_dir: str,
) -> Trainer:

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


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg["general"]["device"])
    data_path = os.path.join(cfg["general"]["data_dir"], "preprocessed_data_w_tags.csv")
    output_dir = os.path.join(cfg["general"]["output_dir"], cfg["encoder"]["short_name"], "Tag")
    checkpoints_dir = os.path.join(cfg["general"]["checkpoints"], cfg["encoder"]["short_name"], "Tag")

    tokenizer = instantiate(cfg["encoder"]["tokenizer"])

    dataset, id2label, label2id = load_dataset(
        data_path=data_path,
        use_col=cfg["data"]["column_use"],
        target="Tag",
        tokenizer=tokenizer,
        test_size=cfg["general"]["test_size"],
        max_length=cfg["data"]["max_length"],
        random_seed=cfg["general"]["random_state"],
        device=device
    )

    model = instantiate(
        cfg["encoder"]["model"], num_labels=len(id2label),
        id2label=id2label, label2id=label2id, problem_type="multi_label_classification"
    )
    model = freeze_model(model, cfg["fine_tuning"]["freeze"])

    trainer = make_training_pipeline(model, tokenizer, dataset, cfg["training"], cfg["loss"], checkpoints_dir)
    trainer.train()
    trainer.save_model(output_dir)

    # logging
    log.info(f"Fine tuning result: {cfg['encoder']['short_name']} with loss {cfg['loss']['_target_'].split('.')[-1]}")
    [log.info(f"{i + 1}: {epoch}") for i, epoch in enumerate(trainer.state.log_history[:-1])]
    log.info(f"{trainer.state.log_history[-1]}")
    log.info(f"\nvalidation: {trainer.evaluate()}")
    log.info(f"\ntest: {trainer.evaluate(eval_dataset=dataset['test'])}")
    

if __name__ == "__main__":
    main()
