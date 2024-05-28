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
from utils.metrics import make_classification_report, multi_label_metrics
from utils.utils import freeze_model

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

from peft import (
    LoraConfig,
    get_peft_model
)

log = logging.getLogger(__name__)


def make_training_pipeline(
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        target: str,
        train_conf: dict[str, Any],
        loss: torch.nn.Module,
        checkpoints_dir: str,
        eval_metric: str = "f1"
) -> Trainer:

    if target == "tag":
        def compute_metrics(p: EvalPrediction) -> dict[str, float]:
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            result = multi_label_metrics(predictions=preds, labels=p.label_ids)
            return result

    else:  # category
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


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg["general"]["device"]) if torch.cuda.is_available() else torch.device("cpu")
    checkpoints_dir = os.path.join(cfg["model"]["checkpoints"], cfg["encoder"]["short_name"])
    target = cfg["data"]["target"]

    tokenizer = instantiate(cfg["encoder"]["tokenizer"])

    dataset, id2label, label2id = load_dataset(
        data_path=cfg["data"]["interim_output"],
        use_col=cfg["data"]["column_use"],
        target=target,
        tokenizer=tokenizer,
        test_size=cfg["general"]["test_size"],
        max_length=cfg["data"]["max_length"],
        random_seed=cfg["general"]["random_state"],
        device=device
    )

    model = instantiate(
        cfg["encoder"]["model"], num_labels=len(id2label), id2label=id2label, label2id=label2id,
        problem_type="multi_label_classification" if target == "tag" else "single_label_classification"
    )

    if cfg["fine_tuning"]["lora"]:
        checkpoints_dir = os.path.join(checkpoints_dir, "lora")

        config = LoraConfig(
            r=cfg["peft"]["r"],
            lora_alpha=cfg["peft"]["lora_alpha"],
            target_modules=cfg["peft"]["target_modules"],
            lora_dropout=cfg["peft"]["lora_dropout"],
            bias=cfg["peft"]["bias"],
            modules_to_save=cfg["peft"]["modules_to_save"]
        )
        model = get_peft_model(model, config)
    else:
        model = freeze_model(model, cfg["fine_tuning"]["freeze"])

    trainer = make_training_pipeline(model, tokenizer, dataset, target, cfg["training"], cfg["loss"], checkpoints_dir)
    trainer.train()
    trainer.save_model(cfg["model"]["output_dir"])

    # logging
    log.info(f"Fine tuning result for {target}: "
             f"{cfg['encoder']['short_name']} with loss: {cfg['loss']['_target_'].split('.')[-1]} "
             f"lora: {cfg['fine_tuning']['lora']}")
    [log.info(f"{i + 1}: {epoch}") for i, epoch in enumerate(trainer.state.log_history[:-1])]
    log.info(f"{trainer.state.log_history[-1]}")

    if target == "tag":
        log.info(f"\nvalidation: {trainer.evaluate()}")
        log.info(f"\ntest: {trainer.evaluate(eval_dataset=dataset['test'])}")
    else:  # category
        report = make_classification_report(trainer, dataset["test"])
        log.info(f"\n{report}")


if __name__ == "__main__":
    main()
