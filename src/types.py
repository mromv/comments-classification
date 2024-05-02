from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Any


@dataclass
class PreprocessingConfig:
    padding: str = "max_length"
    max_length: int = 512
    tokenizer_name: str = MISSING


@dataclass
class RubertPreprocessing(PreprocessingConfig):
    tokenizer_name: str = "cointegrated/rubert-tiny2"


@dataclass
class MpnetPreprocessing(PreprocessingConfig):
    tokenizer_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    max_length: int = 256


@dataclass
class ModelConfig:
    _target_: str = "transformers.AutoModelForSequenceClassification.from_pretrained"
    _convert_: str = "object"
    pretrained_model_name_or_path: str = MISSING
    problem_type: str = "single_label_classification"


@dataclass
class RubertModel(ModelConfig):
    pretrained_model_name_or_path: str = "cointegrated/rubert-tiny2"


@dataclass
class MpnetModel(ModelConfig):
    pretrained_model_name_or_path: str = (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )


@dataclass
class TrainingConfig:
    _target_: str = "transformers.TrainingArguments"
    output_dir: str = MISSING
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 180
    per_device_eval_batch_size: int = 180
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    report_to: str = "none"
    fp16: bool = True


@dataclass
class RubertTraining(TrainingConfig):
    output_dir: str = "rubert_outputs"


@dataclass
class MpnetTraining(TrainingConfig):
    output_dir: str = "mpnet_outputs"
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64


@dataclass
class SecretsConfig:
    hf_token: str = "${oc.env:HF_TOKEN}"


defaults = [
    "_self_",
    {"preprocessing": "rubert"},
    {"model": "rubert"},
    # {"model": MISSING},
    {"training": "rubert"},
    {"secrets": "secrets"},
]

small_classes: list[str] = [
    "Качество материалов",
    "Интерфейс платформы",
    "Общение с куратором",
]


@dataclass
class MainConfig:
    defaults: list[Any] = field(default_factory=lambda: defaults)
    preprocessing: PreprocessingConfig = MISSING
    model: ModelConfig = MISSING
    training: TrainingConfig = MISSING
    secrets: SecretsConfig = MISSING
    test_size: float = 0.2
    small_classes: list[str] = field(default_factory=lambda: small_classes)