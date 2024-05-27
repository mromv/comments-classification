import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def label_encode(series: pd.Series) -> tuple[LabelEncoder, pd.Series]:
    le = LabelEncoder()
    series = le.fit_transform(series)
    return le, series


def tags_encode(series: pd.Series) -> tuple[LabelEncoder, pd.Series]:
    mlb = MultiLabelBinarizer()
    series = mlb.fit_transform(series.str.split()).astype(float).tolist()
    return mlb, series


def split(
    dataframe: pd.DataFrame,
    target: str,
    test_size: float,
    random_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    all_ids = np.arange(len(dataframe))
    train_idx, temp = train_test_split(all_ids, test_size=test_size,
                                       random_state=random_seed,
                                       stratify=dataframe["labels"] if target=="Category" else None
                                       )
    test_idx, val_idx = train_test_split(temp, test_size=test_size,
                                         random_state=random_seed,
                                         stratify=dataframe.iloc[temp]["labels"] if target=="Category" else None
                                         )

    return train_idx, val_idx, test_idx


def dataset_tokenize(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int,
    device: torch.device
) -> DatasetDict:

    dataset = dataset.map(lambda e:
                          tokenizer(
                              e["text"],
                              truncation=True,
                              max_length=max_length,
                              padding="max_length"),
                          batched=True, remove_columns="text")

    dataset.set_format(type="torch", device=device)
    return dataset


def load_dataset(
    data_path: str,
    use_col: str,
    target: str,
    tokenizer: AutoTokenizer,
    test_size: float,
    max_length: int,
    random_seed: int,
    device: torch.device,
) -> tuple[DatasetDict, dict, dict]:

    df = pd.read_csv(data_path, usecols=[use_col, "Category", "Tag"]).rename(\
            columns={use_col: "text"})

    if target == "Tag":
        le, df["labels"] = tags_encode(df["Tag"])
    else:
        le, df["labels"] = label_encode(df["Category"])

    df = df[["labels", "text"]]
    df = df.iloc[:1000]

    train_idx, val_idx, test_idx = split(df, target, test_size, random_seed)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(df.loc[train_idx], preserve_index=False),
        "val": Dataset.from_pandas(df.loc[val_idx], preserve_index=False),
        "test": Dataset.from_pandas(df.loc[test_idx], preserve_index=False)
    })
    dataset = dataset_tokenize(dataset, tokenizer, max_length, device)

    id2label = {i: j for i, j in enumerate(le.classes_)}
    label2id = {j: i for i, j in enumerate(le.classes_)}

    return dataset, id2label, label2id
