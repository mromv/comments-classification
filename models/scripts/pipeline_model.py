import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

os.environ['HYDRA_FULL_ERROR'] = '1'

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../src/conf", config_name="config")
def main(cfg: DictConfig):
    # settings
    random_state = cfg["general"]["random_state"]
    data_path = os.path.join(cfg["general"]["data_dir"], "2_preprocessed_data.csv")
    embeddings_path = os.path.join(cfg["general"]["data_dir"], "embeddings", cfg["data"]["ml_data"])

    # load data
    X = np.load(embeddings_path)
    target = pd.read_csv(data_path)["Category"]

    le = LabelEncoder()
    y = le.fit_transform(target)

    # split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=cfg["general"]["test_size"],
                                                        random_state=random_state, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5,
                                                    random_state=random_state, stratify=y_temp)

    pipeline = Pipeline([
        ('dim_reducer', instantiate(cfg["dim_reduce"]["Model"])),
        ('classifier', instantiate(cfg["classifier"]["Model"]))
    ])

    pipeline.fit(X_train, y_train)

    sets = ["train", "validation", "test"]

    log.info(f"Data: {cfg['data']['ml_data']}; \nPipeline params: {pipeline.get_params()}")
    for i, (x_temp, y_temp) in enumerate(zip([X_train, X_val, X_test], [y_train, y_val, y_test])):
        log.info(f"Scores on {sets[i]} set:")
        log.info(f"{classification_report(y_temp, pipeline.predict(x_temp))}")


if __name__ == "__main__":
    main()