import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import pandas as pd
import numpy as np
import torch

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer


class SentenceTransformerSelector(BaseEstimator, TransformerMixin):
    def __init__(self, model, output_type="sentence_embedding", device="CPU"):
        self.device = torch.device(device)
        self.model = instantiate(model)

        assert isinstance(self.model, CountVectorizer | SentenceTransformer)
        assert output_type in ["sentence_embedding", "token_embeddings-mean", "token_embeddings-idf"]

        self.based_transformer = isinstance(self.model, SentenceTransformer)
        self.output_type = output_type.split("-")[0]
        self.tfidf = None

        if self.based_transformer and self.output_type == "token_embeddings":
            self.pool = output_type.split("-")[1]
            if self.pool == "idf":
                self.tfidf = TfidfVectorizer(use_idf=True,
                                             tokenizer=self.model.tokenizer.encode,
                                             token_pattern=None)

    def fit(self, X):
        if self.based_transformer:
            if self.tfidf:
                self.tfidf.fit(X)
                self.tok_ids_idfs = {tok: self.tfidf.idf_[self.tfidf.vocabulary_[tok]]
                                     for tok in self.tfidf.vocabulary_.keys()}

        else:
            self.model.fit(X)

        return self

    def transform(self, X) -> np.ndarray:
        if self.based_transformer:
            # if transformer => to get embeddings
            model_output = self.model.encode(X, output_value=None)

            if self.output_type == "token_embeddings":
                # embeddings for each token
                tok_embeddings = [i["token_embeddings"] for i in model_output]

                if self.pool == "mean":
                    # mean of token embeddings
                    out = torch.vstack([tens.mean(axis=0) for tens in tok_embeddings]).cpu().numpy()

                else:
                    # embeddings weighted by idf
                    out_ids = [i["input_ids"] for i in model_output]
                    tens_idf = [
                                torch.FloatTensor([
                                    self.tok_ids_idfs.get(tok.item(), 1.0) for tok in tens]).to(self.device)
                                        for tens in out_ids
                                ]

                    out = torch.vstack([
                        (tens_idf[tens_ind].unsqueeze(1) * tok_embeddings[tens_ind]).sum(axis=0) / tens_idf[tens_ind].sum()
                            for tens_ind in range(len(tok_embeddings))
                                        ]).cpu().numpy()

            else:
                # sentence embedding
                out = np.array([emb["sentence_embedding"].cpu() for emb in model_output])

        else:
            # count based approach
            out = self.model.transform(X).toarray()

        return out


@hydra.main(version_base=None, config_path="../../src/conf", config_name="config")
def main(cfg: DictConfig):
    data_path = os.path.join(cfg["general"]["data_dir"], "2_preprocessed_data.csv")
    save_path = os.path.join(cfg["general"]["data_dir"], "embeddings", f"{cfg['encoder']['short_name']}")

    data = pd.read_csv(data_path)

    model = SentenceTransformerSelector(
        cfg["encoder"]["SentenceEncoder"],
        cfg["encoder"]["encode_output"],
        cfg["general"]["device"]
    )
    model.fit(data[cfg["data"]["column_use"]])

    np.save(save_path, model.transform(data[cfg["data"]["column_use"]]))


if __name__ == "__main__":
    main()
