import os
import pandas as pd
import pymorphy3
import nltk
import re

import hydra
from omegaconf import DictConfig

nltk.download("stopwords")


def source_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).rename(
        columns={"Комментарий": "Comment", "Категория": "Category", "Тег": "Tag"}
    )
    df = df.dropna(subset=["Category", "Comment", "Tag"])
    df = df.drop(df[df["Comment"].str.len() < 10].index)
    df["Comment"] = df["Comment"].str.lower()
    df = df.loc[~df["Category"].isin(["Качество материалов", "Интерфейс платформы", "Общение с куратором"])]
    df.reset_index(drop=True, inplace=True)
    return df[["Category", "Tag", "Comment"]]


def preprocess_comments(df: pd.DataFrame) -> pd.DataFrame:
    def del_space(text: str) -> str:
        return re.sub(" +", " ", text)

    def drop_patterns(text: str) -> str:
        patterns = r"[A-z0-9!‘’#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
        text = re.sub(patterns, " ", text) # отбросили паттерны
        text = re.sub(r"[\n\t]", " ", text)
        return del_space(text)

    def drop_emojis(text: str) -> str:
        emoj = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002500-\U00002BEF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"
            u"\u3030"
                          "]+", re.UNICODE)
        return re.sub(emoj, "", text)

    def remove_stopwords(text: str) -> str:
        russian_stopwords = nltk.corpus.stopwords.words("russian")
        return " ".join([word for word in text.split() if word not in (russian_stopwords)])

    morph = pymorphy3.MorphAnalyzer()

    def lemmatization(text: str) -> str:
        return " ".join([morph.parse(word)[0].normal_form for word in text.split()])

    df["data_patterns"] = df["Comment"].apply(drop_patterns).apply(drop_emojis)
    df["data_stopwords"] = df["data_patterns"].apply(remove_stopwords)
    df["data_lemma"] = df["data_stopwords"].apply(lemmatization)

    df = df[df['data_lemma'].str.len() > 3]

    return df


def preprocess_tags(df: pd.DataFrame) -> pd.DataFrame:

    def remove_sub_tags(tags: str) -> str:
        split = tags.split(sep=" ")
        new_tag = [x[:-1] if x[-1].isdigit() else x for x in split]
        return " ".join(new_tag)

    df["corrected_tag"] = df["Tag"].apply(
        lambda x: " ".join(re.findall(r"[A-Z]{1,2}\d|LMS", str(x)))
    ).apply(
        lambda x:
        re.sub(r"\s\s+", " ",
               re.sub(r"VC4|VP4|VC5|S4|T4|H4|EA1", "", x).strip()
               )
        .replace(r"GH3", "H3")
        .replace(r"HH3", "H3")
        .replace(r"BP3", "VP3")
        .replace(r"V3", "VC3")
        .replace(r"V2", "VP2")
    )

    df = df[df["corrected_tag"].apply(len) != 0]
    df = df.loc[~df["corrected_tag"].str.contains("E2")]
    df["Tag"] = df["corrected_tag"].apply(remove_sub_tags)
    return df.drop(columns="corrected_tag")


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    in_path = os.path.join(cfg["general"]["data_dir"], cfg["general"]["source_data"])
    out_path = os.path.join(cfg["general"]["data_dir"], "preprocessed_data_w_tags.csv")

    data = source_data(in_path)
    data = preprocess_comments(data)
    data = preprocess_tags(data)

    data.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
