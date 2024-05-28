rule all:
    input:
        expand("results/{model}/{target}", model=["rubert_tiny_2", "ruRoberta"], target=["tag", "category"])


rule preprocessing:
    input:
        path="data/raw/practice_cleaned.csv",
    output:
        path="data/interim/{target}.csv"
    shell:
        "python src/preprocessing.py\
        data.input_path={input}\
        data.interim_output={output}\
        data.target={wildcards.target}"


rule train_model:
    input:
        path="data/interim/{target}.csv"
    output:
        path=directory("results/{model}/{target}")
    shell:
        "python src/fine_tuning.py\
        encoder={wildcards.model}\
        data.interim_output={input}\
        model.output_dir={output}\
        data.target={wildcards.target}"
