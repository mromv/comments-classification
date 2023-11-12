# comments-classification
ML practice

## Getting Started
This is an analyst workplace template for working with data hosted in a remote storage, where researches are formatted as `quarto` notebooks, utility source code is formatted as a python library, dependencies are managed using `pixi` and configurations are managed using `hydra`.

## Project Organization
```
├── README.md
│
├── src/                              <- Source code for use in `Quarto` reports.
│   │
│   ├── conf/                         <- `Hydra` configuration.
│   │   │
│   │   ├── data/                     <- `Hydra` group for data sources configs.
│   │   │   │
│   │   │   └── lakefs.yaml           <- `LakeFS` configuration.
│   │   │
│   │   ├── secrets/                  <- Sensitive configuration.
│   │   │
│   │   └── config.yaml               <- Main config for `Hydra` default values.
│   │
│   ├── config.py                     <- Example of `Hydra` config compose usage.
│   │
│   └── __init__.py
│
├── {task id| 0001}-{report-name}.qmd <- `Quarto` report.
│
├── ...
│
├── {task id| XXXX}-{report-name}.qmd
│
├── pixi.toml                         <- `Pixi` project file for
│                                        binary dependencies and python version.
└── pyproject.toml                    <- Python tools configurations.
```

## System requirements
- `git` 2.42+
- `python` 3.11+
- `quarto` 1.3+
- `pixi` 0.6+

### Pixi installation
There is a simple way of installing `pixi`. Simply execute the installation script in your preferred shell.

For Linux & macOS:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

For Windows:
```bash
iwr -useb https://pixi.sh/install.ps1 | iex
```

## Initial setup

### Pixi initial setup
Install `pixi` project with all dependencies:
```bash
pixi install
```

### Git initial setup
Create git repository in comments-classification, use `main` as `initial-branch` name:
```bash
git init --initial-branch=main
```

Set valid username and email:
```bash
git config user.name "Maksim Romanov"
git config user.email "mxromv@gmail.com"
```

### Pre-commit initial setup
Configure `pre-commit` hooks:

```bash
pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type commit-msg
```

### Work with sensitive information
Sensitive information is stored in the `secrets` folder and imported using `hydra`.

**Contact the person in charge for sensitive information.**

## How-to

### Pixi
`Pixi`` is a package management tool for developers. It allows the developer to install libraries and applications in a reproducible way, more about it you can read on it's [documentation page][pixi_docs].

#### Add dependencies
```bash
pixi add "python>=3.11"
```

#### Run the code inside the environment
```bash
pixi run python hello_world.py
```

#### Start a new shell in the environment
```bash
pixi shell
```

### Hydra
`Hydra` allows you to streamline the maintenance of different configurations and their use. To work in `quarto` notebooks, `hydra compose api` is used, more about it you can read on it's [documentation page][hydra_compose_api_docs].

All configurations are stored in the `conf` folder. Sub folders of `conf` folder are groups within which you can choose one of the configurations to use, for example:
```
├── conf
    ├── config.yaml
    └── data
        ├── lakefs.yaml
        └── postgresql.yaml
```
You have two data sources, and each source needs its own set of configurations. You can create a `data` folder inside `conf` and specify the configuration for different sources in separate files.

Then you can specify inside the `config.yaml` that by default you need to use the `lakefs` source configuration in the data group:
```
# conf/config.yaml
defaults:
  - data: lakefs
```

Now, when composing the config, if you do not specify which config should be used for the `data` group, the `lakefs` config will be used by default.

#### Compose conf folder into one dictionary
```python
from hydra import compose, initialize

with initialize(version_base=None, config_path=<conf folder path>)):
    hydra_config = compose(config_name=<config name>)
```

#### Change Hydra configuration on composing
Let's use the example above. Suppose you need to change the default configuration of the `data` group for a specific task. Then, you can use the `overrides` parameter, and specify a different value for the `data` group:
```python
from hydra import compose, initialize

initialize(version_base=None, config_path=<conf folder path>))
    hydra_config = compose(config_name=<config name>,
                           overrides=["data=postgresql"])
```

**In the same way, you can overwrite values from config files at startup, but this is not recommended.**

### Quarto
`Quarto` is an open-source scientific and technical publishing system built on `pandoc`. You can weave together narrative text and code to produce elegantly formatted output as documents, web pages, blog posts, books and more.

#### Install VSCode extension
Launch VS Code Quick Open (Ctrl+P), paste the following command, and press enter:
```
ext install quarto.quarto
```

### Ruff
An extremely fast Python linter and code formatter, written in Rust. Ruff can be used to replace Flake8 (plus dozens of plugins), Black, isort, pydocstyle, pyupgrade, autoflake, and more, all while executing tens or hundreds of times faster than any individual tool.

#### Install VSCode extension
Launch VS Code Quick Open (Ctrl+P), paste the following command, and press enter:
```
ext install charliermarsh.ruff
```

### Commitizen
Commitizen is release management tool designed for teams. Using a standardized set of rules to write commits, makes commits easier to read, and enforces writing descriptive commits.

#### Commit changes
```bash
cz commit
```

[pixi_docs]: https://prefix.dev/docs/pixi/overview
[hydra_compose_api_docs]: https://hydra.cc/docs/advanced/compose_api/
"# comments-classification"
