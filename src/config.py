from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize
from omegaconf import OmegaConf
from pydantic import BaseModel


class LakeFS(BaseModel):
    """LakeFS security config."""

    host: str
    username: str
    password: str


class Config(BaseModel):
    """Config model for typing."""

    data: LakeFS


def compose_config(
    overrides: list[str] | None = None,
    config_path: Path = Path("conf"),
    config_name: str = "config",
) -> Config:
    """Get Hydra config dictionary.

    Initializes environment variables, composes configuration files and returns hydra
    config dictionary.

    Args:
        overrides (list[str], optional): overrides config values.
                Defaults to None,
        config_path (Path, optional): Path to conf folder. Defaults to Path("conf").
        config_name (str, optional): Config name from conf folder.
                Defaults to "config".

    Returns:
        DictConfig: Hydra config dictionary.
    """
    with initialize(version_base=None, config_path=str(config_path)):
        hydra_config = compose(config_name=config_name, overrides=overrides)
        dict_config: dict[str, Any] = cast(
            dict[str, Any], OmegaConf.to_container(hydra_config, resolve=True)
        )
        return Config(**dict_config)
