from omegaconf import OmegaConf, MISSING
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass


@dataclass
class BackboneConfig:
    _target_: str = MISSING


@dataclass
class HuggingFaceBackboneConfig(BackboneConfig):
    _target_:str = "src.moels.backones.HuggingFaceBackbone"
    pretrained_model_name_or_path : str = MISSING
    pretrained : bool = False


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="hugging_face_backbone_schema",
        group="tasks/lightning_module/model/backbone",
        node=HuggingFaceBackboneConfig,
    )