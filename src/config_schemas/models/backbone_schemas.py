from omegaconf import OmegaConf, MISSING
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

from src.config_schemas.transformations_schemas import TransformationConfig
@dataclass
class BackboneConfig:
    _target_: str = MISSING
    transformations : TransformationConfig = MISSING # it happens that we have defined a tranformation class in transformation_schme thats why we have fetched from ther


@dataclass
class HuggingFaceBackboneConfig(BackboneConfig):
    _target_:str = "src.models.backones.HuggingFaceBackbone"
    pretrained_model_name_or_path : str = MISSING
    pretrained : bool = False


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="hugging_face_backbone_schema",
        group="tasks/lightning_module/model/backbone",
        node=HuggingFaceBackboneConfig,
    )