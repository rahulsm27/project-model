from omegaconf import OmegaConf, MISSING
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

from src.config_schemas.models.transformation_schemas import TransformationConfig,CustomHuggingFaceTokenizationTrasnformationConfig

from src.utils.mixins import LoggableParamsMixin


@dataclass
class BackboneConfig(LoggableParamsMixin):
    _target_: str = MISSING
    transformation : TransformationConfig = MISSING # it happens that we have defined a tranformation class in transformation_schme thats why we have fetched from ther
    def loggable_params(self) -> list[str]:
        return ["_target_"]

@dataclass
class HuggingFaceBackboneConfig(BackboneConfig):
    _target_:str = "src.models.backbones.HuggingFaceBackbone"
    pretrained_model_name_or_path : str = MISSING
    pretrained : bool = False

    def loggable_params(self) -> list[str]:
        return super().loggable_params() + ["pretrained_model_name_or_path", "pretrained"]


@dataclass
class BertTinyHuggingFaceBackboneConfig(HuggingFaceBackboneConfig):
    
    transformation : TransformationConfig = CustomHuggingFaceTokenizationTrasnformationConfig
    pretrained_model_name_or_path : str = "prajjwal1/bert-tiny"
  



def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="hugging_face_backbone_schema",
        group="tasks/lightning_module/model/backbone",
        node=HuggingFaceBackboneConfig,
    )

    cs = ConfigStore.instance()
    cs.store(
        name="test_backbone_config",
       # group="tasks/data_module",
        node=BertTinyHuggingFaceBackboneConfig,
    )