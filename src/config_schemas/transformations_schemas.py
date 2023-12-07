from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI


@dataclass
class TransformationConfig:
    _target_:str = MISSING

@dataclass
class HuggingFaceTokenizationTransformationConfig:
    _target_:str = "src.data_modules.transformations.HuggingFace"
    pretrained_tokenzier_name_or_path: str = MISSING
    max_sequence_length : int = MISSING

def setup_config()-> None:
    cs = ConfigStore.instance()
    cs.store(
        name = "transformation_schema",
        group = "tasks/data_module/transformation",
        node = HuggingFaceTokenizationTransformationConfig
    )