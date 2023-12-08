from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI


@dataclass
class TransformationConfig:
    _target_:str = MISSING

@dataclass
class HuggingFaceTokenizationTransformationConfig(TransformationConfig):
    _target_:str = "src.data_modules.transformations.HuggingFaceTokenizationTransformation"
    pretrained_tokenizer_name_or_path: str = MISSING
    max_sequence_length : int = MISSING

@dataclass
class CustomHuggingFaceTokenizationTrasnformationConfig(HuggingFaceTokenizationTransformationConfig):
    pretrained_tokenizer_name_or_path : str = "gs://cyberbully_r/data/processed/default_run/trained_tokenizer"
    max_sequence_length : int = 100


def setup_config()-> None:
    cs = ConfigStore.instance()
    cs.store(
        name = "transformation_schema",
        group = "tasks/data_module/transformation",
        node = HuggingFaceTokenizationTransformationConfig
    )