from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI
from src.utils.mixins import LoggableParamsMixin

@dataclass
class TransformationConfig(LoggableParamsMixin):
    _target_:str = MISSING

    def loggable_params(self) -> list[str]:
        return ["_target_"]

@dataclass
class HuggingFaceTokenizationTransformationConfig(TransformationConfig):
    _target_:str = "src.data_modules.transformations.HuggingFaceTokenizationTransformation"
    pretrained_tokenizer_name_or_path: str = MISSING
    max_sequence_length : int = MISSING

    def loggable_params(self) -> list[str]:
        return super().loggable_params() +["pretrained_tokenizer_name_or_path","max_sequence_length"]
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