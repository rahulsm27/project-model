from dataclasses import dataclass, field

from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from src.config_schemas.base_schemas import TaskConfig

from src.config_schemas.config_schema import Config

from src.config_schemas.training.training_task_schemas import DefaultCommonTrainingTaskConfig

@dataclass
class LocalBertExperiment(Config):
    tasks :dict[str,TaskConfig] = field(default_factory=lambda:{"binary_text_classifcation_task" : DefaultCommonTrainingTaskConfig})

# to overwrite individual parameters
FinalLocalBertExperiment = OmegaConf.merge(
    LocalBertExperiment,
    OmegaConf.from_dotlist([])
)

cs = ConfigStore.instance()
cs.store(name="local_bert", group="experiment/bert",node=FinalLocalBertExperiment, package="_global_" )