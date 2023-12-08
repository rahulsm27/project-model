from src.config_schemas.data_module_schemas import DataModuleConfig
from src.config_schemas.trainer.trainer_schemas import TrainerConfig
from dataclasses import dataclass
from src.utils.mixins import LoggableParamsMixin
from omegaconf import MISSING

@dataclass
class LightningModuleConfig(LoggableParamsMixin):
    _target_: str = MISSING

@dataclass
class TaskConfig:
    _target_:str = MISSING
    name:str = MISSING
    data_module : DataModuleConfig = MISSING
    lightning_module : LightningModuleConfig = MISSING
    trainer:TrainerConfig = MISSING
    
    def loggable_params(self) -> list[str]:
        return ["_target_"]