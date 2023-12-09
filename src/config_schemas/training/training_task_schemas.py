from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI

from src.config_schemas import data_module_schemas
from src.config_schemas.base_schemas import TaskConfig
from src.config_schemas.trainer import trainer_schemas
from src.config_schemas.training import training_lightning_module_schemas


@dataclass
class TrainingTaskConfig(TaskConfig):
    best_training_checkpoint: str = SI("${infrastructure.mlflow.artifact_uri}/best-checkpoints/last.ckpt")
    last_training_checkpoint: str = SI("${infrastructure.mlflow.artifact_uri}/last-checkpoints/last.ckpt")


@dataclass
class TarModelExportingTrainingTaskConfig(TrainingTaskConfig):
    tar_model_export_path: str = SI("${infrastructure.mlflow.artifact_uri}/exported_model.tar.gz")


@dataclass
class CommonTrainingTaskConfig(TrainingTaskConfig):
    _target_: str = "src.training.tasks.common_training_task.CommonTrainingTask"


@dataclass
class DefaultCommonTrainingTaskConfig(TarModelExportingTrainingTaskConfig):
    _target_ :str = "src.training.tasks.tar_model_exporting_training_task.TarModelExportingTrainingTask"
    name : str = "BinaryTextClassification"

    data_module: data_module_schemas.DataModuleConfig = (
        data_module_schemas.ScrappedDataTextClassificationDataModuleConfig()
    )
    lightning_module: training_lightning_module_schemas.TrainingLightningModuleConfig = (
        training_lightning_module_schemas.DefaultBinaryTextClassificationTrainingLightningModuleConfig()
    )
    trainer: trainer_schemas.TrainerConfig = trainer_schemas.GPUDev()
    tar_model_export_path : str = SI("${infrastructure.mlflow.artifact_uri}/exported_model.tar.gz")

def setup_config() -> None:
    data_module_schemas.setup_config()
    training_lightning_module_schemas.setup_config()
    trainer_schemas.setup_config()

    cs = ConfigStore.instance()
    cs.store(
        name="common_training_task_schema",
        group="tasks",
        node=CommonTrainingTaskConfig,
    )

    cs.store(
        name="test_training_task",
     #   group="tasks",
        node=DefaultCommonTrainingTaskConfig,
    )    