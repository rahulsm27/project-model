from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Union

from lightning.pytorch import Trainer

from src.data_modules.data_modules import DataModule, PartialDataModuleType
from src.training.lightning_modules.bases import TrainingLightningModule
from src.utils.utils import get_logger
from src.training.tasks.bases import TrainingTask
from src.utils.mlflow_utils import activate_mlflow,log_artifacts_for_reproducibility
from src.utils.io_utils import is_file


if TYPE_CHECKING:
    from src.config_schemas.config_schema import Config
    from src.config_schemas.training.training_task_schemas import TrainingTaskConfig

class CommonTrainingTask(TrainingTask):

    def __init__(self, 
                 name: str, 
                 data_module: Union[DataModule,  PartialDataModuleType], 
                 lightning_module: TrainingLightningModule, trainer: Trainer, 
                 best_training_checkpoint: str, last_training_checkpoint: str) -> None:
        super().__init__(
            name=name, 
            data_module=data_module, 
            lightning_module=lightning_module, 
            trainer=trainer, 
            best_training_checkpoint=best_training_checkpoint, 
            last_training_checkpoint=last_training_checkpoint
            )
        
    def run(self,config:"Config", task_config:"TrainingTaskConfig")->None:
        experiment_name = config.infrastructure.mlflow.experiment_name
        run_id = config.infrastructure.mlflow.run_id
        run_name = config.infrastructure.mlflow.run_name
        with activate_mlflow(experiment_name=experiment_name,run_id=run_id,run_name=run_name) as _:
            if self.trainer.is_global_zero: # if mutliple gpus are used global zero means we perform only once
                log_artifacts_for_reproducibility()
         #       log_training_hparams(config)
            assert isinstance(self.data_module, DataModule)
            if is_file(self.last_training_checkpoint):
                self.logger.info("Found checkpoint her :{self.last_training_checkpoint}. Resume training..")
                self.trainer.fit(model=self.lightning_module, datamodule=self.data_module, ckpt_path = self.last_training_checkpoint)
            else:
                self.logger.info("Starting fresh training")
                self.trainer.fit(model=self.lightning_module,datamodule=self.data_module)

            self.logger.info("training finished..")
 