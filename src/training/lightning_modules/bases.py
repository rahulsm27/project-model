from abc import abstractmethod
from typing import Callable,Iterable, Optional, Union, Any
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.training.loss_functions import LossFunction
from src.training.schedulers import LightningScheduler

from src.models.models import Model
from src.utils.utils import get_logger

PartialOptimizerType = Callable[[Union[Iterable[Tensor], dict[str, Iterable[Tensor]]]], Optimizer]

from torchmetrics.classification import BinaryAccuracy,BinaryF1Score,BinaryConfusionMatrix

class TrainingLightningModule(LightningModule):

    def __init__(
            self,
            model : Model,
            loss : LossFunction,
            optimizer : PartialOptimizerType,
            scheduler : Optional[LightningScheduler] = None
    ) -> None:
        super().__init__()

        self.model = model
        self.loss = loss
        self.partial_optimzier = optimizer
        self.scheduler = scheduler

        self.logging_logger = get_logger(self.__class__.__name__)

    def configure_optimizers(self)-> Union[Optimizer, tuple[list[Optimizer],list[dict[str,Any]]]] :
        optimizer = self.partial_optimizer(self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler.configure_scheduler(optimizer=optimizer,estimated_stepping_batches=self.trainer.estimated_stepping_batches)#automatically defined in self.trainer
            return [optimizer],[scheduler]
        else:
            return optimizer


    @abstractmethod
    def training_step(self,batch: Any, batch_idx: int)-> Tensor:
        ...

    @abstractmethod
    def validation_step(self,batch: Any, batch_idx: int)-> Tensor:
        ...
    
    
