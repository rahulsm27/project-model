from abc import abstractmethod
from torch import nn, Tensor
from typing import Optional
from src.models.backbones import Backbone
from src.models.adapters import Adapter
from src.models.heads import Head

from src.data_modules.transformations import Transformation
from transformers import BatchEncoding

class Model(nn.Module):
    @abstractmethod
    def get_trasformation(self) -> Transformation :
        ...

class BinaryTextClassificationModel(Model):
    def __init__(self, backbone:Backbone, head: Head, adapter: Optional[Adapter]) -> None:
        super().__init__()

        self.backbone = backbone
        self.adapter = adapter
        self.head = head

    def forward(self, encodings : BatchEncoding) -> Tensor:

        output = self.backbone(encodings)
        if self.adapter is not None:
            output = self.adapter(output)
        output =self.head(output)
        return output
    
    def get_trasformation(self) -> Transformation:
        return self.backbone.get_transfomration()

    

