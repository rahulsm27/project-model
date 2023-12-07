from torch import nn, Tensor
from typing import Optional
from src.models.backbones import Backbone
from src.models.adapters import Adapter
from src.models.heads import Head

from transformers import BatchEncoding

class Model(nn.Module):
    pass 

class BinaryTextClassificationModel(Model):
    def __init__(self, backbone:Backbone, head: Head, adapter: Optional[Adapter]) -> None:
        super().__init__()

        self.backone = backbone
        self.adapter = adapter
        self.head = head

    def forward(self, encodings : BatchEncoding) -> Tensor:

        output = self.backone(encodings)
        if self.adapter is not None:
            output = self.adapter(output)
        output =self.head(output)
        return output

    

