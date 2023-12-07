
from torch import nn
from transformers import AutoConfig, AutoModel , BatchEncoding
from transformers.model_outputs import BaseModelOutput
from src.utils.io_utils import translate_gcs_dir_to_local
class Backbone(nn.Module):
    pass


class HuggingFaceBackone(Backbone):
    def __init__(self,pretrained_model_name_or_path:str, pretrained:bool = False) -> None:
        super().__init__()


        self.backbone = self.get_backbone(pretrained_model_name_or_path,pretrained)

    def get_backone(self,pretrained_model_name_or_path : str,pretrained) -> nn.Module:
        path = translate_gcs_dir_to_local(pretrained_model_name_or_path)

        config = AutoConfig.from_pretrained(path)
        if self.pretrained:
            return AutoModel.from_pretrained(path,config=config)
        return AutoModel.from_config(config)
        

    def forward(self, encodings : BatchEncoding) -> BaseModelOutput:
        output : BaseModelOutput = self.backbone(**encodings)
        return output