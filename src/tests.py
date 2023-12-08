import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.config_schemas.models.backbone_schemas import setup_config

setup_config()

@hydra.main(config_name="test_backbone_config",version_base=None)
def main(config : DictConfig):
    print("*" * 60)
    print(OmegaConf.to_yaml(config))
    print("*" * 60)

    backone = instantiate(config)
    print(backone)

if __name__ == "__main__":
    main()