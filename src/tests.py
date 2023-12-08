import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.config_schemas.training.training_task_schemas import setup_config

setup_config()

@hydra.main(config_name="test_training_task",version_base=None)
def main(config : DictConfig):
    print("*" * 60)
    print(OmegaConf.to_yaml(config))
    print("*" * 60)

    # model = instantiate(config)

    # texts = ["hello how are you?"]
    # encodings = model.backbone.transformation(texts)
    # output=model(encodings)
    # print(f"{output.shape=}")

if __name__ == "__main__":
    main()