from src.config_schemas.config_schema import Config
from src.utils.config_utils import get_config
from src.utils.utils import get_logger
from src.utils.torch_utils import get_local_rank
import torch
from lightning.pytorch import seed_everything
from omegaconf import OmegaConf

from hydra.utils import instantiate
# custom decorater created
@get_config(config_path="../configs/automatically_generated", config_name="config", to_object=False ,return_dict_config=True)
def run_tasks(config: Config) -> None:
   # print(OmegaConf.to_yaml(config,resolve=True))
   # exit()
    logger = get_logger(__file__)
    assert config.infrastructure.mlflow.run_id is not None, "Run id has to be set for running tasks"

    backend = "gloo" # for cpu 
    if torch.cuda.is_available():
       torch.cuda.set_device(f"cuda:{get_local_rank()}")
       backend = "nccl" # for gpu

    torch.distributed.init_process_group(backend=backend)

    seed_everything(seed=config.seed, workers =True) #seed everthing pytorch lightning module, model etc..so that every run produces same result

    for task_name, task_config in config.tasks.items():
        logger.info(F"Running task : {task_name}")
        task  = instantiate(task_config)
        task.run(config=config, task_config=task_config)

if __name__ == "__main__":
    run_tasks()  # type: ignore