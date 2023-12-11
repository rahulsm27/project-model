from typing import TYPE_CHECKING
from pathlib import Path
from omegaconf import OmegaConf , DictConfig
from src.utils.config_utils import get_config_and_dict_config,save_config_as_yaml
from src.utils.mlflow_utils import activate_mlflow,log_training_hparams
from src.utils.config_utils import save_config_as_yaml
import mlflow
if TYPE_CHECKING:
    from src.config_schemas.config_schema import Config
# Yes, you are right, this is a little bit confusing. This is because of the way Hydra works. We are using their decorator in our decorator to read the config file. And their decorator works with relative path to where it was defined. Because we are using Hydra's decorator under utils module, the relative path to the config's directory is ../configs
@get_config_and_dict_config(config_path="../configs", config_name="config" )
def generate_final_config(config : "Config",dict_config : DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    with activate_mlflow(config.infrastructure.mlflow.experiment_name ,run_id= config.infrastructure.mlflow.run_id,run_name=config.infrastructure.mlflow.run_name) as run:
        # getting mlflow paramters
        run_id : str = run.info.run_id
        experiment_id : str = run.info.experiment_id
        artifact_uri:str = run.info.artifact_uri


        dict_config.infrastructure.mlflow.artifact_uri = artifact_uri
        dict_config.infrastructure.mlflow.run_id = run_id
        dict_config.infrastructure.mlflow.experiment_id = experiment_id

        # save config paramters
        config_save_dir = Path("./src/configs/automatically_generated/")
        config_save_dir.mkdir(parents=True, exist_ok=True)
        (config_save_dir / "__init__.py").touch(exist_ok=True)
        

        yaml_config_save_path = config_save_dir / "config.yaml"
        save_config_as_yaml(dict_config,str(yaml_config_save_path))
        
        # saving config to mlflow artifact store
        mlflow.log_artifact(str(yaml_config_save_path)) 


        log_training_hparams(config)

if __name__ == "__main__" :
    generate_final_config()