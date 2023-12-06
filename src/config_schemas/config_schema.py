from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass


from src.config_schemas.infrastructure import infrastructure_schema

@dataclass
class Config:
    infrastructure: infrastructure_schema.InfrastructureConfig = infrastructure_schema.InfrastructureConfig

def setup_config() -> None:
    infrastructure_schema.setup_config()
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)