from dataclasses import dataclass, field
from typing import Any

from omegaconf import SI
from typing import Optional

from src.infrastructure.instance_template_creator import VMType


@dataclass
class BootDiskConfig:
    project_id: str = "deeplearning-platform-release"
    name: str = "common-cu113-v20230925" #cuda image
    size_gb: int = 50
    labels: Any = SI("${..labels}")


@dataclass
class VMConfig:
    machine_type: str = "n1-standard-8"
    accelerator_count: int = 1
    accelerator_type: str = "nvidia-tesla-t4"
    vm_type: VMType = VMType.STANDARD
    disks: list[str] = field(default_factory=lambda: [])


@dataclass
class VMMetadataConfig:
    instance_group_name: str = SI("${infrastructure.instance_group_creator.name}")
    docker_image : Optional[str] = SI("${docker_image}")
    zone: str = SI("${infrastructure.zone}")
    python_hash_seed: int = 42
    mlflow_tracking_uri : str = SI("${infrastructure.mlflow.mlflow_internal_tracking_uri}")
    node_count: int = SI("${infrastructure.instance_group_creator.node_count}")
    disks: list[str] = SI("${..vm_config.disks}")
    etcd_ip: Optional[str] = SI("${infrastructure.etcd_ip}")
# here we creating a class of Instancetempalte creator config created in instance_tempalte_creator and passsing values to this class from above.
# Above has some default values and some values to be fetched
@dataclass
class InstanceTemplateCreatorConfig:
    _target_: str = "src.infrastructure.instance_template_creator.InstanceTemplateCreator"
    scopes: list[str] = field(
        default_factory=lambda: [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/cloud.useraccounts.readonly",
            "https://www.googleapis.com/auth/cloudruntimeconfig",
        ]
    )
    network: str = SI("https://www.googleapis.com/compute/v1/projects/mlendtoend/global/networks/default")
    subnetwork: str = "https://www.googleapis.com/compute/v1/projects/mlendtoend/regions/europe-west4/subnetworks/default"
#    SI(
#        "https://www.googleapis.com/compute/v1/projects/${.project_id}/regions/${infrastructure.region}/subnetworks/default"
#    )
    startup_script_path: str = "scripts/vm_startup/task_runner_startup_script.sh"
    vm_config: VMConfig = VMConfig()
    boot_disk_config: BootDiskConfig = BootDiskConfig()
    vm_metadata_config: VMMetadataConfig = VMMetadataConfig()
    template_name: str = SI("${infrastructure.instance_group_creator.name}")
    project_id: str = SI("${infrastructure.project_id}")
    labels: dict[str, str] = field(default_factory=lambda: {"project": "my-project-name"})