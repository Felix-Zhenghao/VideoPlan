from dataclasses import dataclass, field
from typing import List, Any, Dict

from omegaconf import DictConfig, MISSING

import trainer.accelerators
import trainer.tasks
from trainer.models import * # BaseModelConfig and InfinityVlmConfig
import trainer.criterions
import trainer.datasetss
import trainer.optimizers
import trainer.lr_schedulers
from trainer.accelerators.base_accelerator import BaseAcceleratorConfig
from trainer.tasks.base_task import BaseTaskConfig

defaults = [
    {"accelerator": "deepspeed"},
    {"task": "infinity_vlm"},
    {"model": "infinity_vlm"},
    {"criterion": "infinity_vlm"},
    {"dataset": "libero"},
    {"optimizer": "dummy"},
    {"lr_scheduler": "dummy"},
]


@dataclass
class DebugConfig:
    activate: bool = False
    port: int = 5900


@dataclass
class TrainerConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    accelerator: BaseAcceleratorConfig = MISSING
    task: BaseTaskConfig = MISSING
    model: BaseModelConfig = MISSING
    criterion: Any = MISSING
    dataset: Any = MISSING
    optimizer: Any = MISSING
    lr_scheduler: Any = MISSING
    debug: DebugConfig = field(default_factory=lambda: 
        DebugConfig()
    )
    output_dir: str = "outputs_experiment_reg0.1"
