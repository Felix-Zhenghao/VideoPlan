import sys
import os
sys.path.append(f"{os.getcwd()}/video_gen/")
sys.path.append(f"{os.getcwd()}/video_gen/VideoPlan/")
sys.path.append(f"{os.getcwd()}/video_gen/Infinity/")

import collections
import random
from dataclasses import dataclass

import torch
from PIL import Image
from hydra.utils import instantiate
from accelerate.logging import get_logger
from accelerate.utils import LoggerType
from omegaconf import II
from transformers import AutoTokenizer

from config_util import instantiate_with_cfg
from trainer.accelerators.base_accelerator import BaseAccelerator
from trainer.tasks.base_task import BaseTaskConfig, BaseTask

logger = get_logger(__name__)


@dataclass
class VlaTaskConfig(BaseTaskConfig):
    _target_: str = "trainer.tasks.video_gen_task.VlaTask"
    # pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")
    # label_0_column_name: str = II("dataset.label_0_column_name")
    # label_1_column_name: str = II("dataset.label_1_column_name")

    # input_ids_column_name: str = II("dataset.input_ids_column_name")
    # pixels_0_column_name: str = II("dataset.pixels_0_column_name")
    # pixels_1_column_name: str = II("dataset.pixels_1_column_name")


def numpy_to_pil(images):
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


class VlaTask(BaseTask):
    def __init__(self, cfg: VlaTaskConfig, accelerator: BaseAccelerator):
        super().__init__(cfg, accelerator)
        # self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path)
        self.cfg = cfg

    def train_step(self, model, criterion, batch):
        loss = criterion(model, batch)
        return loss

    @torch.no_grad()
    def evaluate(self, model, criterion, dataloader, save_dir, valid_episodes_length):
        return

if __name__ == "__main__":
    from VideoPlan.trainer.accelerators import DeepSpeedAcceleratorConfig
    
    cfg = DeepSpeedAcceleratorConfig()
    
    import omegaconf
    cfg = omegaconf.OmegaConf.create(cfg)
    accelerator = instantiate_with_cfg(cfg=cfg)
    
    import IPython; IPython.embed()
    