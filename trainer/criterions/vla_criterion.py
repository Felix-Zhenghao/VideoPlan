import sys
import os
sys.path.append(f"{os.getcwd()}/video_gen/")

from typing import Dict
from dataclasses import dataclass

import torch
import numpy as np
import torch.nn as nn
from omegaconf import II
from torch.nn.modules.loss import _Loss

from VideoPlan.trainer.models.qwen_vla_model import Vla


@dataclass
class VlaCriterionConfig:
    _target_: str = "VideoPlan.trainer.criterions.vla_criterion.VlaCriterion"
    is_distributed: bool = False
    


class VlaCriterion(_Loss):
    def __init__(self, cfg: VlaCriterionConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, model: Vla, batch: Dict):
        pred_action = model(batch)
        action = batch["actions"].to(torch.bfloat16)
        loss = nn.MSELoss()(pred_action, action)
        return loss
