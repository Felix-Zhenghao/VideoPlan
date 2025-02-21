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

from VideoPlan.trainer.models.infinity_model import InfinityVlmModel


@dataclass
class InfinityVlmCriterionConfig:
    _target_: str = "VideoPlan.trainer.criterions.infinity_criterion.InfintyVlmCriterion"
    is_distributed: bool = False
    
    use_bit_label: bool = True
    bitloss_type: str = "mean"
    reweight_loss_by_scale: bool = True
    ls: float = 0.0


class InfintyVlmCriterion(_Loss):
    def __init__(self, cfg: InfinityVlmCriterionConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, model: InfinityVlmModel, batch: Dict):
        
        # prepare constants for loss calculation
        # forward pass to get logits and ground truth
        outputs = model.paligemma(**batch["vlm_inputs"])

        input = outputs["logits"][:,:,:1]

        loss = torch.nn.functional.mse_loss(input, torch.zeros_like(input))

        return loss
