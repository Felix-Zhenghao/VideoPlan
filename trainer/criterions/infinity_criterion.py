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
        scale_schedule = model.infinity.scale_schedule
        V = model.vae.vocab_size
        bsz = batch["future_img"].shape[0]
        train_loss = nn.CrossEntropyLoss(label_smoothing=self.cfg.ls, reduction='none')
        
        # forward pass to get logits and ground truth
        logits_BLV, gt_ms_idx_Bl = model(
            vlm_inputs=batch["vlm_inputs"],
            next_frame=batch["future_img"],
        )
        
        # loss calculation
        seq_len = logits_BLV.shape[1]
        seq_len_each = [idx_Bl.shape[1] for idx_Bl in gt_ms_idx_Bl]
        training_seq_len = np.array(scale_schedule).prod(axis=1).sum()
        gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)[:,:training_seq_len].contiguous().type(torch.long) # [bs, 1*1+...+64*64, 16] or [bs, 1*1+...+64*64]
        
        if self.cfg.use_bit_label:
            tmp_bs, tmp_seq_len, tmp_channel = logits_BLV.shape
            loss = train_loss(logits_BLV.reshape(tmp_bs, tmp_seq_len, -1, 2).permute(0,3,1,2), gt_BL)
            if self.cfg.bitloss_type == 'mean':
                loss = loss.mean(dim=-1)
            elif self.cfg.bitloss_type == 'sum':
                loss = loss.sum(dim=-1)
            else:
                raise NotImplementedError(f'{self.cfg.bitloss_type=}')
        else:
            loss = train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(bsz, -1)
            
        if self.cfg.reweight_loss_by_scale:
            lw = []
            last_scale_area = np.sqrt(np.array(scale_schedule[-1]).prod())
            for (pt, ph, pw) in scale_schedule:
                this_scale_area = np.sqrt(pt * ph * pw)
                lw.extend([last_scale_area / this_scale_area for _ in range(pt * ph * pw)])
            lw = torch.tensor(lw, device=loss.device)[None, ...]
            lw = lw / lw.sum()
        else:
            lw = 1. / seq_len

        loss = loss.mul(lw).sum(dim=-1).mean()
        return loss
