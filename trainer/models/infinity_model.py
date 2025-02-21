import sys
import os
sys.path.append(f"{os.getcwd()}/video_gen/")
sys.path.append(f"{os.getcwd()}/video_gen/VideoPlan/")
sys.path.append(f"{os.getcwd()}/video_gen/Infinity/")

from typing import Any, Optional, List
from dataclasses import dataclass, field
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

import torch
from torch import nn
from hydra.utils import instantiate

from Infinity.infinity.models.infinity import Infinity
from Infinity.infinity.models.bitwise_self_correction import BitwiseSelfCorrection
from Infinity.tools.run_infinity import load_visual_tokenizer
# from VideoPlan.trainer.models.base_model import BaseModelConfig # prevent circular
from config_util import instantiate_with_cfg

@dataclass
class BaseModelConfig:
    pass

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

##########
# download VAR - Infinity
##########
@dataclass
class VlmConfig(BaseModelConfig):
    _target_: str = "transformers.PaliGemmaForConditionalGeneration.from_pretrained"
    pretrained_model_name_or_path: str = "google/paligemma2-3b-pt-224"

@dataclass
class InfinityVlmConfig(BaseModelConfig):
    _target_: str = "VideoPlan.trainer.models.infinity_model.InfinityVlmModel"
    vlm_cfg: VlmConfig = field(default_factory=lambda:
        VlmConfig()
    )

class InfinityVlmModel(nn.Module):
    def __init__(self, cfg: InfinityVlmConfig):
        super().__init__()
        self.param1 = torch.nn.Linear(256,1)
        self.paligemma = instantiate(cfg.vlm_cfg).to("cuda")
    
    @property
    def logit_scale(self):
        pass

    def save(self, path):
        pass

if __name__ == "__main__":
    # vlm_cfg = VlmModelConfig()
    # infinity_cfg = InfinityConfig()
    # vae_cfg = VaeConfig()
    # bsc_cfg = BscConfig()
    
    cfg = InfinityVlmConfig()
    # use omegacfg to deal with all cfgs in cfg
    import omegaconf
    cfg = omegaconf.OmegaConf.create(cfg.vlm_cfg)
    
    
    model = instantiate_with_cfg(cfg=cfg)
    model.load_pretrained_infinity("/home/czh/.cache/huggingface/hub/models--FoundationVision--Infinity/snapshots/d4c15777e41bd36eb8eef5a854b018d19962b6d9/infinity_125M_256x256.pth")
    
    from VideoPlan.trainer.datasetss.libero_lerobot_dataset import LiberoLerobotDatasetConfig
    datacfg = LiberoLerobotDatasetConfig()
    dataset = instantiate_with_cfg(cfg=datacfg, split=datacfg.train_split_name)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=8,
        collate_fn=dataset.collate_fn,
        num_workers=0
    )
    
    # criterion
    from VideoPlan.trainer.criterions.infinity_criterion import InfinityVlmCriterionConfig
    criterion_cfg = InfinityVlmCriterionConfig()
    criterion = instantiate_with_cfg(cfg=criterion_cfg)
    
    # forward
    for batch in dataloader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(vlm_inputs=batch["vlm_inputs"], next_frame=batch["future_img"])
            import IPython; IPython.embed();