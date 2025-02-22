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

@dataclass
class LlmModelConfig(BaseModelConfig):
    _target_: str = "vlap.models.Qwen25LLMBackbone"
    llm_backbone_id: str = "qwen25-0_5b"
    hf_token: Optional[str] = None
    inference_mode: bool = False
    use_flash_attention_2: bool = True

@dataclass
class DinoSigLipModelConfig(BaseModelConfig):
    _target_: str = "vlap.models.DinoSigLIPViTBackbone"
    vision_backbone_id: str = "dinosiglip-vit-so-224px"
    image_resize_strategy: str = "resize-naive"
    default_image_size: int = 256
    image_sequence_len: int = 4 # TODO: other place should adjust this according to the lerobot dataset

@dataclass
class ActionHeadConfig(BaseModelConfig):
    pass

    

@dataclass
class VlaConfig(BaseModelConfig):
    _target_: str = "VideoPlan.trainer.models.qwen_vla_model.Vla"
    llm_cfg: LlmModelConfig = field(default_factory=LlmModelConfig)
    dino_siglip_cfg: DinoSigLipModelConfig = field(default_factory=DinoSigLipModelConfig)
    action_head_cfg: ActionHeadConfig = field(default_factory=ActionHeadConfig)

class Vla(nn.Module):
    def __init__(self, cfg: VlaConfig):
        super().__init__()
        
        self.llm_cfg = cfg.llm_cfg
        self.dino_siglip_cfg = cfg.dino_siglip_cfg
        self.action_head_cfg = cfg.action_head_cfg
        
        self.llm = instantiate(cfg=self.llm_cfg)
        self.dino_siglip = instantiate(cfg=self.dino_siglip_cfg)
        self.action_head = instantiate(cfg=self.action_head_cfg)

    def prepare_dino_siglip_input(self, vlm_inputs):
        """
        Use image transformation needed for dino_siglip model to transform the image history.
        """
        pass
    
    def tokenize_proprio_and_action(self, next_frame):
        """
        Tokenize action and proprio for action head input.
        """
            
    def forward(self, vlm_inputs=None, next_frame=None):
        pass
    
    def load_pretrained_vla(self, pretrained_path: str):
        pass

    def get_into_training_stage_1(self,):
        """
        Training stage 1:
        ```
        """

        print("\n\n==========================================================================")
        print("🚀🚀🚀🚀🚀🚀🚀 PAY ATTENTION:\nYOU ARE ENTERING TRAINING STATE 1 (only linear prob part trainable)\n")
        print("==========================================================================\n\n")
                
    def get_into_training_stage_2(self,):
        """
        Training stage 2:
        """

        print("\n\n==========================================================================")
        print("🚀🚀🚀🚀🚀🚀🚀 PAY ATTENTION:\nYOU ARE ENTERING TRAINING STATE 2 (whole vlm and infinity trainable)\n")
        print("==========================================================================\n\n")
    
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
    
    cfg = VlaConfig()
    # use omegacfg to deal with all cfgs in cfg
    import omegaconf
    cfg = omegaconf.OmegaConf.create(cfg)
    
    
    model = instantiate_with_cfg(cfg=cfg)
    model.load_pretrained_infinity("/home/czh/.cache/huggingface/hub/models--FoundationVision--Infinity/snapshots/d4c15777e41bd36eb8eef5a854b018d19962b6d9/infinity_125M_256x256.pth")
    
