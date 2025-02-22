import sys
import os
sys.path.append(f"{os.getcwd()}/video_gen/")
sys.path.append(f"{os.getcwd()}/video_gen/VideoPlan/")
sys.path.append(f"{os.getcwd()}/video_gen/Infinity/")

import gc
from typing import Any, Optional, List
from dataclasses import dataclass, field
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

import torch
from torch import nn
from PIL.Image import Image as PILImage
from hydra.utils import instantiate

from Infinity.infinity.models.infinity import Infinity
from Infinity.infinity.models.bitwise_self_correction import BitwiseSelfCorrection
from Infinity.tools.run_infinity import load_visual_tokenizer
# from VideoPlan.trainer.models.base_model import BaseModelConfig # prevent circular
from config_util import instantiate_with_cfg, _locate

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
class QwenVlmConfig(BaseModelConfig):
    _target_: str = "vlap.models.QwenVLM"
    model_id: str = "qwen25-dinosiglip-224px+0_5b"
    vlm_embed_dim: int = 128
    vlm_hidden_layers: int = 24
    pretrained_checkpoint: Optional[str] = "/data/czh/qwen25-dinosiglip-224px+0_5b+stage-finetune+x42/checkpoints/latest-checkpoint.pt"

@dataclass
class ActionHeadConfig(BaseModelConfig):
    _target_: str = "vlap.models.vla.ActionHead"
    action_dim: int = 7
    proprio_dim: int = 9
    hidden_size: int = 128
    num_heads: int = 8
    dropout: float = 0.1
    gated: bool = False

@dataclass
class VlaConfig(BaseModelConfig):
    _target_: str = "VideoPlan.trainer.models.qwen_vla_model.Vla"
    vla_id: str = "qwen-siglip-224px-libero-spatial"
    algorithm: str = "flow-matching"
    llm_cfg: LlmModelConfig = field(default_factory=LlmModelConfig)
    dino_siglip_cfg: DinoSigLipModelConfig = field(default_factory=DinoSigLipModelConfig)
    vlm_cfg: QwenVlmConfig = field(default_factory=QwenVlmConfig)
    action_head_cfg: ActionHeadConfig = field(default_factory=ActionHeadConfig)

class Vla(nn.Module):
    def __init__(self, cfg: VlaConfig):
        super().__init__()
        
        self.vla_cfg = cfg
        self.llm_cfg = cfg.llm_cfg
        self.dino_siglip_cfg = cfg.dino_siglip_cfg
        self.vlm_cfg = cfg.vlm_cfg
        self.action_head_cfg = cfg.action_head_cfg
        
        # instantiate all models
        self.llm = instantiate(cfg=self.llm_cfg)
        self.dino_siglip = instantiate(cfg=self.dino_siglip_cfg)
        self.vlm = _locate(self.vlm_cfg._target_)(model_id=self.vlm_cfg.model_id, llm_backbone=self.llm, vison_backbone=self.dino_siglip)
        self.action_head = instantiate_with_cfg(cfg=self.action_head_cfg, num_layers=self.vlm_cfg.vlm_hidden_layers, llm_emb_dim=self.vlm_cfg.vlm_embed_dim)
        
    def get_dino_siglip_image_transform(self):
        return self.dino_siglip.get_image_transform()
    
    def get_llm_tokenizer(self):
        return self.llm.get_tokenizer()

    def forward(self, batch):
        out = self.vlm(
            input_ids=batch["input_ids"],
            pixel_values={"dino":batch["dino"].to("cuda"),"siglip":batch["siglip"].to("cuda")},
            use_cache=True, # let it return past_key_values
        )
        
        vlm_v_layers: List[torch.Tensor] = out['past_key_values'].value_cache # [bsz, num_kv_heads, seq_len, kv_head_dim]
        vlm_k_layers: List[torch.Tensor] = out['past_key_values'].key_cache # [bsz, num_kv_heads, seq_len, kv_head_dim]
        
        # free memory
        del out
        gc.collect()
        
        transform_vlm_layers = lambda vlm_v_layers: (
            torch.stack(vlm_v_layers, dim=0)  # [num_layers, bsz, num_kv_heads, seq_len, kv_head_dim]
            .permute(0, 1, 3, 2, 4)  # [num_layers, bsz, seq_len, num_kv_heads, kv_head_dim]
            .reshape(vlm_v_layers.size(0), vlm_v_layers.size(1), vlm_v_layers.size(3), -1)
            # reshape to [num_layers, bsz, seq_len, num_kv_heads*kv_head_dim]
        )
        
        vlm_v_layers = transform_vlm_layers(vlm_v_layers)
        vlm_k_layers = transform_vlm_layers(vlm_k_layers)
        
        if self.vla_cfg.algorithm == "bc":
            predicted_actions = self.action_head(
                proprio=batch["proprio"],
                noisy_action=torch.zero([1, batch["action"].shape[1], self.action_head_cfg.action_dim]).to("cuda"),
                t=torch.zeros(1).to("cuda"),
                llm_key=vlm_k_layers,
                llm_value=vlm_v_layers
            )
        elif self.vla_cfg.algorithm == "flow-matching":
            raise NotImplementedError("flow-matching algorithm is not implemented yet")
        
        return predicted_actions
    
    def load_pretrained_vla(self, pretrained_path: str):
        pass

    def get_into_training_stage_1(self,):
        """
        Training stage 1:
        
        - dino_siglip: not trainable
        - llm: not trainable
        - action_head: trainable
        """
        
        self.llm.eval()
        self.dino_siglip.eval()
        self.vlm.eval()
        self.action_head.train()
        
        for param in self.action_head.parameters():
            param.requires_grad = True
        for param in self.llm.parameters():
            param.requires_grad = False
        for param in self.dino_siglip.parameters():
            param.requires_grad = False
        for param in self.vlm.parameters():
            param.requires_grad = False

        print("\n\n==========================================================================")
        print("🚀🚀🚀🚀🚀🚀🚀 PAY ATTENTION:\nYOU ARE ENTERING TRAINING STATE 1 (only linear prob part trainable)\n")
        print("[DINO_SIGLIP] Total num of trainable parameters: ", sum(p.numel() for p in self.dino_siglip.parameters() if p.requires_grad))
        print("[LLM] Total num of trainable parameters: ", sum(p.numel() for p in self.llm.parameters() if p.requires_grad))
        print("[VLM] Total num of trainable parameters: ", sum(p.numel() for p in self.vlm.parameters() if p.requires_grad))
        print("[Action Head] Total num of trainable parameters: ", sum(p.numel() for p in self.action_head.parameters() if p.requires_grad))
        print("[VLA] Total num of trainable parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))
        print("==========================================================================\n\n")
                
    def get_into_training_stage_2(self,):
        """
        Training stage 2:
        
        - llm: trainable
        - dino_siglip: trainable
        - vlm: trainable
        - action_head: trainable
        """
        
        self.llm.train()
        self.dino_siglip.train()
        self.vlm.train()
        self.action_head.train()
        
        for param in self.action_head.parameters():
            param.requires_grad = True
        for param in self.llm.parameters():
            param.requires_grad = True
        for param in self.dino_siglip.parameters():
            param.requires_grad = True
        for param in self.vlm.parameters():
            param.requires_grad = True

        print("\n\n==========================================================================")
        print("🚀🚀🚀🚀🚀🚀🚀 PAY ATTENTION:\nYOU ARE ENTERING TRAINING STATE 2 (whole vlm and infinity trainable)\n")
        print("[DINO_SIGLIP] Total num of trainable parameters: ", sum(p.numel() for p in self.dino_siglip.parameters() if p.requires_grad))
        print("[LLM] Total num of trainable parameters: ", sum(p.numel() for p in self.llm.parameters() if p.requires_grad))
        print("[VLM] Total num of trainable parameters: ", sum(p.numel() for p in self.vlm.parameters() if p.requires_grad))
        print("[Action Head] Total num of trainable parameters: ", sum(p.numel() for p in self.action_head.parameters() if p.requires_grad))
        print("[VLA] Total num of trainable parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))
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
    
