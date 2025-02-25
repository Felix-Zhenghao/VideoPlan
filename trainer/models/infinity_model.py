import sys
import os
sys.path.append(f"{os.getcwd()}/video_gen/")
sys.path.append(f"{os.getcwd()}/video_gen/VideoPlan/")
sys.path.append(f"{os.getcwd()}/video_gen/Infinity/")

from typing import Any, Optional, List
from dataclasses import dataclass, field
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

import torch
import random
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
class VlmModelConfig(BaseModelConfig):
    _target_: str = "transformers.LlavaOnevisionForConditionalGeneration.from_pretrained"
    pretrained_model_name_or_path: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    
@dataclass
class BscConfig:
    noise_apply_layers: int = -1
    noise_apply_requant: bool = True
    noise_apply_strength: float = 0.0
    apply_spatial_patchify: bool = False
    debug_bsc: bool = True
    
@dataclass
class VaeConfig(BaseModelConfig):
    vae_type: int = 16
    apply_spatial_patchify: bool = False
    vae_path: str = "/data2/czhenghao/.cache/huggingface/hub/models--FoundationVision--Infinity/snapshots/6577e6454575816928a2a8477906c84a49356b9a/infinity_vae_d16.pth"

    
@dataclass
class InfinityConfig(BaseModelConfig):
    """
    To instantiate from the config, need to add another param: vae_local
    """
    # _target_: str = "Infinity.infinity.models.infinity.Infinity"
    text_channels: int = 2048
    text_maxlen: int = 1050 # NOTE: should change this whenever change the history image num
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_eps: float = 1e-6
    rms_norm: bool = False
    shared_aln: bool = True
    head_aln: bool = True
    cond_drop_rate: float = 0.1
    rand_uncond: bool = False
    cross_attn_layer_scale: float = -1
    nm0: bool = False
    tau: float = 1.0
    cos_attn: bool = True
    swiglu: bool = False
    raw_scale_schedule: Optional[Any] = None
    head_depth: int = 1
    top_p: float = 0.0
    top_k: float = 0.0
    customized_flash_attn: bool = True
    fused_mlp: bool = False
    fused_norm: bool = True
    block_chunks: int = 4
    checkpointing: str = "full-block"
    pad_to_multiplier: int = 128
    use_flex_attn: bool = False
    batch_size: int = 1
    add_lvl_embeding_only_first_block: int = 1
    use_bit_label: int = 1
    rope2d_each_sa_layer: int = 1
    rope2d_normalized_by_hw: int = 2
    pn: str = "0.06M"
    train_h_div_w_list: float = 1.000
    video_frames: int = 1
    always_training_scales: int = 100
    apply_spatial_patchify: bool = False
    inference_mode: bool = False
    scale_schedule: List[List[int]] = field(default_factory=lambda: 
        [[1, 1, 1], [1, 2, 2], [1, 4, 4], [1, 6, 6], [1, 8, 8], [1, 12, 12], [1, 16, 16]]
    )
    d_vlm: int = 128

@dataclass
class InfinityVlmConfig(BaseModelConfig):
    _target_: str = "VideoPlan.trainer.models.infinity_model.InfinityVlmModel"
    vlm_cfg: VlmModelConfig = field(default_factory=lambda:
        VlmModelConfig()
    )
    infinity_cfg: InfinityConfig = field(default_factory=lambda:
        InfinityConfig()
    )
    vae_cfg: VaeConfig = field(default_factory=lambda:
        VaeConfig()
    )
    bsc_cfg: BscConfig = field(default_factory=lambda:
        BscConfig()
    )
    

class InfinityVlmModel(nn.Module):
    def __init__(self, cfg: InfinityVlmConfig):
        super().__init__()
        
        self.vae_cfg: VaeConfig = cfg.vae_cfg
        self.infinity_cfg: InfinityConfig = cfg.infinity_cfg
        self.vlm_cfg: VlmModelConfig = cfg.vlm_cfg
        self.bsc_cfg: BscConfig = cfg.bsc_cfg
        
        self.vae = load_visual_tokenizer(self.vae_cfg).to("cuda")
        self.bitwise_self_correction = BitwiseSelfCorrection(self.vae, self.bsc_cfg)
        
        self.vae.train()
        for param in self.vae.parameters():
            param.requires_grad = True

    def forward(self, batch, should_save=False):
        
        image = batch["image"].to("cuda")
        raw_features, _, _ = self.vae.encode_for_raw_features(image, scale_schedule=[(1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 6, 6), (1, 8, 8), (1, 12, 12), (1, 16, 16)])
        _, _, _, loss = self.bitwise_self_correction.flip_requant([(1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 6, 6), (1, 8, 8), (1, 12, 12), (1, 16, 16)], image, raw_features, "cuda", save_path = f"/data2/czhenghao/infinity_125M/vae_check_finetuned/{random.randint(0,100000)}.jpg", should_save=should_save)

        return loss

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
    cfg = omegaconf.OmegaConf.create(cfg)
    
    
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