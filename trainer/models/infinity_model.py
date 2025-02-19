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
class VlmModelConfig(BaseModelConfig):
    _target_: str = "transformers.LlavaOnevisionForConditionalGeneration.from_pretrained"
    pretrained_model_name_or_path: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    
@dataclass
class BscConfig:
    noise_apply_layers: int = 13
    noise_apply_requant: bool = True
    noise_apply_strength: float = 0.3
    apply_spatial_patchify: bool = False
    debug_bsc: bool = False
    
@dataclass
class VaeConfig(BaseModelConfig):
    vae_type: int = 16
    apply_spatial_patchify: bool = False
    vae_path: str = "/home/czh/.cache/huggingface/hub/models--FoundationVision--Infinity/snapshots/d4c15777e41bd36eb8eef5a854b018d19962b6d9/infinity_vae_d16.pth"

@dataclass
class DinoConfig(BaseModelConfig):
    _target_: str = "transformers.AutoModel.from_pretrained"
    pretrained_model_name_or_path: str = "facebook/dinov2-with-registers-large"
    
@dataclass
class InfinityConfig(BaseModelConfig):
    """
    To instantiate from the config, need to add another param: vae_local
    """
    # _target_: str = "Infinity.infinity.models.infinity.Infinity"
    text_channels: int = 2048
    text_maxlen: int = 1350 # NOTE: should change this whenever change the history image num
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
    d_dino: int = 1024

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
    dino_cfg: DinoConfig = field(default_factory=lambda:
        DinoConfig()
    )

class InfinityVlmModel(nn.Module):
    def __init__(self, cfg: InfinityVlmConfig):
        super().__init__()
        
        self.vae_cfg: VaeConfig = cfg.vae_cfg
        self.infinity_cfg: InfinityConfig = cfg.infinity_cfg
        self.vlm_cfg: VlmModelConfig = cfg.vlm_cfg
        self.bsc_cfg: BscConfig = cfg.bsc_cfg
        self.dino_cfg: DinoConfig = cfg.dino_cfg
        
        self.vae = load_visual_tokenizer(self.vae_cfg).to("cuda")
        
        self.infinity = Infinity(**self.infinity_cfg, vae_local=self.vae).to("cuda")
        self.bitwise_self_correction = BitwiseSelfCorrection(self.vae, self.bsc_cfg)
        self.vlm = instantiate(self.vlm_cfg).to("cuda")
        self.dino = instantiate(self.dino_cfg).to("cuda")
        
        # project dino and vlm inputs to 2048
        if self.infinity_cfg.d_vlm !=  self.infinity_cfg.text_channels:
            self.vlm_to_kv_compact = nn.Sequential(
                nn.Linear(self.infinity_cfg.d_vlm, self.infinity_cfg.text_channels),
                nn.GELU(approximate='tanh'),
                nn.Linear(self.infinity_cfg.text_channels, self.infinity_cfg.text_channels),
            )
        if self.infinity_cfg.d_dino !=  self.infinity_cfg.text_channels:
            self.dino_to_kv_compact = nn.Sequential(
                nn.Linear(self.infinity_cfg.d_dino, self.infinity_cfg.text_channels),
                nn.GELU(approximate='tanh'),
                nn.Linear(self.infinity_cfg.text_channels, self.infinity_cfg.text_channels),
            )

    def prepare_condition_input(self, vlm_inputs, dino_input):
        
        # vlm
        for k, v in vlm_inputs.items():
            vlm_inputs[k] = v.to("cuda")
        v_of_last_layer = self.vlm.generate(**vlm_inputs, max_new_tokens=200, do_sample=False, return_dict_in_generate=True)["past_key_values"][-1][1]
        v_of_last_layer = v_of_last_layer.reshape(v_of_last_layer.shape[0], v_of_last_layer.shape[2], -1) # turn from (b,h,len,dim) -> (b,len,h*dim)
        
        # dino
        for k, v in dino_input.items():
            dino_input[k] = v.to("cuda")
        dino_output = self.dino(**dino_input)
        dino_condition = dino_output.last_hidden_state
        
        # projection to 2048
        if self.infinity_cfg.d_vlm !=  self.infinity_cfg.text_channels:
            v_of_last_layer = self.vlm_to_kv_compact(v_of_last_layer)
        if self.infinity_cfg.d_dino !=  self.infinity_cfg.text_channels:
            dino_condition = self.dino_to_kv_compact(dino_condition)
        v_of_last_layer = torch.cat([v_of_last_layer, dino_condition], dim=1) # concat vlm and dino
         
        bsz = v_of_last_layer.shape[0]
        lens: List[int] = [v_of_last_layer.shape[1]] * bsz
        max_len: int = max(lens)
        cu_seqlens_k = torch.arange(0, bsz+1) * max_len
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)
        
        v_of_last_layer = v_of_last_layer.reshape(-1, v_of_last_layer.shape[-1])
        
        return (v_of_last_layer, lens, cu_seqlens_k.to("cuda"), max_len)
 
    def tokenize_image_with_vae(self, next_frame):
        if self.vae_cfg.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in self.infinity_cfg.scale_schedule]
        else:
            vae_scale_schedule = [(pt, ph, pw) for pt, ph, pw in self.infinity_cfg.scale_schedule]
            
        raw_features, _, _ = self.vae.encode_for_raw_features(next_frame, scale_schedule=vae_scale_schedule)
        x_BLC_wo_prefix, gt_ms_idx_Bl = self.bitwise_self_correction.flip_requant(vae_scale_schedule, next_frame, raw_features, "cuda") # x_BLC_wo_prefix: torch.Size([bs, 2*2+3*3+...+64*64, d or 4d])
        
        return x_BLC_wo_prefix, gt_ms_idx_Bl
            
    def forward(self, vlm_inputs=None, next_frame=None, dino_input=None):
        v_of_last_layer, lens, cu_seqlens_k, max_len = self.prepare_condition_input(vlm_inputs, dino_input) # torch.bfloat16, ...
        x_BLC_wo_prefix, gt_ms_idx_Bl = self.tokenize_image_with_vae(next_frame.to("cuda")) # troch.float32, List[torch.int32]
        
        # print(f"🚀🚀🚀🚀🚀🚀🚀 {v_of_last_layer.dtype=}, {x_BLC_wo_prefix.dtype=}")
                
        # remember 1. not to convert v_of_last_layer to float, and 2. add to("cuda") after cu_seqlens_k, and 3. not to convert x_BLC_wo_prefix to float
        logits_BLV = self.infinity(
            label_B_or_BLT=(v_of_last_layer, lens, cu_seqlens_k.to("cuda"), max_len),
            x_BLC_wo_prefix=x_BLC_wo_prefix,
            scale_schedule=[(pt, ph, pw) for pt, ph, pw in self.infinity_cfg.scale_schedule],
            cfg_infer=False,
        )

        return logits_BLV, gt_ms_idx_Bl
    
    def load_pretrained_infinity(self, pretrained_path: str):
        self.infinity.load_state_dict(torch.load(pretrained_path))

    def get_into_training_stage_1(self,):
        """
        Training stage 1:
        - vae: freezed
        - vlm: freezed
        - dino: freezed
        - infinity: trainable
        """
        self.vae.eval()
        self.dino.eval()
        self.vlm.eval()
        self.infinity.train()
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.vlm.parameters():
            param.requires_grad = False
        for param in self.dino.parameters():
            param.requires_grad = False
        for param in self.infinity.parameters():
            param.requires_grad = True

        for param in self.dino_to_kv_compact.parameters():
            param.requires_grad = True
        for param in self.vlm_to_kv_compact.parameters():
            param.requires_grad = True
        
        print("\n\n==========================================================================")
        print("🚀🚀🚀🚀🚀🚀🚀 PAY ATTENTION:\nYOU ARE ENTERING TRAINING STATE 1 (only linear prob part trainable)\n")
        print(f"num. infinity trainable params: {int(sum(p.numel() for p in self.infinity.parameters() if p.requires_grad) // 1e6)}M")
        print(f"num. VAE trainable params: {int(sum(p.numel() for p in self.vae.parameters() if p.requires_grad) // 1e6)}M")
        print(f"num. VLM trainable params: {int(sum(p.numel() for p in self.vlm.parameters() if p.requires_grad) // 1e6)}M")
        print(f"num. DINO trainable params: {int(sum(p.numel() for p in self.dino.parameters() if p.requires_grad) // 1e6)}M")
        print(f"num. Linear Prob trainable params: {int(sum(p.numel() for p in self.dino_to_kv_compact.parameters() if p.requires_grad) // 1e6)} + {int(sum(p.numel() for p in self.vlm_to_kv_compact.parameters() if p.requires_grad) // 1e6)}M")
        print("==========================================================================\n\n")
                
    def get_into_training_stage_2(self,):
        """
        Training stage 2:
        - vae: freezed
        - dino: freezed
        - vlm: trainable
        - infinity: trainable
        """
        self.vae.eval()
        self.dino.eval()
        self.vlm.train()
        self.infinity.train()
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.dino.parameters():
            param.requires_grad = False
        for param in self.vlm.parameters():
            param.requires_grad = True
        for param in self.infinity.parameters():
            param.requires_grad = True
        for param in self.dino_to_kv_compact.parameters():
            param.requires_grad = True
        for param in self.vlm_to_kv_compact.parameters():
            param.requires_grad = True

        print("\n\n==========================================================================")
        print("🚀🚀🚀🚀🚀🚀🚀 PAY ATTENTION:\nYOU ARE ENTERING TRAINING STATE 1 (only linear prob part trainable)\n")
        print(f"num. infinity trainable params: {int(sum(p.numel() for p in self.infinity.parameters() if p.requires_grad) // 1e6)}M")
        print(f"num. VAE trainable params: {int(sum(p.numel() for p in self.vae.parameters() if p.requires_grad) // 1e6)}M")
        print(f"num. VLM trainable params: {int(sum(p.numel() for p in self.vlm.parameters() if p.requires_grad) // 1e6)}M")
        print(f"num. DINO trainable params: {int(sum(p.numel() for p in self.dino.parameters() if p.requires_grad) // 1e6)}M")
        print(f"num. Linear Prob trainable params: {int(sum(p.numel() for p in self.dino_to_kv_compact.parameters() if p.requires_grad) // 1e6)} + {int(sum(p.numel() for p in self.vlm_to_kv_compact.parameters() if p.requires_grad) // 1e6)}M")
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