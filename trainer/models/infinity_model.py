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
from hydra.utils import instantiate
from transformers import Gemma2ForCausalLM
from transformers.cache_utils import HybridCache

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
class ActionHeadConfig:
    _target_: str = "transformers.Gemma2Config"
    architectures: List[str] = field(default_factory=lambda: ["Gemma2ForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_logit_softcapping: float = 50.0
    bos_token_id: int = 2049 # NOTE
    cache_implementation: str = "hybrid"
    eos_token_id: int = 2048 # NOTE
    final_logit_softcapping: float = 30.0
    head_dim: int = 64 # NOTE
    hidden_act: str = "gelu_pytorch_tanh"
    hidden_activation: str = "gelu_pytorch_tanh"
    hidden_size: int = 896 # NOTE
    initializer_range: float = 0.02
    intermediate_size: int = 3584 # NOTE: mlp_ratio = 4
    max_position_embeddings: int = 8192
    model_type: str = "gemma2"
    num_attention_heads: int = 14 # NOTE
    num_hidden_layers: int = 12 # TODO: make it compatible with Qwen with only 12 hidden layers
    num_key_value_heads: int = 2 # NOTE
    pad_token_id: int = 2048 # NOTE
    query_pre_attn_scalar: int = 256
    rms_norm_eps: float = 1e-06
    rope_theta: float = 10000.0
    sliding_window: int = 4096
    torch_dtype: str = "bfloat16" # NOTE
    transformers_version: str = "4.42.4"
    use_cache: bool = True
    vocab_size: int = 2051 # NOTE: special tokens: bos, eos(pad), action chunk split signal
    _attn_implementation: str = "eager"
    
@dataclass
class VlaHybridCacheConfig:
    max_extra_tokens: int = 200
    device: str = "cuda"

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
class QwenVlmInfinityHeadGemmaActionHeadConfig(BaseModelConfig):
    _target_: str = "VideoPlan.trainer.models.infinity_model.QwenVlmInfinityHeadGemmaActionHeadBase"
    layer_id_of_vlm_kv_used: List[int] = field(default_factory=lambda: 
        [1,3,5,7,9,11,13,15,17,19,21,23]
    )
    vlm_cfg: VlmModelConfig = field(default_factory=lambda:
        VlmModelConfig()
    )
    infinity_cfg: InfinityConfig = field(default_factory=lambda:
        InfinityConfig()
    )
    hybrid_cache_cfg: VlaHybridCacheConfig = field(default_factory=lambda:
        VlaHybridCacheConfig()
    )
    action_head_cfg: ActionHeadConfig = field(default_factory=lambda:
        ActionHeadConfig()
    )
    vae_cfg: VaeConfig = field(default_factory=lambda:
        VaeConfig()
    )
    bsc_cfg: BscConfig = field(default_factory=lambda:
        BscConfig()
    )


@dataclass
class QwenVlmInfinityConfig(QwenVlmInfinityHeadGemmaActionHeadConfig):
    _target_: str = "VideoPlan.trainer.models.infinity_model.QwenVlmInfinityHead"


@dataclass
class QwenVlmGemmaActionHeadConfig(QwenVlmInfinityHeadGemmaActionHeadConfig):
    _target_: str = "VideoPlan.trainer.models.infinity_model.QwenVlmGemmaActionHead"



class QwenVlmInfinityHeadGemmaActionHeadBase(nn.Module):
    def __init__(self, cfg: QwenVlmInfinityHeadGemmaActionHeadConfig):
        super().__init__()
        
        self.vae_cfg: VaeConfig = cfg.vae_cfg
        self.infinity_cfg: InfinityConfig = cfg.infinity_cfg
        self.vlm_cfg: VlmModelConfig = cfg.vlm_cfg
        self.bsc_cfg: BscConfig = cfg.bsc_cfg
        self.action_head_cfg = instantiate(cfg.action_head_cfg)
        self.hybrid_cache_cfg = cfg.hybrid_cache_cfg
        self.layer_id_of_vlm_kv_used = cfg.layer_id_of_vlm_kv_used
        
        # only instantiate vlm when initialize the model
        # TODO: modify training stages to add model loading logics
        self.vlm = instantiate(self.vlm_cfg)
        self.training_stage = 0
        
        self.vae, self.infinity, self.bitwise_self_correction, self.action_head = None, None, None, None

    def load_infinity(self):

        if self.vae is None and self.infinity is None and self.bitwise_self_correction is None:
            self.vae = load_visual_tokenizer(self.vae_cfg)
            self.infinity = Infinity(**self.infinity_cfg, vae_local=self.vae)
            self.bitwise_self_correction = BitwiseSelfCorrection(self.vae, self.bsc_cfg)
        
    def load_action_head(self,):
        
        if self.action_head is None:
            self.action_head = Gemma2ForCausalLM(self.action_head_cfg)










class QwenVlmInfinityHead(QwenVlmInfinityHeadGemmaActionHeadBase):
    
    def __init__(self, cfg: QwenVlmInfinityConfig):
        super().__init__(cfg)
        self.load_infinity()

    def prepare_infinity_condition_input(self, vlm_inputs):
        for k, v in vlm_inputs.items():
            vlm_inputs[k] = v.to("cuda")
        v_of_last_layer = self.vlm.generate(**vlm_inputs, max_new_tokens=200, do_sample=False, return_dict_in_generate=True)["past_key_values"][-1][1]
        v_of_last_layer = v_of_last_layer.reshape(v_of_last_layer.shape[0], v_of_last_layer.shape[2], -1) # turn from (b,h,len,dim) -> (b,len,h*dim)
        
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

                   
    def forward(self, vlm_inputs=None, next_frame=None,):
        v_of_last_layer, lens, cu_seqlens_k, max_len = self.prepare_infinity_condition_input(vlm_inputs) # torch.bfloat16, ...
        x_BLC_wo_prefix, gt_ms_idx_Bl = self.tokenize_image_with_vae(next_frame.to("cuda")) # troch.float32, List[torch.int32]
                        
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
        - infinity: mostly freezed except for:
        ```
            - (vlm_to_kv_compact): Sequential(
                (0): Linear(in_features=128, out_features=2048, bias=True)
                (1): GELU(approximate='tanh')
                (2): Linear(in_features=2048, out_features=2048, bias=True)
            )
            - (cfg_uncond)
        ```
        """
        self.action_head = None
        self.load_infinity()
        self.training_stage = 1
        
        self.vae.eval()
        self.vlm.eval()
        self.infinity.train()
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.vlm.parameters():
            param.requires_grad = False
        for name, param in self.infinity.named_parameters():
            if "vlm_to_kv_compact" in name or "cfg_uncond" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        print("\n\n==========================================================================")
        print("🚀🚀🚀🚀🚀🚀🚀 PAY ATTENTION:\nYOU ARE ENTERING TRAINING STATE 1 (only linear prob part trainable)\n")
        print(f"num. infinity trainable params: {int(sum(p.numel() for p in self.infinity.parameters() if p.requires_grad) // 1e6)}M")
        print(f"num. VAE trainable params: {int(sum(p.numel() for p in self.vae.parameters() if p.requires_grad) // 1e6)}M")
        print(f"num. VLM trainable params: {int(sum(p.numel() for p in self.vlm.parameters() if p.requires_grad) // 1e6)}M")
        print("==========================================================================\n\n")
                
    def get_into_training_stage_2(self,):
        """
        Training stage 2:
        - vae: freezed
        - vlm: trainable
        - infinity: trainable
        """
        self.action_head = None
        self.load_infinity()
        self.training_stage = 2
        
        self.vae.eval()
        self.vlm.train()
        self.infinity.train()
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.vlm.parameters():
            param.requires_grad = True
        for param in self.infinity.parameters():
            param.requires_grad = True

        print("\n\n==========================================================================")
        print("🚀🚀🚀🚀🚀🚀🚀 PAY ATTENTION:\nYOU ARE ENTERING TRAINING STATE 2 (whole vlm and infinity trainable)\n")
        print(f"num. infinity trainable params: {int(sum(p.numel() for p in self.infinity.parameters() if p.requires_grad) // 1e6)}M")
        print(f"num. VAE trainable params: {int(sum(p.numel() for p in self.vae.parameters() if p.requires_grad) // 1e6)}M")
        print(f"num. VLM trainable params: {int(sum(p.numel() for p in self.vlm.parameters() if p.requires_grad) // 1e6)}M")
        print("==========================================================================\n\n")  





class QwenVlmGemmaActionHead(QwenVlmInfinityHeadGemmaActionHeadBase):
    
    def __init__(self, cfg: QwenVlmGemmaActionHeadConfig):
        super().__init__(cfg)
        self.load_action_head()
        
    def transform_dynamic_cache_to_hybrid_cache(self, dynamic_cache, batch_size):
        """
        QwenVlm uses dynamic cache, while Gemma-based (like Gemma2) model uses hybrid cache.
        This function is to transform dynamic cache (kv cache of vlm) to hybrid cache so gemma tokens can attend to kv of vlm.
        """
        past_len = dynamic_cache.key_cache[0].shape[-2]
        hybrid_cache: HybridCache = HybridCache(
            device=self.hybrid_cache_cfg.device,
            dtype=torch.bfloat16,
            max_batch_size=batch_size,
            config=self.action_head_cfg,
            max_cache_len=past_len+self.hybrid_cache_cfg.max_extra_tokens,
        )
        hybrid_cache_kwargs = {
            "cache_position": torch.arange(dynamic_cache.key_cache[0].shape[-2], device="cuda"),
            "sliding_window": self.action_head_cfg.sliding_window,
        }
        
        for dynamic_layer_idx, layer_idx in enumerate(self.layer_id_of_vlm_kv_used):
            k,v = dynamic_cache[layer_idx]
            hybrid_cache.update(
                key_states=k,
                value_states=v,
                layer_idx=dynamic_layer_idx,
                cache_kwargs=hybrid_cache_kwargs,
            )

        return hybrid_cache
    
    def forward(self, vlm_inputs=None, action_tokens=None, action_labels=None):

        kv_cache_from_vlm = self.vlm(**vlm_inputs)["past_key_values"]
        hybrid_cache_for_gemma_bases_action_head = self.transform_dynamic_cache_to_hybrid_cache(
            dynamic_cache=kv_cache_from_vlm, batch_size=action_tokens.shape[0],
        )
        
        del kv_cache_from_vlm
        gc.collect()
        
        loss = self.action_head(
            input_ids=action_tokens.to("cuda"),
            labels=action_labels.to("cuda"),
            past_key_values=hybrid_cache_for_gemma_bases_action_head,
            use_cache=True,
        ).loss
        
        return loss
    
    def infer_action_tokens(self, vlm_inputs=None):
        """
        Output action tokens. Need to further call `tokenizer.decode(output)`
        """
        kv_cache_from_vlm = self.vlm(**vlm_inputs)["past_key_values"]
        hybrid_cache_for_gemma_bases_action_head = self.transform_dynamic_cache_to_hybrid_cache(
            dynamic_cache=kv_cache_from_vlm, batch_size=vlm_inputs["input_ids"].shape[0]
        )

        del kv_cache_from_vlm
        gc.collect()

        past_seq_len = hybrid_cache_for_gemma_bases_action_head.get_seq_length()
        generated = self.action_head.generate(
            input_ids=torch.full_like(torch.randn(vlm_inputs["input_ids"].shape[0], past_seq_len+1), self.action_head_cfg.bos_token_id, device="cuda", dtype=torch.long),
            cache_implementation=None,
            past_key_values=hybrid_cache_for_gemma_bases_action_head, # TODO: expand max_cache_len of the kv cache according to max_length
            use_cache=True,
            # max_new_tokens=20, # TODO: delete this in real inference
        )
        
        action_tokens = generated[:, past_seq_len+1:-1]

        return action_tokens

    def get_into_training_stage_1(self,):
        """
        - Action head: trainable
        - Vlm: frozen
        """
        self.vae, self.infinity, self.bitwise_self_correction = None, None, None
        self.training_stage = 1
        
        self.action_head.train()
        self.vlm.eval()
        for param in self.action_head.parameters():
            param.requires_grad = True
        for param in self.vlm.parameters():
            param.requires_grad = False
            
        print("\n\n==========================================================================")
        print("🚀🚀🚀🚀🚀🚀🚀 PAY ATTENTION:\nYOU ARE ENTERING TRAINING STATE 1 (only action head trainable)\n")
        print(f"num. VLM trainable params: {int(sum(p.numel() for p in self.vlm.parameters() if p.requires_grad) // 1e6)}M")
        print(f"num. Action Head trainable params: {int(sum(p.numel() for p in self.action_head.parameters() if p.requires_grad) // 1e6)}M")
        print("==========================================================================\n\n")

    
    def get_into_training_stage_2(self,):
        """
        - Action head: trainable
        - Vlm: trainable
        """
        self.vae, self.infinity, self.bitwise_self_correction = None, None, None
        self.training_stage = 2
        
        self.action_head.train()
        self.vlm.train()
        for param in self.action_head.parameters():
            param.requires_grad = True
        for param in self.vlm.parameters():
            param.requires_grad = True
            
        print("\n\n==========================================================================")
        print("🚀🚀🚀🚀🚀🚀🚀 PAY ATTENTION:\nYOU ARE ENTERING TRAINING STATE 1 (only action head trainable)\n")
        print(f"num. VLM trainable params: {int(sum(p.numel() for p in self.vlm.parameters() if p.requires_grad) // 1e6)}M")
        print(f"num. Action Head trainable params: {int(sum(p.numel() for p in self.action_head.parameters() if p.requires_grad) // 1e6)}M")
        print("==========================================================================\n\n")



if __name__ == "__main__":
    import omegaconf
    
    from VideoPlan.trainer.datasetss.libero_lerobot_dataset import LiberoLerobotDatasetConfig
    datacfg = LiberoLerobotDatasetConfig()
    datacfg = omegaconf.OmegaConf.create(datacfg)
    dataset = instantiate_with_cfg(cfg=datacfg, split="validation_unique")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=2,
        collate_fn=dataset.collate_fn,
        num_workers=0
    )







    TEST_CASE = "vla_without_infinity"
    
    if TEST_CASE == "vlm_with_infinity":
        cfg = QwenVlmInfinityConfig()
        # use omegacfg to deal with all cfgs in cfg
        import omegaconf
        cfg = omegaconf.OmegaConf.create(cfg)
        
        
        model = instantiate_with_cfg(cfg=cfg)
        model.load_pretrained_infinity("/home/czh/.cache/huggingface/hub/models--FoundationVision--Infinity/snapshots/d4c15777e41bd36eb8eef5a854b018d19962b6d9/infinity_125M_256x256.pth")
        
        # forward
        for batch in dataloader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(vlm_inputs=batch["vlm_inputs"], next_frame=batch["future_img"])
                import IPython; IPython.embed();
                
                
                
    if TEST_CASE == "vla_without_infinity":
        cfg = QwenVlmGemmaActionHeadConfig()
        cfg = omegaconf.OmegaConf.create(cfg)
        model = instantiate_with_cfg(cfg=cfg).to(torch.bfloat16)
        
        # forward
        for batch in dataloader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(vlm_inputs=batch["vlm_inputs"].to("cuda").to(torch.bfloat16),
                            action_tokens=batch["action_tokens"].to("cuda"),
                            action_labels=batch["action_labels"].to("cuda"))
                import IPython; IPython.embed();

        