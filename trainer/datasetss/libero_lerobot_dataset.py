import sys
import os
sys.path.append(f"{os.getcwd()}/video_gen/")

from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional, List, Dict, Tuple

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch
from torch.utils.data._utils.collate import default_collate
from PIL import Image
from accelerate.logging import get_logger
from datasets import Dataset
from hydra.utils import instantiate
from omegaconf import II
from torch.nn.utils.rnn import pad_sequence


from VideoPlan.trainer.datasetss.base_dataset import BaseDataset, BaseDatasetConfig

logger = get_logger(__name__)


def simple_collate(batch, column_name):
    return torch.cat([item[column_name] for item in batch], dim=0)


@dataclass
class VlmProcessorConfig:
    _target_: str = "transformers.AutoProcessor.from_pretrained"
    pretrained_model_name_or_path: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    
@dataclass
class BscConfig:
    _target_: str = "Infinity.infinity.models.bitwise_self_correction.BitwiseSelfCorrection"
    noise_apply_layers=13
    noise_apply_requant=True
    noise_apply_strength=0.3
    apply_spatial_patchify=False
    debug_bsc=False

@dataclass
class ActionTokenizer:
    _target_: str = "transformers.AutoProcessor.from_pretrained"
    pretrained_model_name_or_path: str = "physical-intelligence/fast"
    trust_remote_code: bool = True

@dataclass
class PadActionTokensForAutoregressiveInput:
    bos_token_id: int = 2049
    pad_token_id: int = 2048
    padding_side: str = "right"
    ignore_index: int = -100
    
    def __call__(self, action_tokens: List[List[int]]) -> torch.Tensor:
        action_tokens = [torch.tensor(action) for action in action_tokens]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        action_tokens = pad_sequence(action_tokens, batch_first=True, padding_value=self.pad_token_id)
        
        # create autoregressive labels, if the label is pad_token_id, make it ignore_index
        labels = action_tokens.clone()
        labels = pad_sequence(action_tokens, batch_first=True, padding_value=self.ignore_index)
        
        # at the beginning of the sequence, add bos_token_id
        action_tokens = torch.cat([
            torch.full((action_tokens.shape[0], 1), self.bos_token_id, dtype=action_tokens.dtype),
            action_tokens,
            torch.full((action_tokens.shape[0], 1), self.pad_token_id, dtype=action_tokens.dtype)
        ],dim=1)
        
        # only keep the first padding token as eos, ignore the rest
        eos_mask = (action_tokens == self.pad_token_id)
        eos_cumsum = eos_mask.cumsum(dim=1)
        eos_mask = eos_mask & (eos_cumsum > 1)
        labels = action_tokens.clone()
        labels[eos_mask] = self.ignore_index
        
        return action_tokens, labels
    

@dataclass
class LiberoLerobotDatasetConfig(BaseDatasetConfig):
    _target_: str = "VideoPlan.trainer.datasetss.libero_lerobot_dataset.LiberoLerobotDataset"
    dataset_name: str = "Felix-Zhenghao/libero"
    dataset_config_name: str = "null"

    train_split_name: str = "train"
    valid_split_name: str = "validation_unique"
    test_split_name: str = "test_unique"
    cache_dir: Optional[str] = None
    
    # lerobot dataset config
    fps: int = 10
    num_episodes: int = 400
    training_episodes: List[int] = field(default_factory=lambda num_episodes=num_episodes:
        list(range(num_episodes))
    )
    validation_episodes: Optional[List[int]] = field(default_factory=lambda:
        [0, 50]
    )
    test_episodes: Optional[List[int]] = field(default_factory=lambda:
        [0]
    )
    validation_episodes_length: List[int] = field(default_factory=lambda:
        [214,290]
    )
    delta_timestamps: Dict[str, List[float]] = field(default_factory=lambda fps=fps: {
        # loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
        "image": [-0.8, -0.6, -0.4, -0.2, 0.],
        # loads 8 state vectors: 1.5 seconds before, 1 second before, ... 200 ms, 100 ms, and current frame
        # "state": [-0.2, -0.1, 0, 0.1],
        # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
        "actions": [t / fps for t in range(10)],
    })

    # columns
    task_description_name: str = "task"
    history_imgs_name: str = "image"
    future_imgs_name: str = "future_image"
    
    apply_spatial_patchify: bool = False
    future_img_length: int = 1
    scale_schedule: List[List[int]] = field(default_factory=lambda: 
        [[1, 1, 1], [1, 2, 2], [1, 4, 4], [1, 6, 6], [1, 8, 8], [1, 12, 12], [1, 16, 16]]
    )

    vlm_processor: VlmProcessorConfig = field(default_factory = lambda: 
        VlmProcessorConfig()
    )
    bsc: Optional[BscConfig] = None
    action_tokenizer: ActionTokenizer = field(default_factory=lambda:
        ActionTokenizer()
    )
    pad_action_tokens_for_autoregressive_input: PadActionTokensForAutoregressiveInput = field(default_factory=lambda:
        PadActionTokensForAutoregressiveInput()
    )


class LiberoLerobotDataset(BaseDataset):

    def __init__(self, cfg: LiberoLerobotDatasetConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split

        self.dataset = self.load_hf_dataset(self.split)
        self.vlm_processor = instantiate(cfg.vlm_processor)
        self.action_tokenizer = instantiate(cfg.action_tokenizer)
        self.pad_action_tokens_for_autoregressive_input = PadActionTokensForAutoregressiveInput(
            **cfg.pad_action_tokens_for_autoregressive_input
        )

    def load_hf_dataset(self, split: str) -> Dataset:
        if split == self.cfg.train_split_name:
            dataset = LeRobotDataset(
                self.cfg.dataset_name,
                episodes=self.cfg.training_episodes, # [0,100,200,300,400,500,600]
                delta_timestamps=self.cfg.delta_timestamps,
                local_files_only=True,
            )
        elif split == self.cfg.valid_split_name:
            if self.cfg.validation_episodes is None:
                raise ValueError("Validation episodes must be specified for validation split")
            dataset = LeRobotDataset(
                self.cfg.dataset_name,
                episodes=self.cfg.validation_episodes,
                delta_timestamps=self.cfg.delta_timestamps,
                local_files_only=True,
            )
        elif split == self.cfg.test_split_name:
            if self.cfg.test_episodes is None:
                raise ValueError("Test episodes must be specified for test split")
            dataset = LeRobotDataset(
                self.cfg.dataset_name,
                episodes=self.cfg.test_episodes,
                delta_timestamps=self.cfg.delta_timestamps,
                local_files_only=True,
            )
        return dataset

    def process_vlm_inputs(self, example):
        task_descriptions = example[self.cfg.task_description_name]
        history_imgs = example[self.cfg.history_imgs_name]
        prompts = [[
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": f"The robot should {task}."},
                ],
            }
        ] for task in task_descriptions]
        prompts = [self.vlm_processor.apply_chat_template(prompt, add_generation_prompt=True) for prompt in prompts]
        vlm_inputs = self.vlm_processor(videos=torch.unbind(history_imgs, dim=0), text=prompts, return_tensors='pt', padding=True)
        return vlm_inputs

    def process_vae_inputs(self, future_img, vae):
        if self.cfg.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in self.cfg.scale_schedule]
        else:
            vae_scale_schedule = [(pt, ph, pw) for pt, ph, pw in self.cfg.scale_schedule]
            
        raw_features, _, _ = vae.encode_for_raw_features(future_img, scale_schedule=vae_scale_schedule)
        bitwise_self_correction = instantiate(self.cfg.bsc, vae=vae)
        x_BLC_wo_prefix, gt_ms_idx_Bl = bitwise_self_correction.flip_requant(vae_scale_schedule, future_img, raw_features, "cuda") # x_BLC_wo_prefix: torch.Size([bs, 2*2+3*3+...+64*64, d or 4d])

        return x_BLC_wo_prefix, gt_ms_idx_Bl
    
    def process_action_inputs(self, actions):
        action_tokens = self.action_tokenizer(actions)
        action_tokens, action_labels = self.pad_action_tokens_for_autoregressive_input(action_tokens)
        return action_tokens, action_labels

    # TODO: check how to define the __getitem__ method
    def __getitem__(self, idx):
        example = self.dataset[idx]
        return example

    def collate_fn(self, batch):
        """
        Returned keys:
        - wrist_image
        - state
        - timestamp
        - frame_index
        - episode_index
        - index
        - task_index
        - image_is_pad
        - future_img
        - action_tokens
        - action_labels
        - vlm_inputs
            - input_ids
            - pixel_values_videos
            - attention_mask
        """
        collated_batch = default_collate(batch)
        
        # 'collated_batch["image"]' has shape [batch_size, seq_len, 3, 256, 256]
        full_images = collated_batch["image"]

        # Split the images
        collated_batch["image"] = full_images[:, :-self.cfg.future_img_length, ...]        # [batch_size, seq_len - future_img_len, 3, 256, 256]
        collated_batch["future_img"] = full_images[:, -self.cfg.future_img_length:, ...]   # [batch_size, future_img_len, 3, 256, 256]
        
        collated_batch["future_img"] = collated_batch["future_img"].float().div(255).squeeze(1) if collated_batch["future_img"].shape[1] == 1 else collated_batch["future_img"].float().div(255).view(-1, 3, 256, 256)

        vlm_inputs = self.process_vlm_inputs(collated_batch)
        
        action_tokens, action_labels = self.process_action_inputs(collated_batch["actions"])

        # delete self.cfg.history_imgs_name and self.cfg.task_description_name from example
        # add vlm_inputs to example
        collated_batch.pop("actions")
        collated_batch.pop(self.cfg.history_imgs_name) # free memory
        collated_batch.pop(self.cfg.task_description_name) # free memory
        collated_batch["vlm_inputs"] = vlm_inputs
        collated_batch["action_tokens"] = action_tokens
        collated_batch["action_labels"] = action_labels

        return collated_batch

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    """
    Unit Test for dataset loading
    """
    import omegaconf
    cfg = LiberoLerobotDatasetConfig()
    cfg = omegaconf.OmegaConf.create(cfg)
    dataset = LiberoLerobotDataset(cfg, split="validation_unique")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    
    for batch in dataloader:
        import IPython; IPython.embed()
        break
