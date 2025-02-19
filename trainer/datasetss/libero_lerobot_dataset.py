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

from VideoPlan.trainer.datasetss.base_dataset import BaseDataset, BaseDatasetConfig

logger = get_logger(__name__)


def simple_collate(batch, column_name):
    return torch.cat([item[column_name] for item in batch], dim=0)


@dataclass
class VlmProcessorConfig:
    _target_: str = "transformers.AutoProcessor.from_pretrained"
    pretrained_model_name_or_path: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"

@dataclass
class DinoProcessorConfig:
    _target_: str = "transformers.AutoImageProcessor.from_pretrained"
    pretrained_model_name_or_path: str = "facebook/dinov2-with-registers-large"
    
@dataclass
class BscConfig:
    _target_: str = "Infinity.infinity.models.bitwise_self_correction.BitwiseSelfCorrection"
    noise_apply_layers=13
    noise_apply_requant=True
    noise_apply_strength=0.3
    apply_spatial_patchify=False
    debug_bsc=False


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
    num_episodes: int = 1693
    training_episodes: List[int] = field(default_factory=lambda num_episodes=num_episodes:
        list(range(num_episodes))
    )
    validation_episodes: Optional[List[int]] = field(default_factory=lambda:
        [0]
    )
    test_episodes: Optional[List[int]] = field(default_factory=lambda:
        [0]
    )
    validation_episodes_length: List[int] = field(default_factory=lambda:
        [0]
    )
    delta_timestamps: Dict[str, List[float]] = field(default_factory=lambda: {
        # loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
        "image": [-100., -0.8, -0.5, -0.2, 0., 0.1], # a very big first value is needed to load the first image
        # loads 8 state vectors: 1.5 seconds before, 1 second before, ... 200 ms, 100 ms, and current frame
        # "state": [-0.2, -0.1, 0, 0.1],
        # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
        # "actions": [t / FPS for t in range(16)],
    })

    # columns
    task_description_name: str = "task"
    history_imgs_name: str = "image"
    future_imgs_name: str = "future_image"
    dino_input_name: str = "dino_input"
    
    apply_spatial_patchify: bool = False
    future_img_length: int = 1
    scale_schedule: List[List[int]] = field(default_factory=lambda: 
        [[1, 1, 1], [1, 2, 2], [1, 4, 4], [1, 6, 6], [1, 8, 8], [1, 12, 12], [1, 16, 16]]
    )

    vlm_processor: VlmProcessorConfig = field(default_factory = lambda: 
        VlmProcessorConfig()
    )
    dino_processor: DinoProcessorConfig = field(default_factory = lambda:
        DinoProcessorConfig()
    )
    bsc: Optional[BscConfig] = None


class LiberoLerobotDataset(BaseDataset):

    def __init__(self, cfg: LiberoLerobotDatasetConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split
        # logger.info(f"🚀🚀🚀🚀🚀🚀🚀 Loading {self.split} dataset")

        self.dataset = self.load_hf_dataset(self.split)
        # logger.info(f"🚀🚀🚀🚀🚀🚀🚀 Loaded {len(self.dataset)} examples from {self.split} dataset")

        self.vlm_processor = instantiate(cfg.vlm_processor)
        # logger.info(f"🚀🚀🚀🚀🚀🚀🚀 Loaded VLM processor")
        
        self.dino_processor = instantiate(cfg.dino_processor)

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
                    {"type": "text", "text": f"The video I give you shows the robot doing the task {task}. Describe things on the table and the whole environment in great details. Finally describe what you think the robot should do next."},
                ],
            }
        ] for task in task_descriptions]
        prompts = [self.vlm_processor.apply_chat_template(prompt, add_generation_prompt=True) for prompt in prompts]
        vlm_inputs = self.vlm_processor(videos=torch.unbind(history_imgs, dim=0), text=prompts, return_tensors='pt', padding=True)
        return vlm_inputs
    
    def process_dino_inputs(self, example):
        images = example[self.cfg.dino_input_name].permute(0, 2, 3, 1).numpy()
        images = [Image.fromarray(img) for img in images]
        inputs = self.dino_processor(images=images, return_tensors="pt")
        return inputs

    def process_vae_inputs(self, future_img, vae):
        if self.cfg.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in self.cfg.scale_schedule]
        else:
            vae_scale_schedule = [(pt, ph, pw) for pt, ph, pw in self.cfg.scale_schedule]
            
        raw_features, _, _ = vae.encode_for_raw_features(future_img, scale_schedule=vae_scale_schedule)
        bitwise_self_correction = instantiate(self.cfg.bsc, vae=vae)
        x_BLC_wo_prefix, gt_ms_idx_Bl = bitwise_self_correction.flip_requant(vae_scale_schedule, future_img, raw_features, "cuda") # x_BLC_wo_prefix: torch.Size([bs, 2*2+3*3+...+64*64, d or 4d])

        return x_BLC_wo_prefix, gt_ms_idx_Bl

    # TODO: check how to define the __getitem__ method
    def __getitem__(self, idx):
        example = self.dataset[idx]
        return example

    def collate_fn(self, batch):
        """
        Returned keys:
        - wrist_image
        - state
        - actions
        - timestamp
        - frame_index
        - episode_index
        - index
        - task_index
        - image_is_pad
        - future_img
        - vlm_inputs
            - input_ids
            - pixel_values_videos
            - attention_mask
        """
        collated_batch = default_collate(batch)
        
        # 'collated_batch["image"]' has shape [batch_size, seq_len, 3, 256, 256]
        full_images = collated_batch["image"]

        # Split the images
        collated_batch["dino_input"] = full_images[:, 0, ...] # [batch_size, 3, 256, 256]
        collated_batch["image"] = full_images[:, 1:-self.cfg.future_img_length, ...]        # [batch_size, seq_len - future_img_len, 3, 256, 256]
        collated_batch["future_img"] = full_images[:, -self.cfg.future_img_length:, ...]   # [batch_size, future_img_len, 3, 256, 256]
        
        collated_batch["future_img"] = collated_batch["future_img"].float().div(255).squeeze(1) if collated_batch["future_img"].shape[1] == 1 else collated_batch["future_img"].float().div(255).view(-1, 3, 256, 256)

        vlm_inputs = self.process_vlm_inputs(collated_batch)
        collated_batch["dino_input"] = self.process_dino_inputs(collated_batch)

        # delete self.cfg.history_imgs_name and self.cfg.task_description_name from example
        # add vlm_inputs to example
        collated_batch.pop(self.cfg.history_imgs_name) # free memory
        collated_batch.pop(self.cfg.task_description_name) # free memory
        collated_batch["vlm_inputs"] = vlm_inputs

        return collated_batch

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    cfg = LiberoLerobotDatasetConfig()
    dataset = LiberoLerobotDataset(cfg, split="train")
    print(len(dataset))
    print(dataset[0])
