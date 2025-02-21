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
    pretrained_model_name_or_path: str = "google/paligemma2-3b-pt-224"


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
    delta_timestamps: Dict[str, List[float]] = field(default_factory=lambda: {
        # loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
        "image": [-0.3, -0.2, -0.1, 0., 0.1],
        # loads 8 state vectors: 1.5 seconds before, 1 second before, ... 200 ms, 100 ms, and current frame
        "state": [-0.3, -0.2, -0.1, 0., 0.1],
        # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
        "actions": [t / 10 for t in range(16)],
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



class LiberoLerobotDataset(BaseDataset):

    def __init__(self, cfg: LiberoLerobotDatasetConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split
        # logger.info(f"🚀🚀🚀🚀🚀🚀🚀 Loading {self.split} dataset")

        self.dataset = self.load_hf_dataset(self.split)
        # logger.info(f"🚀🚀🚀🚀🚀🚀🚀 Loaded {len(self.dataset)} examples from {self.split} dataset")

        self.vlm_processor = instantiate(cfg.vlm_processor)
        # logger.info(f"🚀🚀🚀🚀🚀🚀🚀 Loaded VLM processor")

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
        image = example["future_img"].permute(0, 2, 3, 1).numpy()
        image = [Image.fromarray(img) for img in image]
        prompt = ""
        model_inputs = self.vlm_processor(text=[prompt]*len(image), images=image, return_tensors="pt")

        return model_inputs

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
        collated_batch["image"] = full_images[:, :-self.cfg.future_img_length, ...]        # [batch_size, seq_len - future_img_len, 3, 256, 256]
        collated_batch["future_img"] = full_images[:, -self.cfg.future_img_length:, ...]   # [batch_size, future_img_len, 3, 256, 256]
        
        collated_batch["future_img"] = collated_batch["future_img"].squeeze(1) if collated_batch["future_img"].shape[1] == 1 else collated_batch["future_img"].float().div(255).view(-1, 3, 256, 256)

        vlm_inputs = self.process_vlm_inputs(collated_batch)

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
