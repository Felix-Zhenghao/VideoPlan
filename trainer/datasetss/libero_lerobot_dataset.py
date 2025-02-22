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
        "image": [-0.3, -0.2, -0.1, 0., 0.1],
        # loads 8 state vectors: 1.5 seconds before, 1 second before, ... 200 ms, 100 ms, and current frame
        "state": [-0.3, -0.2, -0.1, 0.],
        # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
        "actions": [t / fps for t in range(16)],
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


class LiberoLerobotDataset(BaseDataset):

    def __init__(self, cfg: LiberoLerobotDatasetConfig, split: str = "train",
                 dino_siglip_image_transform=None, llm_tokenizer=None):
        self.cfg = cfg
        self.split = split
        self.dino_siglip_image_transform = dino_siglip_image_transform
        self.llm_tokenizer = llm_tokenizer
        # logger.info(f"🚀🚀🚀🚀🚀🚀🚀 Loading {self.split} dataset")

        self.dataset = self.load_hf_dataset(self.split)
        # logger.info(f"🚀🚀🚀🚀🚀🚀🚀 Loaded {len(self.dataset)} examples from {self.split} dataset")

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

    def process_llm_inputs(self, example):
        task_descriptions = example[self.cfg.task_description_name]
        messages = [[
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"{task}"},
        ] for task in task_descriptions]
        texts = self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        llm_inputs = self.llm_tokenizer(texts, return_tensors="pt", padding=True,).input_ids
    
        return llm_inputs

    def __getitem__(self, idx):
        example: Dict = self.dataset[idx]
        
        # get future image
        future_img = example["image"][-self.cfg.future_img_length:] if self.cfg.future_img_length > 0 else None
        history_imgs = example["image"][:-self.cfg.future_img_length] if self.cfg.future_img_length > 0 else example["image"]
        
        example.pop("image") # free memory
        
        # do image transformation on history images
        history_imgs = [Image.fromarray(img) for img in history_imgs.permute(0,2,3,1).numpy()]
        if self.dino_siglip_image_transform is not None:
            history_imgs = self.dino_siglip_image_transform(history_imgs)
            example["dino"] = history_imgs["dino"]
            example["siglip"] = history_imgs["siglip"]
            
        if self.cfg.future_img_length > 0:
            example["future_img"] = future_img
        
        return example

    def collate_fn(self, batch):
        """
        Returned batch: 
        
        ```
        In [1]: batch.keys()
        Out[1]: dict_keys(['wrist_image', 'state', 'actions', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'image_is_pad', 'dino', 'siglip', 'future_img', 'input_ids'])
        ```
        """
        collated_batch = default_collate(batch)

        input_ids = self.process_llm_inputs(collated_batch)

        # delete self.cfg.history_imgs_name and self.cfg.task_description_name from example
        # add vlm_inputs to example
        collated_batch.pop(self.cfg.task_description_name) # free memory
        collated_batch["input_ids"] = input_ids

        return collated_batch

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    cfg = LiberoLerobotDatasetConfig()
    dataset = LiberoLerobotDataset(cfg, split="train")
    print(len(dataset))
    print(dataset[0])
