from dataclasses import dataclass

import torch


@dataclass
class BaseDatasetConfig:
    train_split_name: str = "train"
    valid_split_name: str = "validation"
    test_split_name: str = "test"

    batch_size: int = 14
    num_workers: int = 2
    drop_last: bool = True


class BaseDataset(torch.utils.data.Dataset):
    pass

"""
==========================================================================
🚀🚀🚀🚀🚀🚀🚀 PAY ATTENTION:
YOU ARE ENTERING TRAINING STATE 2 (whole vlm and infinity trainable)

[DINO_SIGLIP] Total num of trainable parameters:  731059648
[LLM] Total num of trainable parameters:  493803392
[VLM] Total num of trainable parameters:  1227617344
[Action Head] Total num of trainable parameters:  2816263
[VLA] Total num of trainable parameters:  1230433607
==========================================================================
"""