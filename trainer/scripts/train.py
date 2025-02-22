"""
############################################
# VERY IMPORTANT:
############################################

1. To make the code works, you need to change the LeRobot internal code:
- Go to lerobot.common.datasets.utils
- Go to `hf_transform_to_torch`
- Change the line `to_tensor = transforms.ToTensor()` to `to_tensor = transforms.PILToTensor()`

Effect of the change above: prevent lerobot from normalizing PIL images to values between [0,1] , so PIL images are only transformed into tensor with uint8 value between [0,255]

--------------------------------------------

2. To make the code works, you need to change the DeepSpeed internal code:
- Go to deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine
- Go to class TorchCheckpointEngine => load() method
- Change the line `partition = torch.load(path, map_location=map_location)` to `partition = torch.load(path, map_location=map_location, weights_only=False)`

Effect of the change ensures that deepspeed can correctly load the checkpoint file to resume training.
The reason is new version of pytorch change the default setting of torch.load as weights_only=True, which only load the model weights, not the optimizer and scheduler states.

--------------------------------------------

3. Pay attention to the difference between DeepSpeed mixture-precision and PyTorch AMP:
- DeepSpeed use lower precision for forward and backward pass, and fp32 for optimizer step
- PyTorch AMP use fp32 for forward and backward pass, but automatically cast the input to lower precision for some ops.

The result is that if there is some code disables the autocast, then the code uses DeepSpeed for training v.s. not using DeepSpeed for training will be different.
The main difference is that you need to cast/uncast some inputs to make sure the inputs are in lower precision because DeepSpeed uses lower precision for parameters and gradients.

--------------------------------------------

4. To make the code works, you need to change the vlap internal code:
- Go to vlap.models.vision.base_vision
- Go to funtion `compute_sequence_patches`
- Change to line `patches = featurizer(trunc_pixels.reshape(-1, C, H, W)).reshape(B, T, -1, featurizer.embed_dim).reshape(B, -1, featurizer.embed_dim)` to `patches = featurizer(trunc_pixels.reshape(-1, C, H, W))[0].reshape(B, T, -1, featurizer.embed_dim).reshape(B, -1, featurizer.embed_dim)`

The featurizer returns a list of torch.Tensor with length 1.
If you don't do so, you will always get `list object does not have attribute reshape` error.
"""

import sys
import os
sys.path.append(f"{os.getcwd()}/video_gen/")
sys.path.append(f"{os.getcwd()}/video_gen/VideoPlan/")
sys.path.append(f"{os.getcwd()}/video_gen/Infinity/")

import json
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from torch import nn

from trainer.accelerators.base_accelerator import BaseAccelerator
from trainer.configs.configs import TrainerConfig
from config_util import instantiate_with_cfg

logger = get_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_dataloaders(cfg: DictConfig) -> Any:
    dataloaders = {}
    for split in [cfg.train_split_name, cfg.valid_split_name, cfg.test_split_name]:
        dataset = instantiate_with_cfg(cfg, split=split)
        should_shuffle = split == cfg.train_split_name
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            shuffle=should_shuffle,
            batch_size=1 if split == cfg.valid_split_name else cfg.batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=cfg.num_workers
        )
        
        if split == cfg.valid_split_name:
            valid_episode_length = dataset.cfg.validation_episodes_length
    return dataloaders, valid_episode_length


def load_optimizer(cfg: DictConfig, model: nn.Module):
    optimizer = instantiate(cfg, model=model)
    return optimizer


def load_scheduler(cfg: DictConfig, optimizer):
    scheduler = instantiate_with_cfg(cfg, optimizer=optimizer)
    return scheduler


def load_task(cfg: DictConfig, accelerator: BaseAccelerator):
    task = instantiate_with_cfg(cfg, accelerator=accelerator)
    return task


def verify_or_write_config(cfg: TrainerConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    yaml_path = os.path.join(cfg.output_dir, "config.yaml")
    if not os.path.exists(yaml_path):
        OmegaConf.save(cfg, yaml_path, resolve=True)
    with open(yaml_path) as f:
        existing_config = f.read()
    if existing_config != OmegaConf.to_yaml(cfg, resolve=True):
        raise ValueError(f"Config was not saved correctly - {yaml_path}")
    logger.info(f"Config can be found in {yaml_path}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: TrainerConfig) -> None:
    accelerator = instantiate_with_cfg(cfg.accelerator)
    # accelerator.end_training()

    if cfg.debug.activate and accelerator.is_main_process:
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=cfg.debug.port, stdoutToServer=True, stderrToServer=True)

    if accelerator.is_main_process:
        verify_or_write_config(cfg)

    logger.info(f"Loading task")
    task = load_task(cfg.task, accelerator)
    logger.info(f"Loading model")
    model = instantiate_with_cfg(cfg.model)
    logger.info(f"Loading criterion")
    criterion = instantiate_with_cfg(cfg.criterion)
    logger.info(f"Loading optimizer")
    optimizer = load_optimizer(cfg.optimizer, model)
    logger.info(f"Loading lr scheduler")
    lr_scheduler = load_scheduler(cfg.lr_scheduler, optimizer)
    logger.info(f"Loading dataloaders")
    split2dataloader, valid_episodes_length = load_dataloaders(cfg.dataset)

    dataloaders = list(split2dataloader.values())
    model, optimizer, lr_scheduler, *dataloaders = accelerator.prepare(model, optimizer, lr_scheduler, *dataloaders)
    split2dataloader = dict(zip(split2dataloader.keys(), dataloaders))

    accelerator.load_state_if_needed()

    accelerator.recalc_train_length_after_prepare(len(split2dataloader[cfg.dataset.train_split_name]))

    accelerator.init_training(cfg)
    
    if accelerator.get_latest_checkpoint() is not None:
        model.load_pretrained_infinity("/home/czh/.cache/huggingface/hub/models--FoundationVision--Infinity/snapshots/d4c15777e41bd36eb8eef5a854b018d19962b6d9/infinity_125M_256x256.pth")

    def evaluate():
        return
        model.eval()
        end_of_train_dataloader = accelerator.gradient_state.end_of_dataloader
        logger.info(f"*** Evaluating {cfg.dataset.valid_split_name} ***")
        task.evaluate(model, criterion, split2dataloader[cfg.dataset.valid_split_name], accelerator.cfg.save_dir, valid_episodes_length)
        # metrics = task.evaluate(model, criterion, split2dataloader[cfg.dataset.valid_split_name])
        # accelerator.update_metrics(metrics)
        # accelerator.gradient_state.end_of_dataloader = end_of_train_dataloader

    logger.info(f"task: {task.__class__.__name__}")
    logger.info(f"model: {model.__class__.__name__}")
    logger.info(f"num. model params: {int(sum(p.numel() for p in model.parameters()) // 1e6)}M")
    logger.info(
        f"num. model trainable params: {int(sum(p.numel() for p in model.parameters() if p.requires_grad) // 1e6)}M")
    logger.info(f"criterion: {criterion.__class__.__name__}")
    logger.info(f"num. train examples: {len(split2dataloader[cfg.dataset.train_split_name].dataset)}")
    logger.info(f"num. valid examples: {len(split2dataloader[cfg.dataset.valid_split_name].dataset)}")
    logger.info(f"num. test examples: {len(split2dataloader[cfg.dataset.test_split_name].dataset)}")

    for epoch in range(accelerator.cfg.num_epochs):
        train_loss, lr = 0.0, 0.0
        
        # skip epochs if resume training
        if epoch < accelerator.epoch:
            logger.info(f"🚀🚀🚀🚀🚀🚀🚀 Skipping epoch {epoch}.")
            accelerator.progress_bar.update(accelerator.num_steps_per_epoch)
            if epoch == accelerator.epoch - 1:
                logger.info(f"🚀🚀🚀🚀🚀🚀🚀 Start to skipping the remaining batches in the last epoch you end the training. THIS MAY TAKE FEW MINUTES IF YOU DO LOTS OF PREPROCESSING DURING DATALOADING.")
            continue

        for step, batch in enumerate(split2dataloader[cfg.dataset.train_split_name]):

            if accelerator.should_skip(epoch, step):
                accelerator.update_progbar_step()
                continue

            if accelerator.should_eval():
                evaluate()

            if accelerator.should_save():
                accelerator.save_checkpoint()

            if (accelerator.should_stage_2() and not accelerator.has_changed_to_stage_2) or (not accelerator.cfg.enable_stage_1 and not accelerator.has_changed_to_stage_2):
                model.get_into_training_stage_2()
                accelerator.has_changed_to_stage_2 = True
            elif not accelerator.should_stage_2() and not accelerator.has_changed_to_stage_1:
                model.get_into_training_stage_1()
                accelerator.has_changed_to_stage_1 = True

            with accelerator.accumulate(model):
                loss = task.train_step(model, criterion, batch)
                avg_loss = accelerator.gather(loss).mean().item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters())

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            train_loss += avg_loss / accelerator.cfg.gradient_accumulation_steps

            if accelerator.sync_gradients:
                accelerator.update_global_step(train_loss)
                train_loss = 0.0

            if accelerator.global_step > 0:
                lr = lr_scheduler.get_last_lr()[0]

            accelerator.update_step(avg_loss, lr)

            if accelerator.should_end():
                evaluate()
                accelerator.save_checkpoint()
                break

        if accelerator.should_end():
            break

        accelerator.update_epoch()

    accelerator.wait_for_everyone()
    accelerator.load_best_checkpoint()
    # logger.info(f"*** Evaluating {cfg.dataset.valid_split_name} ***")
    # metrics = task.evaluate(model, criterion, split2dataloader[cfg.dataset.valid_split_name])
    # accelerator.update_metrics(metrics)
    # logger.info(f"*** Evaluating {cfg.dataset.test_split_name} ***")
    # metrics = task.evaluate(model, criterion, split2dataloader[cfg.dataset.test_split_name])
    # metrics = {f"{cfg.dataset.test_split_name}_{k}": v for k, v in metrics.items()}
    # accelerator.update_metrics(metrics)
    accelerator.unwrap_and_save(model)
    accelerator.end_training()


if __name__ == '__main__':
    main()
