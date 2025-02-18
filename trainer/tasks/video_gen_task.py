import sys
import os
sys.path.append(f"{os.getcwd()}/video_gen/")
sys.path.append(f"{os.getcwd()}/video_gen/VideoPlan/")
sys.path.append(f"{os.getcwd()}/video_gen/Infinity/")

import collections
import random
from dataclasses import dataclass

import torch
from PIL import Image
from hydra.utils import instantiate
from accelerate.logging import get_logger
from accelerate.utils import LoggerType
from omegaconf import II
from transformers import AutoTokenizer

from config_util import instantiate_with_cfg
from trainer.accelerators.base_accelerator import BaseAccelerator
from trainer.tasks.base_task import BaseTaskConfig, BaseTask

logger = get_logger(__name__)


@dataclass
class InfinityVlmTaskConfig(BaseTaskConfig):
    _target_: str = "trainer.tasks.video_gen_task.InfinityVlmTask"
    # pretrained_model_name_or_path: str = II("model.pretrained_model_name_or_path")
    # label_0_column_name: str = II("dataset.label_0_column_name")
    # label_1_column_name: str = II("dataset.label_1_column_name")

    # input_ids_column_name: str = II("dataset.input_ids_column_name")
    # pixels_0_column_name: str = II("dataset.pixels_0_column_name")
    # pixels_1_column_name: str = II("dataset.pixels_1_column_name")


def numpy_to_pil(images):
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


class InfinityVlmTask(BaseTask):
    def __init__(self, cfg: InfinityVlmTaskConfig, accelerator: BaseAccelerator):
        super().__init__(cfg, accelerator)
        # self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_or_path)
        self.cfg = cfg

    def train_step(self, model, criterion, batch):
        loss = criterion(model, batch)
        return loss
    
    @staticmethod
    def gen_one_img(model, vlm_inputs, future_img,
                    save_dir, episode_id, step_id):
        infinity_cfg = model.cfg.infinity_cfg
        
        label_B_or_BLT = model.prepare_condition_input(vlm_inputs)
        _, _, img_list = model.infinity.autoregressive_infer_cfg(
            vae=model.vae,
            scale_schedule=infinity_cfg.scale_schedule,
            label_B_or_BLT=label_B_or_BLT,
            g_seed=random.randint(0,10000),
            B=future_img.shape[0], negative_label_B_or_BLT=None,
            cfg_list=[4]*len(infinity_cfg.scale_schedule),
            tau_list=[0.5]*len(infinity_cfg.scale_schedule),
            top_k=900, top_p=0.97,
            returns_vemb=1,
            cfg_insertion_layer=[0],
            vae_type=model.cfg.vae_cfg.vae_type, ret_img=True,
            trunk_scale=1000,
            gt_leak=0, gt_ls_Bl=None, inference_model=True,
        )
        
        if not os.path.exists(f"{save_dir}/episode_{episode_id}"):
            os.makedirs(f"{save_dir}/episode_{episode_id}")
        
        for idx in range(future_img.shape[0]):
            id = random.randint(0,10000)
            
            # save pred_img (predicted by the model)
            pred_img = img_list[idx].cpu.numpy()
            pred_img.save(f"{save_dir}/episode_{episode_id}/pred_img_{step_id}_{id}.png")
            
            # save future_img (ground truth): turn from [1,3,256,256] to [3,256,256] and mul_255&uint8 and store as png
            future_img = future_img[idx, ...].cpu().numpy()
            future_img = (future_img * 255).astype("uint8")
            future_img = Image.fromarray(future_img.transpose(1, 2, 0))
            future_img.save(f"{save_dir}/episode_{episode_id}/true_img_{step_id}_{id}.png")
    
    @torch.no_grad()
    def evaluate(self, model, criterion, dataloader, save_dir, valid_episodes_length):
        self.valid_episodes_length = valid_episodes_length
        
        for step, batch in enumerate(dataloader):
            
            #########
            # Add code here
            #########
            
            self.gen_one_img(model, batch["vlm_inputs"], batch["future_img"],
                             save_dir, "infinity_125M_0_5B", f"{step}_{random.randint(0,10000)}")
        
        return

        eval_dict = self.run_inference(model, criterion, dataloader)
        eval_dict = self.gather_dict(eval_dict)
        metrics = {
            "accuracy": sum(eval_dict["is_correct"]) / len(eval_dict["is_correct"]),
            "num_samples": len(eval_dict["is_correct"])
        }
        if LoggerType.WANDB == self.accelerator.cfg.log_with:
            self.log_to_wandb(eval_dict)
        return metrics

    @staticmethod
    def features2probs(model, text_features, image_0_features, image_1_features):
        return
        image_0_scores = model.logit_scale.exp() * torch.diag(
            torch.einsum('bd,cd->bc', text_features, image_0_features))
        image_1_scores = model.logit_scale.exp() * torch.diag(
            torch.einsum('bd,cd->bc', text_features, image_1_features))
        scores = torch.stack([image_0_scores, image_1_scores], dim=-1)
        probs = torch.softmax(scores, dim=-1)
        image_0_probs, image_1_probs = probs[:, 0], probs[:, 1]
        return image_0_probs, image_1_probs

    @torch.no_grad()
    def valid_step(self, model, criterion, batch):
        return
        image_0_features, image_1_features, text_features = criterion.get_features(
            model,
            batch[self.cfg.input_ids_column_name],
            batch[self.cfg.pixels_0_column_name],
            batch[self.cfg.pixels_1_column_name]
        )
        return self.features2probs(model, text_features, image_0_features, image_1_features)

    @staticmethod
    def pixel_values_to_pil_images(pixel_values):
        return
        images = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = numpy_to_pil(images)
        return images

    def run_inference(self, model, criterion, dataloader):
        return
        eval_dict = collections.defaultdict(list)
        logger.info("Running clip score...")
        for batch in dataloader:
            image_0_probs, image_1_probs = self.valid_step(model, criterion, batch)
            agree_on_0 = (image_0_probs > image_1_probs) * batch[self.cfg.label_0_column_name]
            agree_on_1 = (image_0_probs < image_1_probs) * batch[self.cfg.label_1_column_name]
            is_correct = agree_on_0 + agree_on_1
            eval_dict["is_correct"] += is_correct.tolist()
            eval_dict["captions"] += self.tokenizer.batch_decode(
                batch[self.cfg.input_ids_column_name],
                skip_special_tokens=True
            )
            eval_dict["image_0"] += self.pixel_values_to_pil_images(batch[self.cfg.pixels_0_column_name])
            eval_dict["image_1"] += self.pixel_values_to_pil_images(batch[self.cfg.pixels_1_column_name])
            eval_dict["prob_0"] += image_0_probs.tolist()
            eval_dict["prob_1"] += image_1_probs.tolist()

            eval_dict["label_0"] += batch[self.cfg.label_0_column_name].tolist()
            eval_dict["label_1"] += batch[self.cfg.label_1_column_name].tolist()

        return eval_dict

if __name__ == "__main__":
    from VideoPlan.trainer.accelerators import DeepSpeedAcceleratorConfig
    
    cfg = DeepSpeedAcceleratorConfig()
    
    import omegaconf
    cfg = omegaconf.OmegaConf.create(cfg)
    accelerator = instantiate_with_cfg(cfg=cfg)
    
    import IPython; IPython.embed()
    