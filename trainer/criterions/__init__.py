from hydra.core.config_store import ConfigStore

from video_gen.VideoPlan.trainer.criterions.vla_criterion import VlaCriterionConfig


cs = ConfigStore.instance()
cs.store(group="criterion", name="qwen_vla", node=VlaCriterionConfig)
