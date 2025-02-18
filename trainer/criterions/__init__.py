from hydra.core.config_store import ConfigStore

from VideoPlan.trainer.criterions.infinity_criterion import InfinityVlmCriterionConfig


cs = ConfigStore.instance()
cs.store(group="criterion", name="infinity_vlm", node=InfinityVlmCriterionConfig)
