from hydra.core.config_store import ConfigStore

from VideoPlan.trainer.models.infinity_model import InfinityVlmConfig, BaseModelConfig

cs = ConfigStore.instance()
cs.store(group="model", name="infinity_vlm", node=InfinityVlmConfig)

