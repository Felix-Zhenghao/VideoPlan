from hydra.core.config_store import ConfigStore

from video_gen.VideoPlan.trainer.models.qwen_vla_model import VlaConfig, BaseModelConfig

cs = ConfigStore.instance()
cs.store(group="model", name="qwen_vla", node=VlaConfig)

