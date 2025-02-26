from hydra.core.config_store import ConfigStore

from VideoPlan.trainer.models.infinity_model import QwenVlmInfinityConfig, QwenVlmGemmaActionHeadConfig, BaseModelConfig

cs = ConfigStore.instance()
cs.store(group="model", name="infinity_vlm", node=QwenVlmInfinityConfig)
cs.store(group="model", name="vla_without_infinity", node=QwenVlmGemmaActionHeadConfig)

