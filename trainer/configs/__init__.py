from hydra.core.config_store import ConfigStore

from VideoPlan.trainer.configs.configs import TrainerConfig

cs = ConfigStore.instance()
cs.store(name="base_config", node=TrainerConfig)
