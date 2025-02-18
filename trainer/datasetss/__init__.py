from hydra.core.config_store import ConfigStore

from VideoPlan.trainer.datasetss.libero_lerobot_dataset import LiberoLerobotDatasetConfig

cs = ConfigStore.instance()
cs.store(group="dataset", name="libero", node=LiberoLerobotDatasetConfig)
