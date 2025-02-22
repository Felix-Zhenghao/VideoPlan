
from hydra.core.config_store import ConfigStore

from VideoPlan.trainer.tasks.video_gen_task import InfinityVlmTaskConfig

cs = ConfigStore.instance()
cs.store(group="task", name="qwen_vla", node=InfinityVlmTaskConfig)

