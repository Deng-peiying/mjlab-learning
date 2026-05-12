# src/mjlab/tasks/cartpole/config/cartpole/__init__.py
from mjlab.tasks.registry import register_mjlab_task
from .env_cfg import cartpole_env_cfg
from .rl_cfg import cartpole_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-CartPole-PPO",
    env_cfg=cartpole_env_cfg(),
    play_env_cfg=cartpole_env_cfg(play=True),
    rl_cfg=cartpole_ppo_runner_cfg(),
    runner_cls=None,   # 使用默认的 MjlabOnPolicyRunner
)
