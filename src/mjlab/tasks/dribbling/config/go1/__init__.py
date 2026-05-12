"""Unitree Go1 运球任务注册入口。

通过 register_mjlab_task 将 rough 和 flat 两种地形配置注册到任务系统：
  - Mjlab-Dribbling-Rough-Unitree-Go1  — 崎岖地形运球
  - Mjlab-Dribbling-Flat-Unitree-Go1    — 平坦地形运球

训练命令：
  python -m mjlab.scripts.train Mjlab-Dribbling-Flat-Unitree-Go1
  python -m mjlab.scripts.train Mjlab-Dribbling-Rough-Unitree-Go1

每个任务注册包含：
  env_cfg          — 环境配置（训练用）
  play_env_cfg     — 环境配置（play 模式：无限 episode、无噪声、无扰动）
  rl_cfg           — PPO 超参数配置
  runner_cls       — 自定义 Runner（支持 ONNX 导出 + wandb 日志）
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.dribbling.rl import DribblingOnPolicyRunner
from .env_cfgs import unitree_go1_flat_env_cfg, unitree_go1_rough_env_cfg
from .rl_cfg import unitree_go1_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Dribbling-Rough-Unitree-Go1",
    env_cfg=unitree_go1_rough_env_cfg(),
    play_env_cfg=unitree_go1_rough_env_cfg(play=True),
    rl_cfg=unitree_go1_ppo_runner_cfg(),
    runner_cls=DribblingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Dribbling-Flat-Unitree-Go1",
    env_cfg=unitree_go1_flat_env_cfg(),
    play_env_cfg=unitree_go1_flat_env_cfg(play=True),
    rl_cfg=unitree_go1_ppo_runner_cfg(),
    runner_cls=DribblingOnPolicyRunner,
)
