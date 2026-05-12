"""运球（Dribbling）任务模块。

对齐 DribbleBot 论文 (Ji, Margolis, Agrawal, 2023)：
  https://gmargo11.github.io/dribblebot

本模块提供四足机器人运球控制的环境配置、MDP 组件和 RL runner。
适用于 Unitree Go1 机器人，支持平坦地形和崎岖地形两种训练模式。

目录结构：
  dribbling/
  ├── __init__.py              ← 本文件
  ├── dribbling_env_cfg.py     ← 核心：RL 环境配置工厂函数
  ├── config/
  │   └── go1/
  │       ├── env_cfgs.py      ← Go1 机器人特定配置（传感器、实体、奖励定制）
  │       ├── rl_cfg.py        ← PPO 超参数（对齐论文 Table V）
  │       └── __init__.py      ← 任务注册入口
  ├── mdp/
  │   ├── __init__.py          ← 组装所有 MDP 组件到统一命名空间
  │   ├── ball_command.py      ← 球速指令（世界坐标系 vx, vy）
  │   ├── ball_reward.py       ← 论文 5 个运球奖励函数
  │   ├── ball_events.py       ← 球传送 + 球阻力模型（drag）事件
  │   ├── total_reward.py      ← 乘法奖励组合器 r_t = r_pos * exp(r_neg)
  │   ├── gait.py              ← Trot 步态时钟（Walk These Ways）
  │   ├── gait_reward.py       ← Swing/Stance 相位奖励
  │   ├── observations.py      ← 球位置、球速度、步态参考、偏航角 观测
  │   ├── rewards.py           ← 通用奖励（直立、碰撞、能耗等）
  │   ├── terminations.py      ← 终止条件（超时、跌倒、出界）
  │   ├── curriculums.py       ← 地形难度课程学习
  │   └── terrain_utils.py     ← 地形法线估计工具
  └── rl/
      ├── __init__.py          ← 导出 runner 类
      └── runner.py            ← ONNX 导出 + wandb 日志
"""
