"""地形难度课程学习。

根据机器人行走距离自动调整地形难度：
  - 机器人走够远 -> 升级到更崎岖地形
  - 机器人走不够目标距离的一半 -> 降级到更简单地形

由于 dribbling 任务的指令是球速而非身体速度，这里用球速指令的范数
来估算"应该走多远"的目标距离。
"""

from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from .ball_command import BallVelocityCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
    """速度阶段配置（保留接口，dribbling 中未使用）。"""
    step: int
    lin_vel_x: tuple[float, float] | None
    lin_vel_y: tuple[float, float] | None
    ang_vel_z: tuple[float, float] | None


def terrain_levels_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
    """根据机器人行走距离调整地形等级。

    原理：
      - 用机器人实际位置与出生点的距离作为"已走距离"
      - 已走距离 > 子地形边长的一半 -> 升级
      - 已走距离 < 目标距离的 50% -> 降级
      - 目标距离 = |cmd| * episode_length（用球速指令估算）

    Args:
      env: 环境实例。
      env_ids: 要更新课程的环境 ID。
      command_name: 指令名称（"ball_vel"）。
      asset_cfg: 机器人资产配置。

    Returns:
      dict: 各地形类型的平均难度等级。
    """
    asset: Entity = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    command = env.command_manager.get_command(command_name)
    assert command is not None

    # 机器人从出生点出发的行走距离
    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )

    # 升级条件：行走距离超过子地形边长一半
    move_up = distance > terrain_generator.size[0] / 2

    # 降级条件：行走距离不足目标的一半
    # 目标距离 = 指令速率 * episode 最大时长 * 0.5
    move_down = (
        distance
        < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    )
    move_down *= ~move_up  # 已升级的不会同时降级

    terrain.update_env_origins(env_ids, move_up, move_down)

    # 统计各地形类型的平均等级
    levels = terrain.terrain_levels.float()
    result: dict[str, torch.Tensor] = {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
    }

    # 在课程模式下，每列对应一种地形类型
    sub_terrain_names = list(terrain_generator.sub_terrains.keys())
    terrain_origins = terrain.terrain_origins
    assert terrain_origins is not None
    num_cols = terrain_origins.shape[1]
    if num_cols == len(sub_terrain_names):
        types = terrain.terrain_types
        for i, name in enumerate(sub_terrain_names):
            mask = types == i
            if mask.any():
                result[name] = torch.mean(levels[mask])

    return result
