from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        # force_history: [B, N, H, 3]
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
    assert data.found is not None
    return torch.any(data.found, dim=-1)


def out_of_terrain_bounds(
    env: ManagerBasedRlEnv,
    margin: float = 0.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """当机器人离开已生成地形范围时截断。

    对于非生成器地形（如平面），返回全 False。
    """
    terrain = env.scene.terrain
    if terrain is None or terrain.cfg.terrain_type != "generator":
        return torch.zeros(
            (env.num_envs,),
            device=env.device,
            dtype=torch.bool,
        )

    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None or terrain.terrain_origins is None:
        return torch.zeros(
            (env.num_envs,),
            device=env.device,
            dtype=torch.bool,
        )

    asset: Entity = env.scene[asset_cfg.name]
    root_xy_w = asset.data.root_link_pos_w[:, :2]

    # 使用生成的网格形状（课程模式会用 len(sub_terrains) 覆盖 cfg.num_cols），
    # 并计入网格周围的平坦边界。
    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    half_x = (
        0.5 * (num_rows * terrain_generator.size[0]) + terrain_generator.border_width
    )
    half_y = (
        0.5 * (num_cols * terrain_generator.size[1]) + terrain_generator.border_width
    )
    limit_x = max(0.0, half_x - margin)
    limit_y = max(0.0, half_y - margin)

    return (root_xy_w[:, 0].abs() > limit_x) | (root_xy_w[:, 1].abs() > limit_y)


def terrain_edge_reached(
    env: ManagerBasedRlEnv,
    threshold_fraction: float = 0.95,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """当机器人偏移出生点的距离超过子地形大小时终止。

    建议设置为 ``time_out=True``（成功穿越，不惩罚）。跳过重置后
    前 2 步，避免因过时的位置数据误触发。
    """
    terrain = env.scene.terrain
    if terrain is None or terrain.cfg.terrain_type != "generator":
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    asset: Entity = env.scene[asset_cfg.name]
    displacement = (
        asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    ).abs()

    half_x = terrain_generator.size[0] / 2.0 * threshold_fraction
    half_y = terrain_generator.size[1] / 2.0 * threshold_fraction

    at_edge = (displacement[:, 0] > half_x) | (displacement[:, 1] > half_y)

    # 重置后前 2 步不触发（位置数据可能已过时）。
    at_edge &= env.episode_length_buf > 2

    return at_edge
