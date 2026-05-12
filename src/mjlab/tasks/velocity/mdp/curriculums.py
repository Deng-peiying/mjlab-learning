from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
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
    asset: Entity = env.scene[asset_cfg.name]

    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    command = env.command_manager.get_command(command_name)
    assert command is not None

    # 计算机器人行走的距离。
    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )

    # 行走足够远的机器人进入更困难的地形。
    move_up = distance > terrain_generator.size[0] / 2

    # 行走距离不足目标一半的机器人进入更简单的地形。
    move_down = (
        distance
        < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    )
    move_down *= ~move_up

    # 更新地形等级。
    terrain.update_env_origins(env_ids, move_up, move_down)

    # 计算每种地形类型的平均等级。
    levels = terrain.terrain_levels.float()
    result: dict[str, torch.Tensor] = {
        "mean": torch.mean(levels),
        "max": torch.max(levels),
    }

    # 在课程模式下，num_cols == num_terrains（每种类型一列），
    # 因此列索引直接映射到子地形名称。
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


def commands_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
    del env_ids  # Unused.
    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None
    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
    for stage in velocity_stages:
        if env.common_step_counter >= stage["step"]:
            if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
                cfg.ranges.lin_vel_x = stage["lin_vel_x"]
            if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
                cfg.ranges.lin_vel_y = stage["lin_vel_y"]
            if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
                cfg.ranges.ang_vel_z = stage["ang_vel_z"]
    return {
        "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
        "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
        "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
        "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
        "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
        "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
    }
