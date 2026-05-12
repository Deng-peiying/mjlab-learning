"""Swing/Stance Phase gait rewards. Aligned with DribbleBot paper Table III.

Swing Phase Schedule:  [1-κ] * exp(-δcf * |f_foot|²)   weight 4.0
  - 脚在空中时不应有接触力。κ=0（摆动）奖励 = exp(-f²/σ²)；κ=1（支撑）奖励 = 0。

Stance Phase Schedule: κ * exp(-δcv * |v_foot_xy|²)     weight 4.0
  - 脚着地时不应滑动。κ=1（支撑）奖励 = exp(-v²/σ²)；κ=0（摆动）奖励 = 0。
"""

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def swing_phase(
    env,
    sensor_name: str = "feet_ground_contact",
    force_sigma: float = 50.0,
) -> torch.Tensor:
    """摆动相奖励：脚离地时接触力越小越好。

    返回 [0, 1]，1 表示脚在空中且完全没有接触力（理想状态）。
    只在 κ→0（摆动相）时激活。
    """
    sensor: ContactSensor = env.scene[sensor_name]
    foot_forces = torch.norm(sensor.data.force, dim=-1)  # [B, 4]
    # gait_state 由 gait_clock step event 更新，但 reward 可能在 event 之前被调用
    gait_state = getattr(env, "gait_state", None) or {}
    desired_contact = gait_state.get("desired_contact", torch.ones_like(foot_forces))  # [B, 4]

    # 摆动相 (κ≈0)：力越小 → exp(-f²/σ²) 越接近 1
    # 支撑相 (κ≈1)：此项权重 ≈0
    reward = 0.0
    for i in range(4):
        swing_weight = (1.0 - desired_contact[:, i])
        force_quality = torch.exp(-foot_forces[:, i] ** 2 / force_sigma)
        reward += swing_weight * force_quality
    return reward / 4.0


def stance_phase(
    env,
    sensor_name: str = "feet_ground_contact",
    vel_sigma: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """支撑相奖励：脚着地时滑动越小越好。

    返回 [0, 1]，1 表示脚稳稳踩住完全不滑（理想状态）。
    只在 κ→1（支撑相）时激活。
    """
    robot: Entity = env.scene[asset_cfg.name]
    foot_vel_w = robot.data.site_lin_vel_w[:, asset_cfg.site_ids, :]  # [B, 4, 3]
    foot_vel_xy = torch.norm(foot_vel_w[:, :, :2], dim=-1)            # [B, 4]
    # gait_state 由 gait_clock step event 更新，reward 可能在 event 前被调用
    gait_state = getattr(env, "gait_state", None) or {}
    desired_contact = gait_state.get("desired_contact", torch.ones_like(foot_vel_xy))

    # 支撑相 (κ≈1)：速度越小 → exp(-v²/σ²) 越接近 1
    # 摆动相 (κ≈0)：此项权重 ≈0
    reward = 0.0
    for i in range(4):
        stance_weight = desired_contact[:, i]
        vel_quality = torch.exp(-foot_vel_xy[:, i] ** 2 / vel_sigma)
        reward += stance_weight * vel_quality
    return reward / 4.0
