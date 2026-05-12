from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg # 场景实体引用配置（指定机器人/关节/身体/几何体名称）
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """各足部相对于地形的垂直离地高度。

    Returns:
      形状为 [B, F] 的张量，其中 F 为足端数量。
    """
    sensor = env.scene[sensor_name]
    assert isinstance(sensor, TerrainHeightSensor), (
        f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
    )
    return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    current_air_time = sensor_data.current_air_time
    assert current_air_time is not None
    return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.found is not None
    return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.force is not None
    forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
    return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))
#-------------new
def ball_pos_b(
    env: ManagerBasedRlEnv,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """球在机器人机体坐标系中的位置 (x, y, z)。

    论文 Table II: b — Ball Position, Body Frame — Dimension: 3。
    保留 Z 分量，策略需要感知球是否被压到身下（Z 异常升高）。

    Returns:
        [B, 3] 机体坐标系中的球位置。
    """
    robot: Entity = env.scene["robot"] if asset_cfg is None else env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    ball_pos_w = ball.data.root_link_pos_w          # [B, 3]
    robot_pos_w = robot.data.root_link_pos_w         # [B, 3]
    robot_quat_w = robot.data.root_link_quat_w       # [B, 4]

    # 球在机体系位置
    ball_pos_b = quat_apply_inverse(robot_quat_w, ball_pos_w - robot_pos_w)
    return ball_pos_b  # [B, 3] — 论文用 3D，保留 Z 让策略感知球是否被压到身下


def ball_vel_w(
    env: ManagerBasedRlEnv,
    ball_name: str = "ball",
) -> torch.Tensor:
    """球在世界坐标系中的线速度 (x, y)。
    Returns:
        [B, 2] 球的世界坐标系水平速度。
    """
    ball: Entity = env.scene[ball_name]
    return ball.data.root_link_lin_vel_w[:, :2]


def gait_timing_ref(env: ManagerBasedRlEnv) -> torch.Tensor:
    """步态时序参考变量 θ_cmd (Walk These Ways, CoRL 2022)。

    返回 sin/cos 两个 trot 对角的相位，策略据此感知当前在步态周期中的位置，
    从而协调踢球时机与脚步落点。

    Returns:
        [B, 4] sin(2π·φ₀), cos(2π·φ₀), sin(2π·φ₁), cos(2π·φ₁)
    """
    gait_state = getattr(env, "gait_state", None) or {}
    timing = gait_state.get("timing_ref")
    if timing is None:
        return torch.zeros(env.num_envs, 4, device=env.device)
    return timing


def body_yaw(env: ManagerBasedRlEnv) -> torch.Tensor:
    """机器人在世界坐标系中的偏航角 (heading)。

    论文 III-A.3：策略需要知道自身在世界坐标系中的朝向，才能处理
    世界坐标系中的球速指令。来自 IMU。

    Returns:
        [B, 1] 偏航角，范围 [-π, π]。
    """
    robot: Entity = env.scene["robot"]
    return robot.data.heading_w.unsqueeze(-1)  # [B, 1]
