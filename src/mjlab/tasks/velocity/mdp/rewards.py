from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
    resolve_matching_names_values,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """跟踪指令基座线速度的奖励。

    假设指令 z 方向速度为零。
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    lin_vel_error = xy_error + z_error
    return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward heading error for heading-controlled envs, angular velocity for others.

    假设指令 xy 方向角速度为零。
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    ang_vel_error = z_error + xy_error
    return torch.exp(-ang_vel_error / std**2)


class upright:
    """保持基座直立的奖励。

    不提供 ``terrain_sensor_names`` 时，惩罚相对于世界竖直方向的倾斜（适用于平
    坦地面）。

    提供 ``terrain_sensor_names`` 时，惩罚相对于地形表面法线的倾斜。
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
            "terrain_sensor_names"
        )
        self._debug_vis_enabled = True
        self._env = env
        self._asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg", _DEFAULT_ASSET_CFG
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        terrain_sensor_names: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]

        if asset_cfg.body_ids:
            body_quat_w = asset.data.body_link_quat_w[
                :, asset_cfg.body_ids, :
            ]  # [B, N, 4]
            body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
        else:
            body_quat_w = asset.data.root_link_quat_w  # [B, 4]

        if terrain_sensor_names is not None:
            terrain_normal = terrain_normal_from_sensors(
                env, terrain_sensor_names
            )  # [B, 3]
            # 将地形法线投影到机体坐标系。当与地形表面对齐时，
            # 应为 (0, 0, 1)；XY 分量衡量倾斜程度。
            target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
            xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
        else:
            gravity_w = asset.data.gravity_vec_w  # [3]
            projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
            xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

        return torch.exp(-xy_squared / std**2)

    def reset(self, env_ids: torch.Tensor) -> None:
        del env_ids  # Unused.

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        if not self._debug_vis_enabled or self._terrain_sensor_names is None:
            return

        env = self._env
        asset: Entity = env.scene[self._asset_cfg.name]

        env_indices = list(visualizer.get_env_indices(env.num_envs))
        if not env_indices:
            return

        terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
        if self._asset_cfg.body_ids:
            body_quat_w = asset.data.body_link_quat_w[
                :, self._asset_cfg.body_ids, :
            ].squeeze(1)
        else:
            body_quat_w = asset.data.root_link_quat_w
        up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
            body_quat_w[:, :3]
        )
        body_up_w = quat_apply(body_quat_w, up_local)

        positions = asset.data.root_link_pos_w.cpu().numpy()
        offset = np.array([0.0, 0.3, 0.0])
        terrain_normal_np = terrain_normal.cpu().numpy()
        body_up_np = body_up_w.cpu().numpy()
        scale = 0.25

        for i in env_indices:
            origin = positions[i] + offset
            # Terrain normal (magenta).
            visualizer.add_arrow(
                start=origin,
                end=origin + terrain_normal_np[i] * scale,
                color=(0.8, 0.2, 0.8, 0.8),
                width=0.01,
            )
            # Body up (orange).
            visualizer.add_arrow(
                start=origin,
                end=origin + body_up_np[i] * scale,
                color=(1.0, 0.5, 0.0, 0.8),
                width=0.01,
            )


def self_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """惩罚自碰撞。

    当传感器提供力历史记录时（来自 ``history_length > 0``），
    统计子步骤中接触力超过 *force_threshold* 的次数。
    否则回退到瞬时 ``found`` 计数。
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        # force_history: [B, N, H, 3]
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
        return hit.sum(dim=-1).float()  # [B]
    assert data.found is not None
    return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚过大的身体角速度。"""
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
    ang_vel = ang_vel.squeeze(1)
    ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
    return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """惩罚全身角动量，以鼓励自然的摆臂动作。"""
    angmom_sensor: BuiltinSensor = env.scene[sensor_name]
    angmom = angmom_sensor.data
    angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
    angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
    env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
    return angmom_magnitude_sq


def feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_name: str | None = None,
    command_threshold: float = 0.5,
) -> torch.Tensor:
    """奖励足部腾空时间。"""
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    current_air_time = sensor_data.current_air_time
    assert current_air_time is not None
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)
    in_air = current_air_time > 0
    num_in_air = torch.sum(in_air.float())
    mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
        num_in_air, min=1
    )
    env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


def feet_clearance(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚偏离目标离地高度，按足部速度加权。"""
    asset: Entity = env.scene[asset_cfg.name]
    height_sensor = env.scene[height_sensor_name]
    assert isinstance(height_sensor, TerrainHeightSensor), (
        f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    foot_height = height_sensor.data.heights  # [B, F]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
    delta = torch.abs(foot_height - target_height)  # [B, F]
    cost = torch.sum(delta * vel_norm, dim=1)  # [B]
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            active = (total_command > command_threshold).float()
            cost = cost * active
    return cost


class feet_swing_height:
    """惩罚偏离目标摆动高度，在着地时评估。"""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        height_sensor = env.scene[cfg.params["height_sensor_name"]]
        assert isinstance(height_sensor, TerrainHeightSensor), (
            f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
        )
        num_feet = height_sensor.num_frames
        self.peak_heights = torch.zeros(
            (env.num_envs, num_feet), device=env.device, dtype=torch.float32
        )
        self.step_dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        height_sensor_name: str,
        target_height: float,
        command_name: str,
        command_threshold: float,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene[sensor_name]
        command = env.command_manager.get_command(command_name)
        assert command is not None
        height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
        foot_heights = height_sensor.data.heights
        in_air = contact_sensor.data.found == 0
        self.peak_heights = torch.where(
            in_air,
            torch.maximum(self.peak_heights, foot_heights),
            self.peak_heights,
        )
        first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        active = (total_command > command_threshold).float()
        error = self.peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
        num_landings = torch.sum(first_contact.float())
        peak_heights_at_landing = self.peak_heights * first_contact.float()
        mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
            num_landings, min=1
        )
        env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
        self.peak_heights = torch.where(
            first_contact,
            torch.zeros_like(self.peak_heights),
            self.peak_heights,
        )
        return cost


def feet_slip(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚足部滑移（接触时的 xy 速度）。"""
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    assert contact_sensor.data.found is not None
    in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
    vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
    vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
    cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
    num_in_contact = torch.sum(in_contact)
    mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
        num_in_contact, min=1
    )
    env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
    return cost


def soft_landing(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """惩罚着地时的高冲击力，鼓励轻柔落脚。"""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = contact_sensor.data
    assert sensor_data.force is not None
    forces = sensor_data.force  # [B, N, 3]
    force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
    landing_impact = force_magnitude * first_contact.float()  # [B, N]
    cost = torch.sum(landing_impact, dim=1)  # [B]
    num_landings = torch.sum(first_contact.float())
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            active = (total_command > command_threshold).float()
            cost = cost * active
    return cost


class variable_posture:
    """惩罚偏离默认姿态，容忍度随速度变化。

    使用每个关节的标准差来控制每个关节可以偏离默认姿态的程度。
    标准差越小 = 越严格（允许的偏差越小），标准差越大 = 越宽容。
    奖励为：exp(-mean(error² / std²))

    三种速度区间（基于线速度 + 角速度指令）：
      - std_standing（速度 < walking_threshold）：严格保持姿态。
      - std_walking（walking_threshold <= 速度 < running_threshold）：中等。
      - std_running（速度 >= running_threshold）：宽松，允许大幅运动。

    根据每个关节在各速度下所需的运动量，调整该关节的 std 值。
    将关节名称模式映射到 std 值，例如 {".*knee.*": 0.35}。
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos

        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

        _, _, std_standing = resolve_matching_names_values(
            data=cfg.params["std_standing"],
            list_of_strings=joint_names,
        )
        self.std_standing = torch.tensor(
            std_standing, device=env.device, dtype=torch.float32
        )

        _, _, std_walking = resolve_matching_names_values(
            data=cfg.params["std_walking"],
            list_of_strings=joint_names,
        )
        self.std_walking = torch.tensor(
            std_walking, device=env.device, dtype=torch.float32
        )

        _, _, std_running = resolve_matching_names_values(
            data=cfg.params["std_running"],
            list_of_strings=joint_names,
        )
        self.std_running = torch.tensor(
            std_running, device=env.device, dtype=torch.float32
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_standing,
        std_walking,
        std_running,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running  # Unused.

        asset: Entity = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        assert command is not None

        linear_speed = torch.norm(command[:, :2], dim=1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = (
            (total_speed >= walking_threshold) & (total_speed < running_threshold)
        ).float()
        running_mask = (total_speed >= running_threshold).float()

        std = (
            self.std_standing * standing_mask.unsqueeze(1)
            + self.std_walking * walking_mask.unsqueeze(1)
            + self.std_running * running_mask.unsqueeze(1)
        )

        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)

        return torch.exp(-torch.mean(error_squared / (std**2), dim=1))
