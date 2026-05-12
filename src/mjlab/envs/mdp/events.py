"""MDP 事件的实用方法。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def randomize_terrain(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """在重置时为每个环境随机化子地形。

    为每个环境随机选择一个地形类型（列）和难度等级（行）。
    适用于 play/evaluation 模式以在多样化地形上进行测试。
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    terrain = env.scene.terrain
    if terrain is not None:
        terrain.randomize_env_origins(env_ids)


def reset_scene_to_default(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> None:
    """将场景中的所有实体重置为默认状态。

    对于浮动基座实体：重置根状态（位置、朝向、速度）。
    对于固定基座 mocap 实体：重置 mocap 姿态。
    对于所有关节实体：重置关节位置和速度。

    自动应用 env_origins 偏移以正确定位所有实体。
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    for entity in env.scene.entities.values():
        if not isinstance(entity, Entity):
            continue

        # 重置根/mocap 姿态。
        if entity.is_fixed_base and entity.is_mocap:
            # 固定基座 mocap 实体 - 使用 env_origins 重置 mocap 姿态。
            default_root_state = entity.data.default_root_state[env_ids].clone()
            mocap_pose = torch.zeros((len(env_ids), 7), device=env.device)
            mocap_pose[:, 0:3] = (
                default_root_state[:, 0:3] + env.scene.env_origins[env_ids]
            )
            mocap_pose[:, 3:7] = default_root_state[:, 3:7]
            entity.write_mocap_pose_to_sim(mocap_pose, env_ids=env_ids)
        elif not entity.is_fixed_base:
            # 浮动基座实体 - 使用 env_origins 重置根状态。
            default_root_state = entity.data.default_root_state[env_ids].clone()
            default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
            entity.write_root_state_to_sim(default_root_state, env_ids=env_ids)

        # 重置关节实体的关节状态。
        if entity.is_articulated:
            default_joint_pos = entity.data.default_joint_pos[env_ids].clone()
            default_joint_vel = entity.data.default_joint_vel[env_ids].clone()
            entity.write_joint_state_to_sim(
                default_joint_pos, default_joint_vel, env_ids=env_ids
            )


def reset_root_state_uniform(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """重置浮动基座或 mocap 固定基座实体的根状态。

    对于浮动基座实体：通过 write_root_state_to_sim() 重置姿态和速度。
    对于固定基座 mocap 实体：仅通过 write_mocap_pose_to_sim() 重置姿态。

    .. note::
      此函数会应用 env_origins 偏移，将实体定位在网格中。
      对于固定基座机器人，这是按环境定位它们的**唯一**方式。
      如果在重置事件中不调用此函数，固定基座机器人将全部堆叠在 (0,0,0) 处。

    参见 FAQ："为什么我的固定基座机器人全部堆叠在原点？"

    Args:
      env: 环境实例。
      env_ids: 要重置的环境 ID。若为 None，则重置所有环境。
      pose_range: 字典，键为 {"x", "y", "z", "roll", "pitch", "yaw"}。
      velocity_range: 速度范围（仅用于浮动基座实体）。
      asset_cfg: 资产配置。
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    # 姿态。
    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=env.device)
    pose_samples = sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
    )

    # 具有 mocap=True 的固定基座实体。
    if asset.is_fixed_base:
        if not asset.is_mocap:
            raise ValueError(
                f"Cannot reset root state for fixed-base non-mocap entity '{asset_cfg.name}'."
            )

        default_root_state = asset.data.default_root_state
        assert default_root_state is not None
        root_states = default_root_state[env_ids].clone()

        positions = (
            root_states[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
        )
        orientations_delta = quat_from_euler_xyz(
            pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
        )
        orientations = quat_mul(root_states[:, 3:7], orientations_delta)

        asset.write_mocap_pose_to_sim(
            torch.cat([positions, orientations], dim=-1), env_ids=env_ids
        )
        return

    # 浮动基座实体。
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()

    positions = (
        root_states[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
    )
    orientations_delta = quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    # 速度。
    if velocity_range is None:
        velocity_range = {}
    range_list = [
        velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=env.device)
    vel_samples = sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
    )
    velocities = root_states[:, 7:13] + vel_samples

    asset.write_root_link_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )

    asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_from_flat_patches(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    patch_name: str = "spawn",
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """通过将资产放置在随机选择的平坦地块上来重置根状态。

    为每个环境从地形中随机选择一个平坦地块，并将资产定位在那里。
    如果地形没有平坦地块，则回退到 ``reset_root_state_uniform``。

    Args:
      env: 环境实例。
      env_ids: 要重置的环境 ID。若为 None，则重置所有环境。
      patch_name: 要使用的 ``terrain.flat_patches`` 中的键。
      pose_range: 在地块位置上叠加的可选随机偏移。
        键：``{"x", "y", "z", "roll", "pitch", "yaw"}``。
      velocity_range: 可选的速度范围（仅浮动基座）。
      asset_cfg: 资产配置。
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    terrain = env.scene.terrain
    if terrain is None or patch_name not in terrain.flat_patches:
        reset_root_state_uniform(
            env,
            env_ids,
            pose_range=pose_range or {},
            velocity_range=velocity_range,
            asset_cfg=asset_cfg,
        )
        return

    patches = terrain.flat_patches[patch_name]  # (num_rows, num_cols, num_patches, 3)
    num_patches = patches.shape[2]

    # 查找每个环境的地形等级（行）和类型（列）。
    levels = terrain.terrain_levels[env_ids]
    types = terrain.terrain_types[env_ids]

    # 为每个环境随机选择一个地块索引。
    patch_ids = torch.randint(0, num_patches, (len(env_ids),), device=env.device)
    positions = patches[levels, types, patch_ids]

    asset: Entity = env.scene[asset_cfg.name]
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()

    # 应用可选的姿态范围偏移。
    if pose_range is None:
        pose_range = {}
    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=env.device)
    pose_samples = sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
    )

    # 位置：平坦地块位置 + 可选偏移。使用地块 z 坐标而非默认值。
    final_positions = positions.clone()
    final_positions[:, 0] += pose_samples[:, 0]
    final_positions[:, 1] += pose_samples[:, 1]
    final_positions[:, 2] += root_states[:, 2] + pose_samples[:, 2]

    orientations_delta = quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    if asset.is_fixed_base:
        if not asset.is_mocap:
            raise ValueError(
                f"Cannot reset root state for fixed-base non-mocap entity '{asset_cfg.name}'."
            )
        asset.write_mocap_pose_to_sim(
            torch.cat([final_positions, orientations], dim=-1), env_ids=env_ids
        )
        return

    # 速度。
    if velocity_range is None:
        velocity_range = {}
    vel_range_list = [
        velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    vel_ranges = torch.tensor(vel_range_list, device=env.device)
    vel_samples = sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=env.device
    )
    velocities = root_states[:, 7:13] + vel_samples

    asset.write_root_link_pose_to_sim(
        torch.cat([final_positions, orientations], dim=-1), env_ids=env_ids
    )
    asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_offset(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits
    assert soft_joint_pos_limits is not None

    joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    joint_pos += sample_uniform(*position_range, joint_pos.shape, env.device)
    joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
    joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)

    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=joint_ids,
    )


def push_by_setting_velocity(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    asset: Entity = env.scene[asset_cfg.name]
    vel_w = asset.data.root_link_vel_w[env_ids]
    range_list = [
        velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=env.device)
    vel_w += sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)
    asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


def apply_external_force_torque(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    asset: Entity = env.scene[asset_cfg.name]
    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
    )
    size = (len(env_ids), num_bodies, 3)
    forces = sample_uniform(*force_range, size, env.device)
    torques = sample_uniform(*torque_range, size, env.device)
    asset.write_external_wrench_to_sim(
        forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids
    )


class apply_body_impulse:
    """对物体施加随机冲量，持续一段随机采样的时长。

    模拟瞬态外部扰动，如碰撞、阵风或与不可见物体的接触。
    对选定物体施加恒定的力/力矩，持续一段随机采样的时长，
    然后在下一次冲量之前进入一段静默冷却期。

    **单次冲量的生命周期：**

    1. **冷却。** 事件在 ``cooldown_s`` 中随机采样的时长内处于空闲状态。
       不施加任何力。
    2. **触发。** 从 ``force_range`` 中按分量均匀采样力向量，
       并写入选定物体的 ``xfrc_applied``。
    3. **持续。** 力保持恒定，持续一段从 ``duration_s`` 中随机采样的时长。
    4. **结束。** 力归零，冷却期从第 1 步重新开始。

    每个环境运行自己独立的计时器，因此冲量在整个批次中互不相关。

    **施力点。** 默认情况下，力作用于每个物体的质心。
    ``body_point_offset`` 在物体局部坐标系中偏移施力点，例如
    ``(0, 0, 0.1)`` 表示在质心上方 10 cm 处。该偏移通过叉积
    ``offset x force`` 产生额外的力矩，使物体发生倾斜而非单纯平移。
    这类似于选择在物体的哪个位置施加外部推力。

    配合 ``mode="step"`` 使用。
    """

    @dataclass
    class VizCfg:
        """活跃冲量力的箭头可视化设置。"""

        rgba: tuple[float, float, float, float] = (0.9, 0.2, 0.8, 0.9)
        """箭头颜色（RGBA）。"""
        scale: float = 0.005
        """每牛顿力对应的箭头长度（米）。"""
        width: float = 0.015
        """箭头杆宽度（米）。"""
        min_force: float = 1.0
        """低于此值（N）时不显示箭头的最小力大小。"""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        self._body_ids = cfg.params["asset_cfg"].body_ids
        self._num_envs = env.num_envs
        self._device = env.device
        self._step_dt = env.step_dt
        self._viz_cfg: apply_body_impulse.VizCfg = cfg.params.get(
            "viz_cfg", apply_body_impulse.VizCfg()
        )
        offset = cfg.params.get("body_point_offset", None)
        self._body_point_offset: torch.Tensor | None = (
            torch.tensor(offset, device=self._device, dtype=torch.float32)
            if offset is not None
            else None
        )

        self._num_bodies = (
            len(self._body_ids)
            if isinstance(self._body_ids, list)
            else self._asset.num_bodies
        )

        self._cooldown_s: tuple[float, float] = cfg.params["cooldown_s"]
        self._time_remaining = torch.zeros(self._num_envs, device=self._device)
        self._active = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.bool
        )
        # 预采样初始冷却时间，使第一次冲量之前有一段冷却期，
        # 而不是在 t=0 时立即触发。
        self._interval_time_left = self._sample_cooldown(self._num_envs)

    def _sample_cooldown(self, n: int) -> torch.Tensor:
        low, high = self._cooldown_s
        return sample_uniform(low, high, n, self._device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        force_range: tuple[float, float],
        torque_range: tuple[float, float],
        duration_s: tuple[float, float],
        cooldown_s: tuple[float, float],
        asset_cfg: SceneEntityCfg,
        body_point_offset: tuple[float, float, float] | None = None,
    ) -> None:
        """推进冲量状态：结束旧冲量，触发新冲量。

        Args:
          env: 环境实例。
          env_ids: 未使用（step 事件始终作用于所有环境）。
          force_range: 每个力分量（N）的 ``(min, max)`` 均匀采样范围。
          torque_range: 每个力矩分量（Nm）的 ``(min, max)`` 均匀采样范围。
          duration_s: 冲量持续时间（秒）的 ``(min, max)`` 均匀采样范围。
          cooldown_s: 连续两次冲量之间的冷却时间（秒）的 ``(min, max)`` 均匀采样范围。
            在初始化时捕获，使第一次冲量之前有一段采样的冷却期；
            此处传入的 kwarg 不会被使用。
          asset_cfg: 实体与物体选择。配置中的 ``body_ids`` 用于选择哪些物体受力。
          body_point_offset: 可选 ``(x, y, z)`` 偏移量（在物体局部坐标系中），
            指定力的施力点。通过 ``cross(offset, force)`` 产生额外力矩。
        """
        del env, env_ids, asset_cfg, cooldown_s  # 调用时未使用。
        dt = self._step_dt

        # 对活跃环境的计时器进行递减。
        self._time_remaining[self._active] -= dt

        # 清除已结束的冲量并重新采样其间隔计时器。
        expired = self._active & (self._time_remaining <= 0)
        if expired.any():
            expired_ids = expired.nonzero(as_tuple=False).squeeze(-1)
            zeros = torch.zeros(
                (len(expired_ids), self._num_bodies, 3), device=self._device
            )
            self._asset.write_external_wrench_to_sim(
                zeros, zeros, env_ids=expired_ids, body_ids=self._body_ids
            )
            self._active[expired_ids] = False
            self._time_remaining[expired_ids] = 0.0
            self._interval_time_left[expired_ids] = self._sample_cooldown(
                len(expired_ids)
            )

        # 递减间隔计时器。
        self._interval_time_left -= dt

        # 为符合条件的环境触发新冲量。
        eligible = (~self._active) & (self._interval_time_left <= 0)
        if not eligible.any():
            return

        trigger_ids = eligible.nonzero(as_tuple=False).squeeze(-1)
        n = len(trigger_ids)

        # 采样力和力矩。
        size = (n, self._num_bodies, 3)
        forces = sample_uniform(*force_range, size, self._device)
        torques = sample_uniform(*torque_range, size, self._device)

        # 针对偏离质心的施力点调整力矩。
        if body_point_offset is not None:
            offset_local = torch.tensor(
                body_point_offset, device=self._device, dtype=torch.float32
            )
            body_quat = self._asset.data.body_com_quat_w[trigger_ids][:, self._body_ids]
            # 将偏移旋转到世界坐标系：(n, num_bodies, 3)。
            offset_w = quat_apply(
                body_quat.reshape(-1, 4), offset_local.expand(n * self._num_bodies, 3)
            ).reshape(n, self._num_bodies, 3)
            torques = torques + torch.cross(offset_w, forces, dim=-1)

        self._asset.write_external_wrench_to_sim(
            forces, torques, env_ids=trigger_ids, body_ids=self._body_ids
        )

        # 采样持续时间并设置计时器。
        dur_low, dur_high = duration_s
        self._time_remaining[trigger_ids] = (
            torch.rand(n, device=self._device) * (dur_high - dur_low) + dur_low
        )
        self._active[trigger_ids] = True

        # 重新采样间隔计时器。
        self._interval_time_left[trigger_ids] = self._sample_cooldown(n)

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        """为活跃的冲量力绘制箭头。"""
        if not self._active.any():
            return
        viz = self._viz_cfg
        min_sq = viz.min_force * viz.min_force
        wrench = self._asset.data.body_external_wrench  # (nworld, nbody, 6)
        com_pos = self._asset.data.body_com_pos_w  # (nworld, nbody, 3)
        offset = self._body_point_offset
        com_quat = self._asset.data.body_com_quat_w if offset is not None else None
        for env_idx in visualizer.get_env_indices(self._num_envs):
            if not self._active[env_idx]:
                continue
            for i in range(wrench.shape[1]):
                force = wrench[env_idx, i, :3]
                if (force * force).sum().item() < min_sq:
                    continue
                force_np = force.cpu().numpy()
                start_np = com_pos[env_idx, i].cpu().numpy()
                if offset is not None and com_quat is not None:
                    offset_w = quat_apply(com_quat[env_idx, i], offset)
                    start_np = start_np + offset_w.cpu().numpy()
                end_np = start_np + force_np * viz.scale
                visualizer.add_arrow(
                    start=start_np,
                    end=end_np,
                    color=viz.rgba,
                    width=viz.width,
                )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        if self._active[env_ids].any():
            if isinstance(env_ids, slice):
                active_ids = self._active.nonzero(as_tuple=False).squeeze(-1)
            else:
                active_ids = env_ids[self._active[env_ids]]
            if len(active_ids) > 0:
                zeros = torch.zeros(
                    (len(active_ids), self._num_bodies, 3),
                    device=self._device,
                )
                self._asset.write_external_wrench_to_sim(
                    zeros, zeros, env_ids=active_ids, body_ids=self._body_ids
                )

        n = self._num_envs if isinstance(env_ids, slice) else len(env_ids)
        self._time_remaining[env_ids] = 0.0
        self._interval_time_left[env_ids] = self._sample_cooldown(n)
        self._active[env_ids] = False
