from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import mujoco_warp as mjwarp
import torch

from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_apply_inverse,
    quat_from_matrix,
    quat_mul,
)

if TYPE_CHECKING:
    from mjlab.entity.entity import EntityIndexing


def compute_velocity_from_cvel(
    pos: torch.Tensor,
    subtree_com: torch.Tensor,
    cvel: torch.Tensor,
) -> torch.Tensor:
    """将 cvel 量转换为世界坐标系下的速度。"""
    lin_vel_c = cvel[..., 3:6]
    ang_vel_c = cvel[..., 0:3]
    offset = subtree_com - pos
    lin_vel_w = lin_vel_c - torch.cross(ang_vel_c, offset, dim=-1)
    ang_vel_w = ang_vel_c
    return torch.cat([lin_vel_w, ang_vel_w], dim=-1)


@dataclass
class EntityData:
    """实体的数据容器。

    注意：写入方法（write_*）直接修改状态。读取属性（如 root_link_pose_w）
    需要 sim.forward() 为最新状态。如果先写入后读取，请在中间调用 sim.forward()。
    混合读写时，事件顺序很重要。所有输入/输出均使用世界坐标系。
    """

    indexing: EntityIndexing
    data: mjwarp.Data
    model: mjwarp.Model
    device: str

    default_root_state: torch.Tensor
    default_joint_pos: torch.Tensor
    default_joint_vel: torch.Tensor

    default_joint_pos_limits: torch.Tensor
    joint_pos_limits: torch.Tensor
    soft_joint_pos_limits: torch.Tensor

    gravity_vec_w: torch.Tensor
    forward_vec_b: torch.Tensor

    is_fixed_base: bool
    is_articulated: bool
    is_actuated: bool

    joint_pos_target: torch.Tensor
    joint_vel_target: torch.Tensor
    joint_effort_target: torch.Tensor

    tendon_len_target: torch.Tensor
    tendon_vel_target: torch.Tensor
    tendon_effort_target: torch.Tensor

    site_effort_target: torch.Tensor

    encoder_bias: torch.Tensor

    # 状态维度。
    POS_DIM = 3
    QUAT_DIM = 4
    LIN_VEL_DIM = 3
    ANG_VEL_DIM = 3
    ROOT_POSE_DIM = POS_DIM + QUAT_DIM  # 7
    ROOT_VEL_DIM = LIN_VEL_DIM + ANG_VEL_DIM  # 6
    ROOT_STATE_DIM = ROOT_POSE_DIM + ROOT_VEL_DIM  # 13

    def write_root_state(
        self, root_state: torch.Tensor, env_ids: torch.Tensor | slice | None = None
    ) -> None:
        if self.is_fixed_base:
            raise ValueError("Cannot write root state for fixed-base entity.")
        assert root_state.shape[-1] == self.ROOT_STATE_DIM

        self.write_root_pose(root_state[:, : self.ROOT_POSE_DIM], env_ids)
        self.write_root_velocity(root_state[:, self.ROOT_POSE_DIM :], env_ids)

    def write_root_pose(
        self, pose: torch.Tensor, env_ids: torch.Tensor | slice | None = None
    ) -> None:
        if self.is_fixed_base:
            raise ValueError("Cannot write root pose for fixed-base entity.")
        assert pose.shape[-1] == self.ROOT_POSE_DIM

        env_ids = self._resolve_env_ids(env_ids)
        self.data.qpos[env_ids, self.indexing.free_joint_q_adr] = pose

    def write_root_velocity(
        self, velocity: torch.Tensor, env_ids: torch.Tensor | slice | None = None
    ) -> None:
        if self.is_fixed_base:
            raise ValueError("Cannot write root velocity for fixed-base entity.")
        assert velocity.shape[-1] == self.ROOT_VEL_DIM

        env_ids = self._resolve_env_ids(env_ids)
        quat_w = self.data.qpos[env_ids, self.indexing.free_joint_q_adr[3:7]]
        ang_vel_b = quat_apply_inverse(quat_w, velocity[:, 3:])
        velocity_qvel = torch.cat([velocity[:, :3], ang_vel_b], dim=-1)
        self.data.qvel[env_ids, self.indexing.free_joint_v_adr] = velocity_qvel

    def write_root_com_velocity(
        self, velocity: torch.Tensor, env_ids: torch.Tensor | slice | None = None
    ) -> None:
        if self.is_fixed_base:
            raise ValueError("Cannot write root COM velocity for fixed-base entity.")
        assert velocity.shape[-1] == self.ROOT_VEL_DIM

        env_ids = env_ids if env_ids is not None else slice(None)
        com_offset_b = self.model.body_ipos[:, self.indexing.root_body_id]
        quat_w = self.data.qpos[:, self.indexing.free_joint_q_adr[3:7]][env_ids]
        com_offset_w = quat_apply(quat_w, com_offset_b[env_ids])
        lin_vel_com = velocity[:, :3]
        ang_vel_w = velocity[:, 3:]
        lin_vel_link = lin_vel_com - torch.cross(ang_vel_w, com_offset_w, dim=-1)
        link_velocity = torch.cat([lin_vel_link, ang_vel_w], dim=-1)
        self.write_root_velocity(link_velocity, env_ids)

    def write_joint_state(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        if not self.is_articulated:
            raise ValueError("Cannot write joint state for non-articulated entity.")

        self.write_joint_position(position, joint_ids, env_ids)
        self.write_joint_velocity(velocity, joint_ids, env_ids)

    def write_joint_position(
        self,
        position: torch.Tensor,
        joint_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        if not self.is_articulated:
            raise ValueError("Cannot write joint position for non-articulated entity.")

        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = joint_ids if joint_ids is not None else slice(None)
        q_slice = self.indexing.joint_q_adr[joint_ids]
        self.data.qpos[env_ids, q_slice] = position

    def write_joint_velocity(
        self,
        velocity: torch.Tensor,
        joint_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        if not self.is_articulated:
            raise ValueError("Cannot write joint velocity for non-articulated entity.")

        env_ids = self._resolve_env_ids(env_ids)
        joint_ids = joint_ids if joint_ids is not None else slice(None)
        v_slice = self.indexing.joint_v_adr[joint_ids]
        self.data.qvel[env_ids, v_slice] = velocity

    def write_external_wrench(
        self,
        force: torch.Tensor | None,
        torque: torch.Tensor | None,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        local_body_ids = body_ids if body_ids is not None else slice(None)
        global_body_ids = self.indexing.body_ids[local_body_ids]
        if force is not None:
            self.data.xfrc_applied[env_ids, global_body_ids, 0:3] = force
        if torque is not None:
            self.data.xfrc_applied[env_ids, global_body_ids, 3:6] = torque

    def write_ctrl(
        self,
        ctrl: torch.Tensor,
        ctrl_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        if not self.is_actuated:
            raise ValueError("Cannot write control for non-actuated entity.")

        env_ids = self._resolve_env_ids(env_ids)
        local_ctrl_ids = ctrl_ids if ctrl_ids is not None else slice(None)
        global_ctrl_ids = self.indexing.ctrl_ids[local_ctrl_ids]
        self.data.ctrl[env_ids, global_ctrl_ids] = ctrl

    def write_mocap_pose(
        self, pose: torch.Tensor, env_ids: torch.Tensor | slice | None = None
    ) -> None:
        if self.indexing.mocap_id is None:
            raise ValueError("Cannot write mocap pose for non-mocap entity.")
        assert pose.shape[-1] == self.ROOT_POSE_DIM

        env_ids = self._resolve_env_ids(env_ids)
        self.data.mocap_pos[env_ids, self.indexing.mocap_id] = pose[:, 0:3].unsqueeze(1)
        self.data.mocap_quat[env_ids, self.indexing.mocap_id] = pose[:, 3:7].unsqueeze(
            1
        )

    def clear_state(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self.is_actuated:
            env_ids = self._resolve_env_ids(env_ids)
            self.joint_pos_target[env_ids] = 0.0
            self.joint_vel_target[env_ids] = 0.0
            self.joint_effort_target[env_ids] = 0.0
            self.tendon_len_target[env_ids] = 0.0
            self.tendon_vel_target[env_ids] = 0.0
            self.tendon_effort_target[env_ids] = 0.0
            self.site_effort_target[env_ids] = 0.0

    def _resolve_env_ids(
        self, env_ids: torch.Tensor | slice | None
    ) -> torch.Tensor | slice:
        """将 env_ids 转换为统一的索引格式。"""
        if env_ids is None:
            return slice(None)
        if isinstance(env_ids, torch.Tensor):
            return env_ids[:, None]
        return env_ids

    def _joint_dof_field(self, field_name: str) -> torch.Tensor:
        """返回按实体关节自由度（DoF）切片后的广义力字段。"""
        field = getattr(self.data, field_name)
        return field[:, self.indexing.joint_v_adr]

    # 根属性

    @property
    def root_link_pose_w(self) -> torch.Tensor:
        """根刚体在世界坐标系下的姿态。形状 (num_envs, 7)。"""
        pos_w = self.data.xpos[:, self.indexing.root_body_id]  # (num_envs, 3)
        quat_w = self.data.xquat[:, self.indexing.root_body_id]  # (num_envs, 4)
        return torch.cat([pos_w, quat_w], dim=-1)  # (num_envs, 7)

    @property
    def root_link_vel_w(self) -> torch.Tensor:
        """根刚体在世界坐标系下的速度。形状 (num_envs, 6)。"""
        # NOTE：等效地，可以从 qvel[:6] 读取，但角速度部分将在物体坐标系中，
        # 需要旋转到世界坐标系。另外可能还需要额外的 forward() 调用才能使
        # 两个值相等。
        pos = self.data.xpos[:, self.indexing.root_body_id]  # (num_envs, 3)
        subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
        cvel = self.data.cvel[:, self.indexing.root_body_id]  # (num_envs, 6)
        return compute_velocity_from_cvel(pos, subtree_com, cvel)  # (num_envs, 6)

    @property
    def root_com_pose_w(self) -> torch.Tensor:
        """根质心在世界坐标系下的姿态。形状 (num_envs, 7)。"""
        pos_w = self.data.xipos[:, self.indexing.root_body_id]
        quat = self.data.xquat[:, self.indexing.root_body_id]
        body_iquat = self.model.body_iquat[:, self.indexing.root_body_id]
        assert body_iquat is not None
        quat_w = quat_mul(quat, body_iquat.squeeze(1))
        return torch.cat([pos_w, quat_w], dim=-1)

    @property
    def root_com_vel_w(self) -> torch.Tensor:
        """根质心在世界坐标系下的速度。形状 (num_envs, 6)。"""
        # NOTE：等效传感器是 objtype="body" 的 framelinvel/frameangvel。
        pos = self.data.xipos[:, self.indexing.root_body_id]  # (num_envs, 3)
        subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
        cvel = self.data.cvel[:, self.indexing.root_body_id]  # (num_envs, 6)
        return compute_velocity_from_cvel(pos, subtree_com, cvel)  # (num_envs, 6)

    # 物体属性

    @property
    def body_link_pose_w(self) -> torch.Tensor:
        """物体刚体在世界坐标系下的姿态。形状 (num_envs, num_bodies, 7)。"""
        pos_w = self.data.xpos[:, self.indexing.body_ids]
        quat_w = self.data.xquat[:, self.indexing.body_ids]
        return torch.cat([pos_w, quat_w], dim=-1)

    @property
    def body_link_vel_w(self) -> torch.Tensor:
        """物体刚体在世界坐标系下的速度。形状 (num_envs, num_bodies, 6)。"""
        # NOTE：等效传感器是 objtype="xbody" 的 framelinvel/frameangvel。
        pos = self.data.xpos[:, self.indexing.body_ids]  # (num_envs, num_bodies, 3)
        subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
        cvel = self.data.cvel[:, self.indexing.body_ids]
        return compute_velocity_from_cvel(pos, subtree_com.unsqueeze(1), cvel)

    @property
    def body_com_pose_w(self) -> torch.Tensor:
        """物体质心在世界坐标系下的姿态。形状 (num_envs, num_bodies, 7)。"""
        pos_w = self.data.xipos[:, self.indexing.body_ids]
        quat = self.data.xquat[:, self.indexing.body_ids]
        body_iquat = self.model.body_iquat[:, self.indexing.body_ids]
        quat_w = quat_mul(quat, body_iquat)
        return torch.cat([pos_w, quat_w], dim=-1)

    @property
    def body_com_vel_w(self) -> torch.Tensor:
        """物体质心在世界坐标系下的速度。形状 (num_envs, num_bodies, 6)。"""
        # NOTE：等效传感器是 objtype="body" 的 framelinvel/frameangvel。
        pos = self.data.xipos[:, self.indexing.body_ids]
        subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
        cvel = self.data.cvel[:, self.indexing.body_ids]
        return compute_velocity_from_cvel(pos, subtree_com.unsqueeze(1), cvel)

    @property
    def body_external_wrench(self) -> torch.Tensor:
        """物体在世界坐标系下的外部力/力矩。形状 (num_envs, num_bodies, 6)。"""
        return self.data.xfrc_applied[:, self.indexing.body_ids]

    # 几何体属性

    @property
    def geom_pose_w(self) -> torch.Tensor:
        """几何体在世界坐标系下的姿态。形状 (num_envs, num_geoms, 7)。"""
        pos_w = self.data.geom_xpos[:, self.indexing.geom_ids]
        xmat = self.data.geom_xmat[:, self.indexing.geom_ids]
        quat_w = quat_from_matrix(xmat)
        return torch.cat([pos_w, quat_w], dim=-1)

    @property
    def geom_vel_w(self) -> torch.Tensor:
        """几何体在世界坐标系下的速度。形状 (num_envs, num_geoms, 6)。"""
        pos = self.data.geom_xpos[:, self.indexing.geom_ids]
        body_ids = self.model.geom_bodyid[self.indexing.geom_ids]  # (num_geoms,)
        subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
        cvel = self.data.cvel[:, body_ids]
        return compute_velocity_from_cvel(pos, subtree_com.unsqueeze(1), cvel)

    # 站点属性

    @property
    def site_pose_w(self) -> torch.Tensor:
        """站点在世界坐标系下的姿态。形状 (num_envs, num_sites, 7)。"""
        pos_w = self.data.site_xpos[:, self.indexing.site_ids]
        mat_w = self.data.site_xmat[:, self.indexing.site_ids]
        quat_w = quat_from_matrix(mat_w)
        return torch.cat([pos_w, quat_w], dim=-1)

    @property
    def site_vel_w(self) -> torch.Tensor:
        """站点在世界坐标系下的速度。形状 (num_envs, num_sites, 6)。"""
        pos = self.data.site_xpos[:, self.indexing.site_ids]
        body_ids = self.model.site_bodyid[self.indexing.site_ids]  # (num_sites,)
        subtree_com = self.data.subtree_com[:, self.indexing.root_body_id]
        cvel = self.data.cvel[:, body_ids]
        return compute_velocity_from_cvel(pos, subtree_com.unsqueeze(1), cvel)

    # 关节属性

    @property
    def joint_pos(self) -> torch.Tensor:
        """关节位置。形状 (num_envs, num_joints)。"""
        return self.data.qpos[:, self.indexing.joint_q_adr]

    @property
    def joint_pos_biased(self) -> torch.Tensor:
        """带编码器偏置的关节位置。形状 (num_envs, num_joints)。"""
        return self.joint_pos + self.encoder_bias

    @property
    def joint_vel(self) -> torch.Tensor:
        """关节速度。形状 (num_envs, nv)。"""
        return self.data.qvel[:, self.indexing.joint_v_adr]

    @property
    def joint_acc(self) -> torch.Tensor:
        """关节加速度。形状 (num_envs, nv)。"""
        return self.data.qacc[:, self.indexing.joint_v_adr]

    # 腱属性

    @property
    def tendon_len(self) -> torch.Tensor:
        """腱长度。形状 (num_envs, num_tendons)。"""
        return self.data.ten_length[:, self.indexing.tendon_ids]

    @property
    def tendon_vel(self) -> torch.Tensor:
        """腱速度。形状 (num_envs, num_tendons)。"""
        return self.data.ten_velocity[:, self.indexing.tendon_ids]

    # 广义力

    @property
    def joint_torques(self) -> torch.Tensor:
        """关节力矩。形状 (num_envs, nv)。"""
        raise NotImplementedError(
            "Joint torques are ambiguous. Use 'qfrc_actuator' for actuator forces "
            "in joint space, or 'qfrc_external' for body wrench contributions."
        )

    @property
    def actuator_force(self) -> torch.Tensor:
        """驱动空间中的标量驱动器输出。形状 (num_envs, nu)。

        这与关节空间中的广义力不同。使用 ``qfrc_actuator`` 获取投影到
        自由度空间中的驱动器贡献。
        """
        return self.data.actuator_force[:, self.indexing.ctrl_ids]

    @property
    def qfrc_actuator(self) -> torch.Tensor:
        """所有驱动器产生的力，映射到关节空间。

        对于电机，这是指令力矩乘以传动比。对于位置和速度驱动器，
        这是内部 PD 定律计算的力。当关节启用 ``actuatorgravcomp`` 时，
        重力补偿力也包含在内。
        """
        return self._joint_dof_field("qfrc_actuator")

    @property
    def qfrc_external(self) -> torch.Tensor:
        """由于施加在物体上的笛卡尔力/力矩而在关节上产生的力。

        当通过 ``xfrc_applied`` 对物体施加力或力矩时，此属性给出等效的
        关节力（雅可比转置映射）。
        """
        # MuJoCo 将 J^T * xfrc_applied 合并到 qfrc_smooth 中而不单独存储。
        # 通过 qfrc_smooth 恒等式恢复：
        #   qfrc_smooth = qfrc_actuator + qfrc_passive - qfrc_bias
        #                 + qfrc_applied + J^T * xfrc_applied
        f = self._joint_dof_field
        return (
            f("qfrc_smooth")
            - f("qfrc_actuator")
            - f("qfrc_applied")
            - f("qfrc_passive")
            + f("qfrc_bias")
        )

    # 姿态和速度分量访问器。

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        """根刚体在世界坐标系下的位置。形状 (num_envs, 3)。"""
        return self.root_link_pose_w[:, 0:3]

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        """根刚体在世界坐标系下的四元数。形状 (num_envs, 4)。"""
        return self.root_link_pose_w[:, 3:7]

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        """根刚体在世界坐标系下的线速度。形状 (num_envs, 3)。"""
        return self.root_link_vel_w[:, 0:3]

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        """根刚体在世界坐标系下的角速度。形状 (num_envs, 3)。"""
        return self.data.cvel[:, self.indexing.root_body_id, 0:3]

    @property
    def root_com_pos_w(self) -> torch.Tensor:
        """根质心在世界坐标系下的位置。形状 (num_envs, 3)。"""
        return self.root_com_pose_w[:, 0:3]

    @property
    def root_com_quat_w(self) -> torch.Tensor:
        """根质心在世界坐标系下的四元数。形状 (num_envs, 4)。"""
        return self.root_com_pose_w[:, 3:7]

    @property
    def root_com_lin_vel_w(self) -> torch.Tensor:
        """根质心在世界坐标系下的线速度。形状 (num_envs, 3)。"""
        return self.root_com_vel_w[:, 0:3]

    @property
    def root_com_ang_vel_w(self) -> torch.Tensor:
        """根质心在世界坐标系下的角速度。形状 (num_envs, 3)。"""
        # 角速度对刚体坐标系和质心坐标系是相同的。
        return self.data.cvel[:, self.indexing.root_body_id, 0:3]

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        """物体刚体在世界坐标系下的位置。形状 (num_envs, num_bodies, 3)。"""
        return self.body_link_pose_w[..., 0:3]

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        """物体刚体在世界坐标系下的四元数。形状 (num_envs, num_bodies, 4)。"""
        return self.body_link_pose_w[..., 3:7]

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        """物体刚体在世界坐标系下的线速度。形状 (num_envs, num_bodies, 3)。"""
        return self.body_link_vel_w[..., 0:3]

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        """物体刚体在世界坐标系下的角速度。形状 (num_envs, num_bodies, 3)。"""
        return self.data.cvel[:, self.indexing.body_ids, 0:3]

    @property
    def body_com_pos_w(self) -> torch.Tensor:
        """物体质心在世界坐标系下的位置。形状 (num_envs, num_bodies, 3)。"""
        return self.body_com_pose_w[..., 0:3]

    @property
    def body_com_quat_w(self) -> torch.Tensor:
        """物体质心在世界坐标系下的四元数。形状 (num_envs, num_bodies, 4)。"""
        return self.body_com_pose_w[..., 3:7]

    @property
    def body_com_lin_vel_w(self) -> torch.Tensor:
        """物体质心在世界坐标系下的线速度。形状 (num_envs, num_bodies, 3)。"""
        return self.body_com_vel_w[..., 0:3]

    @property
    def body_com_ang_vel_w(self) -> torch.Tensor:
        """物体质心在世界坐标系下的角速度。形状 (num_envs, num_bodies, 3)。"""
        # 角速度对刚体坐标系和质心坐标系是相同的。
        return self.data.cvel[:, self.indexing.body_ids, 0:3]

    @property
    def body_external_force(self) -> torch.Tensor:
        """物体在世界坐标系下的外部力。形状 (num_envs, num_bodies, 3)。"""
        return self.body_external_wrench[..., 0:3]

    @property
    def body_external_torque(self) -> torch.Tensor:
        """物体在世界坐标系下的外部力矩。形状 (num_envs, num_bodies, 3)。"""
        return self.body_external_wrench[..., 3:6]

    @property
    def geom_pos_w(self) -> torch.Tensor:
        """几何体在世界坐标系下的位置。形状 (num_envs, num_geoms, 3)。"""
        return self.geom_pose_w[..., 0:3]

    @property
    def geom_quat_w(self) -> torch.Tensor:
        """几何体在世界坐标系下的四元数。形状 (num_envs, num_geoms, 4)。"""
        return self.geom_pose_w[..., 3:7]

    @property
    def geom_lin_vel_w(self) -> torch.Tensor:
        """几何体在世界坐标系下的线速度。形状 (num_envs, num_geoms, 3)。"""
        return self.geom_vel_w[..., 0:3]

    @property
    def geom_ang_vel_w(self) -> torch.Tensor:
        """几何体在世界坐标系下的角速度。形状 (num_envs, num_geoms, 3)。"""
        body_ids = self.model.geom_bodyid[self.indexing.geom_ids]
        return self.data.cvel[:, body_ids, 0:3]

    @property
    def site_pos_w(self) -> torch.Tensor:
        """站点在世界坐标系下的位置。形状 (num_envs, num_sites, 3)。"""
        return self.site_pose_w[..., 0:3]

    @property
    def site_quat_w(self) -> torch.Tensor:
        """站点在世界坐标系下的四元数。形状 (num_envs, num_sites, 4)。"""
        return self.site_pose_w[..., 3:7]

    @property
    def site_lin_vel_w(self) -> torch.Tensor:
        """站点在世界坐标系下的线速度。形状 (num_envs, num_sites, 3)。"""
        return self.site_vel_w[..., 0:3]

    @property
    def site_ang_vel_w(self) -> torch.Tensor:
        """站点在世界坐标系下的角速度。形状 (num_envs, num_sites, 3)。"""
        body_ids = self.model.site_bodyid[self.indexing.site_ids]
        return self.data.cvel[:, body_ids, 0:3]

    # 派生属性。

    @property
    def projected_gravity_b(self) -> torch.Tensor:
        """投影到物体坐标系的重力向量。形状 (num_envs, 3)。"""
        return quat_apply_inverse(self.root_link_quat_w, self.gravity_vec_w)

    @property
    def heading_w(self) -> torch.Tensor:
        """世界坐标系下的朝向角。形状 (num_envs,)。"""
        forward_w = quat_apply(self.root_link_quat_w, self.forward_vec_b)
        return torch.atan2(forward_w[:, 1], forward_w[:, 0])

    @property
    def root_link_lin_vel_b(self) -> torch.Tensor:
        """根刚体在物体坐标系下的线速度。形状 (num_envs, 3)。"""
        return quat_apply_inverse(self.root_link_quat_w, self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self) -> torch.Tensor:
        """根刚体在物体坐标系下的角速度。形状 (num_envs, 3)。"""
        return quat_apply_inverse(self.root_link_quat_w, self.root_link_ang_vel_w)

    @property
    def root_com_lin_vel_b(self) -> torch.Tensor:
        """根质心在物体坐标系下的线速度。形状 (num_envs, 3)。"""
        return quat_apply_inverse(self.root_link_quat_w, self.root_com_lin_vel_w)

    @property
    def root_com_ang_vel_b(self) -> torch.Tensor:
        """根质心在物体坐标系下的角速度。形状 (num_envs, 3)。"""
        return quat_apply_inverse(self.root_link_quat_w, self.root_com_ang_vel_w)
