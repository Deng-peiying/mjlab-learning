"""运球任务专属奖励函数。对齐 DribbleBot 论文 Table III 的五项球相关奖励。

所有奖励函数返回形状为 [B] 的张量（B = num_envs），范围 [0, 1]（越大越好）。
框架（RewardManager）将返回值乘以 weight 后累加到总奖励中。

论文中的总奖励公式：r_t = r_pos * exp(r_neg)，其中 r_pos 是正向任务奖励，
r_neg 是负向惩罚。我们简化为线性求和（mjlab 默认方式），通过调整 weight
来等价近似。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity  # 场景实体，可读取位置/速度/姿态等仿真数据
from mjlab.managers.scene_entity_config import SceneEntityCfg  # 指定要操作的实体、关节、刚体
from mjlab.utils.lab_api.math import quat_apply_inverse  # 世界坐标系向量 → 机体坐标系

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ======================================================================
# 奖励 1: 球速跟踪 (Projected Ball Velocity, 论文权重 0.5)
# ======================================================================
def track_ball_velocity(
    env: ManagerBasedRlEnv,
    std: float,  # 高斯核标准差，越小越严格
    command_name: str,  # 指令名称，配置中指定为 "ball_vel"
    ball_name: str = "ball",
) -> torch.Tensor:
    """奖励世界坐标系中球速度与指令速度匹配。

    公式: exp(-||v_cmd - v_ball||² / σ²)

    - 指令球速来自 BallVelocityCommand，在世界坐标系中采样 (vx, vy)
    - 实际球速从 MuJoCo 仿真读取 ball.data.root_link_lin_vel_w
    - 两者都在世界坐标系中，不需要坐标变换
    - 误差越小 → exp 值越接近 1 → 奖励越大

    std = sqrt(0.25) = 0.5，即误差 0.5 m/s 时奖励 = exp(-1) ≈ 0.37
    """
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    ball: Entity = env.scene[ball_name]

    # 球在世界坐标系的实际速度（只取水平面 xy）
    ball_vel_w = ball.data.root_link_lin_vel_w[:, :2]  # [B, 2]

    # 指令与实际速度的 L2 平方误差
    error_sq = torch.sum(torch.square(command[:, :2] - ball_vel_w), dim=1)  # [B]

    # 高斯核: exp(-error / σ²)，误差=0 时 = 1.0
    return torch.exp(-error_sq / std**2)


# ======================================================================
# 奖励 2: 球距离 (Robot Ball Distance, 论文权重 4.0)
# ======================================================================
def robot_ball_distance(
    env: ManagerBasedRlEnv,
    std: float,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励球靠近两前脚掌（取最小距离）。

    公式: exp(-min(||ball_b - FR_foot_b||², ||ball_b - FL_foot_b||²) / σ²)

    为什么改成前脚掌而不是髋关节？
    - 原版用 FR 髋关节 (0.17, -0.09, 0) → 球被"吸"到身体正下方
    - 前脚掌在 stance 时位于身体前下方 (~0.35, ±0.13, -0.25)
    - 用脚掌位置衡量 → 策略必须把球保持在脚能碰到的地方
    - 取两脚最小距离 → 不偏袒右脚，策略可以学会交替用脚

    计算流程:
    1. 球 + 两前脚掌位置转到机体坐标系
    2. 计算球到 FR/FL 脚掌的 L2 距离平方
    3. 取 min → 高斯奖励
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    # 球和机器人在世界坐标系的位置
    ball_pos_w = ball.data.root_link_pos_w    # [B, 3]
    robot_pos_w = robot.data.root_link_pos_w  # [B, 3]
    robot_quat_w = robot.data.root_link_quat_w  # [B, 4]

    # 球在机体坐标系中的位置
    ball_pos_b = quat_apply_inverse(robot_quat_w, ball_pos_w - robot_pos_w)  # [B, 3]

    # 查找前脚掌 site (FR, FL) 的索引
    site_short_names = [s.name.split("/")[-1] for s in robot.indexing.sites]
    fr_idx = site_short_names.index("FR")
    fl_idx = site_short_names.index("FL")

    # 获取前脚掌在世界坐标系的位置 → 转到机体坐标系
    foot_pos_w = robot.data.site_pos_w  # [B, num_sites, 3]
    fr_foot_b = quat_apply_inverse(
        robot_quat_w, foot_pos_w[:, fr_idx, :] - robot_pos_w
    )  # [B, 3]
    fl_foot_b = quat_apply_inverse(
        robot_quat_w, foot_pos_w[:, fl_idx, :] - robot_pos_w
    )  # [B, 3]

    # 球到两前脚掌的距离平方，取 min（最近的那只脚）
    dist_sq_fr = torch.sum(torch.square(ball_pos_b - fr_foot_b), dim=1)  # [B]
    dist_sq_fl = torch.sum(torch.square(ball_pos_b - fl_foot_b), dim=1)  # [B]
    min_dist_sq = torch.min(dist_sq_fr, dist_sq_fl)  # [B]

    return torch.exp(-min_dist_sq / std**2)


# ======================================================================
# 奖励 3: 偏航对齐 (Yaw Alignment, 论文权重 4.0)
# ======================================================================
def robot_ball_yaw(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励机器人面朝球、并且把球推向指令方向。

    综合两个角度误差（用点积的补数近似角度差的余弦）:

    误差 1: 机器人→球 的方向 vs 指令方向
        - 鼓励机器人站在球的"上游"，把球往正确方向推
        - 向量: 机器人→球  vs  指令速度向量
        - 误差 = 1.0 - cos(θ₁) = 1.0 - (d_robot_ball · unit_cmd)

    误差 2: 机器人朝向 vs 机器人→球 的方向
        - 鼓励机器人转身面朝球
        - 向量: 机身 heading  vs  机器人→球
        - 误差 = 1.0 - cos(θ₂) = 1.0 - (d_robot_ball · body_yaw)

    总误差 = 误差1 + 误差2，范围 [0, 4]（两个 cos 都不对齐时最差）
    奖励 = exp(-total_error / σ²)

    +e-6 防止除零（command 为零时指令方向无定义）
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    # 世界坐标系中从机器人到球的水平方向向量
    robot_ball_vec_w = (
        ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
    )
    d_robot_ball = robot_ball_vec_w / (
        torch.norm(robot_ball_vec_w, dim=-1, keepdim=True) + 1e-6
    )

    # 误差 1: 机器人→球方向 vs 指令方向
    cmd_vel = command[:, :2]
    unit_cmd = cmd_vel / (torch.norm(cmd_vel, dim=-1, keepdim=True) + 1e-6)
    cmd_yaw_error = 1.0 - torch.sum(d_robot_ball * unit_cmd, dim=-1)  # [B], 范围 [0, 2]

    # 误差 2: 机身朝向 vs 机器人→球方向
    heading = robot.data.heading_w  # [B], 机器人在世界坐标系中的偏航角
    body_yaw_vec = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)  # [B, 2]
    body_yaw_error = 1.0 - torch.sum(d_robot_ball * body_yaw_vec, dim=-1)  # [B], 范围 [0, 2]

    total_error = cmd_yaw_error + body_yaw_error
    return torch.exp(-total_error / std**2)


# ======================================================================
# 奖励 4: 球速范数 (Ball Velocity Norm, 论文权重 4.0)
# ======================================================================
def ball_vel_norm(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    ball_name: str = "ball",
) -> torch.Tensor:
    """奖励球速度大小（速率）与指令速率匹配，不考虑方向。

    公式: exp(-(||v_cmd|| - ||v_ball||)² / σ²)

    - 只比较速度的绝对值（标量速率），不关心方向
    - 和 track_ball_velocity（向量级别的速度匹配）互补
    - 跟踪球速方向的任务由 ball_vel_angle 负责

    例如: 指令要球以 1 m/s 向北，球实际以 1 m/s 向东
      → track_ball_velocity 会产生惩罚（方向错了）
      → ball_vel_norm 奖励接近 1（速率对了）
      → ball_vel_angle 会产生惩罚（方向错了）
    """
    command = env.command_manager.get_command(command_name)
    assert command is not None
    ball: Entity = env.scene[ball_name]
    ball_vel_w = ball.data.root_link_lin_vel_w[:, :2]

    # 指令速率 (标量) 和 实际速率 (标量)
    cmd_norm = torch.norm(command[:, :2], dim=-1)    # [B]
    ball_norm = torch.norm(ball_vel_w, dim=-1)        # [B]

    # 速率差平方
    norm_diff_sq = torch.square(cmd_norm - ball_norm)  # [B]

    return torch.exp(-norm_diff_sq / std**2)


# ======================================================================
# 奖励 5: 球速方向 (Ball Velocity Angle, 论文权重 4.0)
# ======================================================================
def ball_vel_angle(
    env: ManagerBasedRlEnv,
    command_name: str,
    ball_name: str = "ball",
) -> torch.Tensor:
    """奖励球速度方向与指令方向对齐。

    公式: 1 - (Δψ)² / π²，范围 [0, 1]

    - Δψ 是指令方向和实际球速度方向的差值，wrap 到 [-π, π]
    - 方向完全一致 → Δψ=0 → 奖励=1.0
    - 方向完全相反 → Δψ=π → 奖励=1-(π²/π²)=0.0

    与 track_ball_velocity 的区别:
    - track_ball_velocity: 奖励速度向量完全匹配（方向和大小同时）
    - ball_vel_norm: 只奖励速度大小匹配
    - ball_vel_angle: 只奖励速度方向匹配
    三项组合起来 = 速度向量完全匹配（分解成大小 + 方向更容易优化）

    这里不是高斯型奖励，因为论文原文公式就是 1-(Δψ)²/π²。
    """
    command = env.command_manager.get_command(command_name)
    assert command is not None
    ball: Entity = env.scene[ball_name]
    ball_vel_w = ball.data.root_link_lin_vel_w[:, :2]

    # 用 arctan2 计算向量方向角
    cmd_angle = torch.atan2(command[:, 1], command[:, 0])      # [B]
    ball_angle = torch.atan2(ball_vel_w[:, 1], ball_vel_w[:, 0])  # [B]

    # 角度差 → wrap 到 [-π, π]（用 arctan2(sinΔ, cosΔ) 技巧）
    angle_diff = cmd_angle - ball_angle
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

    return 1.0 - torch.square(angle_diff) / (torch.pi**2)


# ======================================================================
# 负向惩罚: 球在身体正下方 (Ball Under Body)
# ======================================================================
def ball_under_body(
    env: ManagerBasedRlEnv,
    std: float,
    ball_name: str = "ball",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    base_x_range: tuple[float, float] = (-0.20, 0.30),
    base_y_range: tuple[float, float] = (-0.15, 0.15),
) -> torch.Tensor:
    """惩罚球进入机器人 base 在 xy 平面的矩形投影区域。

    公式: exp(-d_out² / σ²)，其中 d_out 是球到矩形区域的最短距离

    - 球在矩形内 → d_out = 0 → 惩罚 = 1.0（最大）
    - 球在矩形外 → d_out > 0 → 惩罚随距离衰减

    base_x_range / base_y_range 定义了 base 在机体 xy 平面的投影范围，
    球进入这个矩形就会被惩罚，离开矩形后按高斯衰减。

    此函数作为 r_neg 的一项，通过乘法结构 r_pos * exp(w * r_neg)
    在球被"吸入"身体下方时剧烈压制总奖励。
    """
    robot: Entity = env.scene[asset_cfg.name]
    ball: Entity = env.scene[ball_name]

    ball_pos_w = ball.data.root_link_pos_w
    robot_pos_w = robot.data.root_link_pos_w
    robot_quat_w = robot.data.root_link_quat_w

    # 球在机体坐标系中的位置
    ball_pos_b = quat_apply_inverse(robot_quat_w, ball_pos_w - robot_pos_w)  # [B, 3]
    ball_x = ball_pos_b[:, 0]  # [B]
    ball_y = ball_pos_b[:, 1]  # [B]

    # 球在每条轴向上超出矩形边界多远（在矩形内则为 0）
    x_min, x_max = base_x_range
    y_min, y_max = base_y_range
    dx_out = torch.clamp(x_min - ball_x, min=0) + torch.clamp(ball_x - x_max, min=0)  # [B]
    dy_out = torch.clamp(y_min - ball_y, min=0) + torch.clamp(ball_y - y_max, min=0)  # [B]
    dist_out_sq = dx_out**2 + dy_out**2  # [B]

    return torch.exp(-dist_out_sq / std**2)
