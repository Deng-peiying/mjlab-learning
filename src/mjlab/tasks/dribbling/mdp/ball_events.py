"""Ball teleportation events. Aligned with DribbleBot paper Section VIII.

The paper teleports the ball to a random location within 1.0m every 7.0s.
We add a tighter safety net: if the ball exceeds 4m from the robot at any step,
it's immediately teleported back within 1m.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def teleport_ball_if_far(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    max_distance: float = 4.0,
    teleport_radius: float = 1.0,
    robot_name: str = "robot",
    ball_name: str = "ball",
) -> None:
    """Step event: if ball > max_distance from robot, teleport it back.

    Gaussian distance reward exp(-dist²/σ²) with σ²=0.25 is effectively zero
    beyond ~1.5m. When the ball rolls far, all reward signal vanishes and
    the policy stops moving. This event catches that case immediately.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    robot: Entity = env.scene[robot_name]
    ball: Entity = env.scene[ball_name]

    robot_xy = robot.data.root_link_pos_w[:, :2]
    ball_xy = ball.data.root_link_pos_w[:, :2]
    dist = torch.norm(ball_xy - robot_xy, dim=-1)

    far_mask = dist > max_distance
    far_ids = env_ids[far_mask[env_ids]]
    if len(far_ids) == 0:
        return

    # Random position within teleport_radius of robot (uniform in circle)
    r = teleport_radius * torch.sqrt(torch.rand(len(far_ids), device=env.device))
    theta = torch.rand(len(far_ids), device=env.device) * (2 * torch.pi)
    new_x = robot_xy[far_ids, 0] + r * torch.cos(theta)
    new_y = robot_xy[far_ids, 1] + r * torch.sin(theta)

    # Build new pose: preserve quaternion, update position
    ball_pos = ball.data.root_link_pos_w[far_ids].clone()
    ball_quat = ball.data.root_link_quat_w[far_ids].clone()
    ball_pos[:, 0] = new_x
    ball_pos[:, 1] = new_y
    ball_pos[:, 2] = 0.09  # ball radius above ground
    new_pose = torch.cat([ball_pos, ball_quat], dim=-1)

    # Zero velocity
    zero_vel = torch.zeros(len(far_ids), 6, device=env.device)

    ball.write_root_link_pose_to_sim(new_pose, env_ids=far_ids)
    ball.write_root_link_velocity_to_sim(zero_vel, env_ids=far_ids)

    # Clear stale xfrc_applied (e.g. drag computed from pre-teleport velocity).
    zero_w = torch.zeros(len(far_ids), 1, 3, device=env.device)
    ball.write_external_wrench_to_sim(zero_w, zero_w, env_ids=far_ids)


def periodic_ball_teleport(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    teleport_radius: float = 1.0,
    robot_name: str = "robot",
    ball_name: str = "ball",
) -> None:
    """Interval event: periodically teleport ball near the robot.

    Original DribbleBot design: teleport ball to random position within 1.0m
    every ~7s. Simulates human kicking the ball back, ball going out of FOV, etc.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    robot: Entity = env.scene[robot_name]
    ball: Entity = env.scene[ball_name]

    robot_xy = robot.data.root_link_pos_w[env_ids, :2]

    r = teleport_radius * torch.sqrt(torch.rand(len(env_ids), device=env.device))
    theta = torch.rand(len(env_ids), device=env.device) * (2 * torch.pi)
    new_x = robot_xy[:, 0] + r * torch.cos(theta)
    new_y = robot_xy[:, 1] + r * torch.sin(theta)

    ball_pos = ball.data.root_link_pos_w[env_ids].clone()
    ball_quat = ball.data.root_link_quat_w[env_ids].clone()
    ball_pos[:, 0] = new_x
    ball_pos[:, 1] = new_y
    ball_pos[:, 2] = 0.09
    new_pose = torch.cat([ball_pos, ball_quat], dim=-1)
    zero_vel = torch.zeros(len(env_ids), 6, device=env.device)

    ball.write_root_link_pose_to_sim(new_pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(zero_vel, env_ids=env_ids)

    # Clear stale xfrc_applied from pre-teleport drag.
    zero_w = torch.zeros(len(env_ids), 1, 3, device=env.device)
    ball.write_external_wrench_to_sim(zero_w, zero_w, env_ids=env_ids)


def init_ball_position(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    init_radius: float = 2.0,
    robot_name: str = "robot",
    ball_name: str = "ball",
) -> None:
    """Reset event: randomize ball position within init_radius of robot.

    Paper Section III-A.1: "The soccer ball is initialized at a random
    position within 2 m of the robot."
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    robot: Entity = env.scene[robot_name]
    ball: Entity = env.scene[ball_name]

    robot_xy = robot.data.root_link_pos_w[env_ids, :2]

    r = init_radius * torch.sqrt(torch.rand(len(env_ids), device=env.device))
    theta = torch.rand(len(env_ids), device=env.device) * (2 * torch.pi)
    new_x = robot_xy[:, 0] + r * torch.cos(theta)
    new_y = robot_xy[:, 1] + r * torch.sin(theta)

    ball_pos = ball.data.root_link_pos_w[env_ids].clone()
    ball_quat = ball.data.root_link_quat_w[env_ids].clone()
    ball_pos[:, 0] = new_x
    ball_pos[:, 1] = new_y
    ball_pos[:, 2] = 0.09
    new_pose = torch.cat([ball_pos, ball_quat], dim=-1)
    zero_vel = torch.zeros(len(env_ids), 6, device=env.device)

    ball.write_root_link_pose_to_sim(new_pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(zero_vel, env_ids=env_ids)

    # Clear stale xfrc_applied (drag from previous episode).
    zero_w = torch.zeros(len(env_ids), 1, 3, device=env.device)
    ball.write_external_wrench_to_sim(zero_w, zero_w, env_ids=env_ids)


# ======================================================================
# Ball Drag Model (Section III-B.4): FD = CD * v²
# ======================================================================

def init_ball_drag_cd(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    cd_range: tuple[float, float] = (0.0, 1.5),
) -> None:
    """Startup event: sample drag coefficient CD per environment.

    Each env gets a fixed CD for the episode, randomized at startup.
    CD=0 emulates pavement (ball rolls freely), CD=1.5 emulates tall grass.
    Aligned with DribbleBot Table IV: Ball-Terrain Drag Coefficient [0.0, 1.5].
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    cd_min, cd_max = cd_range
    if not hasattr(env, "_ball_drag_cd"):
        env._ball_drag_cd = torch.zeros(env.num_envs, device=env.device)
    env._ball_drag_cd[env_ids] = (
        torch.rand(len(env_ids), device=env.device) * (cd_max - cd_min) + cd_min
    )


def apply_ball_drag(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ball_name: str = "ball",
) -> None:
    """Step event: apply aerodynamic drag force FD = CD * v² to the ball.

    Force is applied opposite to ball velocity via MuJoCo xfrc_applied.
    This both limits ball rolling distance and creates varying ball dynamics
    across environments — critical for learning robust dribbling behaviors.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    ball: Entity = env.scene[ball_name]
    cd = getattr(env, "_ball_drag_cd", torch.zeros(env.num_envs, device=env.device))

    vel_w = ball.data.root_link_lin_vel_w[env_ids, :2]  # [N, 2]
    speed = torch.norm(vel_w, dim=-1)  # [N]
    cd_sel = cd[env_ids]  # [N]

    # F_vec = -CD * ||v|| * v  (drag opposite to velocity, magnitude ∝ v²)
    drag_2d = -cd_sel.unsqueeze(-1) * speed.unsqueeze(-1) * vel_w  # [N, 2]

    # External wrench: force only (no torque), must be 3D with body dim
    drag_3d = torch.cat([
        drag_2d,
        torch.zeros(len(env_ids), 1, device=env.device),
    ], dim=-1).unsqueeze(1)  # [N, 1, 3]
    torque = torch.zeros_like(drag_3d)

    ball.write_external_wrench_to_sim(drag_3d, torque, env_ids=env_ids)
