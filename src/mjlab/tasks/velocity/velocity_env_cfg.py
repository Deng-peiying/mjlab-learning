"""速度任务配置。

本模块提供工厂函数，用于创建基础速度任务配置。
各机器人特定配置调用该工厂函数，并根据需要进行定制。
"""

import math
from dataclasses import replace

# 核心环境配置
from mjlab.envs import ManagerBasedRlEnvCfg  # 管理器式RL环境的总配置容器（场景+观测+动作+奖励+终止+指令+课程）
from mjlab.envs import mdp as envs_mdp  # 通用MDP函数库（高度扫描、地形随机化等）
from mjlab.envs.mdp import dr  # 域随机化函数（摩擦力、质心偏移、编码器偏差等）
from mjlab.envs.mdp.actions import JointPositionActionCfg  # 关节位置动作配置

# 管理器配置
from mjlab.managers.action_manager import ActionTermCfg  # 动作项配置（将关节目标绑定到实体）
from mjlab.managers.command_manager import CommandTermCfg  # 指令项配置（速度指令、步态指令等）
from mjlab.managers.curriculum_manager import CurriculumTermCfg  # 课程学习项配置（自适应难度调整）
from mjlab.managers.event_manager import EventTermCfg  # 事件项配置（重置、域随机化、推力扰动等）
from mjlab.managers.metrics_manager import MetricsTermCfg  # 指标项配置（训练日志中的自定义指标）
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg  # 观测组/项配置（actor/critic分组、噪声、历史记录）
from mjlab.managers.reward_manager import RewardTermCfg  # 奖励项配置（奖励函数+权重）
from mjlab.managers.scene_entity_config import SceneEntityCfg  # 场景实体引用配置（指定机器人/关节/身体/几何体名称）
from mjlab.managers.termination_manager import TerminationTermCfg  # 终止项配置（超时、跌倒、出界等）

# 场景与仿真
from mjlab.scene import SceneCfg  # 场景配置（地形、实体、传感器、并行环境数）
from mjlab.sim import MujocoCfg, SimulationCfg  # 仿真配置（物理时间步、解算器迭代、接触参数）

# 传感器
from mjlab.sensor import (
    GridPatternCfg,  # 网格采样模式配置（射线传感器的采样点排列）
    ObjRef,  # 对象引用（指定传感器参考的实体/刚体/部位）
    RayCastSensorCfg,  # 射线传感器配置（地形高度扫描）
    TerrainHeightSensorCfg,  # 地形高度传感器配置（足部离地高度检测）
)

# 任务与地形
from mjlab.tasks.velocity import mdp  # velocity任务的MDP函数库（观测/奖励/终止/指令的具体实现）
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg  # 均匀速度指令配置（线速度/角速度/航向范围）
from mjlab.terrains import TerrainEntityCfg  # 地形实体配置（平面/生成器/课程学习）
from mjlab.terrains.config import ROUGH_TERRAINS_CFG  # 崎岖地形预设配置

# 工具
from mjlab.utils.noise import UniformNoiseCfg as Unoise  # 均匀噪声配置（观测噪声的上下界）
from mjlab.viewer import ViewerConfig  # 查看器配置（相机位置、视角、分辨率）


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
    """创建基础速度跟踪任务配置。"""

    ##
    # 传感器
    ##

    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="", entity="robot"),  # 各机器人单独设置。
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),  # 仅检测地形。
        debug_vis=True,
    )

    foot_height_scan = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=(),  # 各机器人单独设置：frame 和 pattern。
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),  # 仅检测地形。
        debug_vis=True,
        viz=TerrainHeightSensorCfg.VizCfg(
            show_rays=True,
            hit_color=(1.0, 0.0, 1.0, 0.8),  # 洋红色射线。
            hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
        ),
    )

    ##
    # 观测
    ##

    actor_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
        "height_scan": ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": "terrain_scan"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            scale=1 / terrain_scan.max_distance,
        ),
    }

    critic_terms = {
        **actor_terms,
        "height_scan": ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": "terrain_scan"},
            scale=1 / terrain_scan.max_distance,
        ),
        "foot_height": ObservationTermCfg(
            func=mdp.foot_height,
            params={"sensor_name": "foot_height_scan"},
        ),
        "foot_air_time": ObservationTermCfg(
            func=mdp.foot_air_time,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact_forces": ObservationTermCfg(
            func=mdp.foot_contact_forces,
            params={"sensor_name": "feet_ground_contact"},
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    ##
    # 指标
    ##

    metrics = {
        "mean_action_acc": MetricsTermCfg(
            func=mdp.mean_action_acc,
        ),
    }

    ##
    # 动作
    ##

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,  # 各机器人单独覆盖.
            use_default_offset=True,
        )
    }

    ##
    # 指令
    ##

    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.3,
            rel_forward_envs=0.2,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-0.5, 0.5),
                heading=(-math.pi, math.pi),
            ),
        )
    }

    ##
    # 事件
    ##

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.01, 0.05),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", geom_names=()
                ),  # 各机器人单独设置。
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,  # 所有足部几何体共享相同的摩擦力。
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=()
                ),  # 各机器人单独设置。
                "operation": "add",
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
            },
        ),
    }

    ##
    # 奖励
    ##

    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": math.sqrt(0.5)},
        ),
        "upright": RewardTermCfg(
            func=mdp.upright,
            weight=1.0,
            params={
                "std": math.sqrt(0.2),
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=()
                ),  # 各机器人单独设置。
            },
        ),
        "pose": RewardTermCfg(
            func=mdp.variable_posture,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "command_name": "twist",
                "std_standing": {},  # 各机器人单独设置。
                "std_walking": {},  # 各机器人单独设置。
                "std_running": {},  # 各机器人单独设置。
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
            },
        ),
        "body_ang_vel": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=0.0,  # 各机器人单独覆盖
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=())
            },  # 各机器人单独设置。
        ),
        "angular_momentum": RewardTermCfg(
            func=mdp.angular_momentum_penalty,
            weight=0.0,  # 各机器人单独覆盖
            params={"sensor_name": "robot/root_angmom"},
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "air_time": RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.0,  # 各机器人单独覆盖.
            params={
                "sensor_name": "feet_ground_contact",
                "threshold_min": 0.05,
                "threshold_max": 0.5,
                "command_name": "twist",
                "command_threshold": 0.5,
            },
        ),
        "foot_clearance": RewardTermCfg(
            func=mdp.feet_clearance,
            weight=-2.0,
            params={
                "target_height": 0.1,
                "height_sensor_name": "foot_height_scan",
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg(
                    "robot", site_names=()
                ),  # 各机器人单独设置。
            },
        ),
        "foot_swing_height": RewardTermCfg(
            func=mdp.feet_swing_height,
            weight=-0.25,
            params={
                "sensor_name": "feet_ground_contact",
                "height_sensor_name": "foot_height_scan",
                "target_height": 0.1,
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
        "foot_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.1,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg(
                    "robot", site_names=()
                ),  # 各机器人单独设置。
            },
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-1e-5,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
    }

    ##
    # 终止条件
    ##

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
        "out_of_terrain_bounds": TerminationTermCfg(
            func=mdp.out_of_terrain_bounds,
            time_out=True,
        ),
    }

    ##
    # 课程学习
    ##

    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={"command_name": "twist"},
        ),
        "command_vel": CurriculumTermCfg(
            func=mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                    {
                        "step": 5000 * 24,
                        "lin_vel_x": (-1.5, 2.0),
                        "ang_vel_z": (-0.7, 0.7),
                    },
                    {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
                ],
            },
        ),
    }

    ##
    # 组装并返回
    ##

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=replace(ROUGH_TERRAINS_CFG),
                max_init_terrain_level=5,
            ),
            sensors=(terrain_scan, foot_height_scan),
            num_envs=1,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # 各机器人单独设置。
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=35,
            njmax=1500,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
    )
