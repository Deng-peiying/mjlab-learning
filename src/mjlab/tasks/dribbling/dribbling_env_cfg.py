"""运球任务配置。

本模块提供工厂函数 make_dribbling_env_cfg()，创建基础运球任务配置。
各机器人特定配置（config/go1/env_cfgs.py）调用该工厂函数并根据需要进行定制。

架构说明：
  dribbling_env_cfg.py 通过 `from mjlab.tasks.dribbling import mdp` 访问所有 MDP 函数。
  mdp 命名空间由 dribbling/mdp/__init__.py 组装。

关键设计决策（对齐 DribbleBot 论文 Ji, Margolis, Agrawal 2023）：
  1. 奖励结构：r_t = r_pos * exp(r_neg)  — 乘法组合（非加法）
  2. 指令接口：球速 (vx, vy) 在世界坐标系中指定，非身体速度
  3. 观测：15 步历史 + 步态时序参考 (theta_cmd) + 球位置 + 本体感觉 + 偏航角
  4. 球事件：drag 阻力模型 (FD = CD*v^2) + 距离传送 + 周期传送
  5. Episode 长度：40s（论文值）

修改记录（对齐论文的改动，详见文件末尾）：
  - 奖励结构：加法 -> 乘法 (DribblingTotalReward)，upright -> flat_orientation_l2
  - 新增：action_acc_l2（二阶平滑），joint_vel_l1（L1 范数替代 L2）
  - Episode：20s -> 40s
  - 球：初始位置随机化 2m 内，半径 0.089 -> 0.09m
  - 域随机化：新增 payload_mass [-1,3]kg, motor_strength [90,110]%
  - CoM 位移范围：+/-2.5cm -> +/-15cm，calibration +/-0.015 -> +/-0.02
  - 摩擦范围：[0.3,1.5] -> [0.40,1.00]
"""

# --- 标准库 ---
import math  # 数学函数（三角函数、指数 exp、sqrt 等，用于奖励计算）
from dataclasses import replace  # 深拷贝 dataclass 并替换指定字段，用于基于默认配置创建变体

# --- 环境配置基类 ---
from mjlab.envs import ManagerBasedRlEnvCfg  # Manager-based RL 环境的总配置类，聚合所有子 Manager 的配置

# --- MDP 基础组件 ---
from mjlab.envs import mdp as envs_mdp  # 通用 MDP 函数命名空间（噪声、观测、奖励等基础函数）
from mjlab.envs.mdp import dr  # 域随机化模块（质量、摩擦、推进力等物理参数的随机化）
from mjlab.envs.mdp.actions import JointPositionActionCfg  # 关节位置控制动作配置（PD 控制器目标关节角度）

# --- Manager 配置类 ---
# Manager 是 Isaac Lab 的模块化 RL 架构：每种功能由一个 Manager 负责
from mjlab.managers.action_manager import ActionTermCfg  # 动作项配置：定义动作空间 + 将网络输出映射到关节指令
from mjlab.managers.command_manager import CommandTermCfg  # 指令项配置：定义任务指令（如球速 vx,vy），随机采样或课程学习
from mjlab.managers.curriculum_manager import CurriculumTermCfg  # 课程项配置：根据训练进度动态调整难度参数
from mjlab.managers.event_manager import EventTermCfg  # 事件项配置：在 episode 中触发一次/周期性事件（重置、域随机化等）
from mjlab.managers.metrics_manager import MetricsTermCfg  # 指标项配置：记录训练/评估指标到日志（球位移误差等）
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg  # 观测组/项配置：定义哪些传感器数据组合成观测
from mjlab.managers.reward_manager import RewardTermCfg  # 奖励项配置：定义单个奖励项（权重、函数、缩放等）
from mjlab.managers.scene_entity_config import SceneEntityCfg  # 场景实体引用：通过名称索引场景中的刚体/关节/传感器
from mjlab.managers.termination_manager import TerminationTermCfg  # 终止项配置：定义 episode 提前终止的条件（摔倒、出界等）

# --- 场景 & 仿真 ---
from mjlab.scene import SceneCfg  # 场景配置：管理所有实体（机器人、地形、光源等）的加载与实例化
from mjlab.sim import MujocoCfg, SimulationCfg  # MuJoCo 物理引擎参数 + 仿真配置（时间步长、求解器、重力等）

# --- 传感器 ---
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg, TerrainHeightSensorCfg  # 栅格采样模式 + 物体引用 + 射线传感器 + 地形高度传感器（用于感知球位置和脚下地形）

# --- 运球任务 MDP ---
from mjlab.tasks.dribbling import mdp  # 运球专属 MDP 函数（球速指令、运球奖励、球事件等）
from mjlab.tasks.dribbling.mdp import BallVelocityCommandCfg  # 球速指令配置：指定球在世界坐标系中的目标速度 (vx, vy)

# --- 地形 ---
from mjlab.terrains import TerrainEntityCfg  # 地形实体配置：定义地形的类型、大小、难度参数
from mjlab.terrains.config import ROUGH_TERRAINS_CFG  # 预定义的崎岖地形配置（高度图、台阶、斜坡等组合）

# --- 工具 ---
from mjlab.utils.noise import UniformNoiseCfg as Unoise  # 均匀噪声配置：为传感器/动作添加噪声实现域随机化
from mjlab.viewer import ViewerConfig  # 可视化配置：控制渲染窗口、相机位置、分辨率等


def make_dribbling_env_cfg() -> ManagerBasedRlEnvCfg:
    """创建基础运球任务配置。

    返回 ManagerBasedRlEnvCfg，包含场景、观测、动作、奖励、
    事件、终止条件、课程学习的完整定义。

    Go1 配置 (config/go1/env_cfgs.py) 在此基础之上添加：
      - 球实体（MuJoCo sphere + freejoint）
      - Go1 机器人实体
      - 接触传感器（足端、大腿、小腿、躯干、自碰撞）
      - 追加碰撞惩罚到负向奖励列表
      - 替换 foot_friction 为逐轴版本适配 condim=6
      - 平坦模式：移除射线传感器和地形课程
    """

    ## 传感器 ============================================================
    # 地形高度扫描：从躯干向地面发射射线，测量地形高度
    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=True,
    )

    # 足部高度扫描：从每只脚的 site 向下发射射线
    foot_height_scan = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=(),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=True,
        viz=TerrainHeightSensorCfg.VizCfg(
            show_rays=True,
            hit_color=(1.0, 0.0, 1.0, 0.8),
            hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
        ),
    )

    ## 观测 ==============================================================
    # actor：策略网络输入（带噪声模拟真实传感器）
    # critic：价值网络输入（不带噪声，包含特权信息）

    # 论文 III-A.3 观测空间 (37 dims): command(2) + ball_pos(3) + joint_pos(12)
    #   + joint_vel(12) + projected_gravity(3) + body_yaw(1) + gait_timing_ref(4)
    # 注：base_lin_vel / base_ang_vel / last_action 不在论文观测中 ——
    #   真实机器人无法直接测量 base velocity，动作历史已由关节历史隐含。
    actor_terms = {
        # --- 本体感觉 ---
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
        # --- 指令（无噪声）---
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "ball_vel"},
        ),
        # --- 球位置（带 YOLO 检测噪声，论文 VIII）---
        "ball_pos": ObservationTermCfg(
            func=mdp.ball_pos_b,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        # --- 步态时序参考 theta_cmd（论文 III-A.3）---
        "gait_timing_ref": ObservationTermCfg(func=mdp.gait_timing_ref),
        # --- 世界坐标系偏航角（论文 III-A.3）---
        "body_yaw": ObservationTermCfg(func=mdp.body_yaw),
        # --- 地形扫描（仅 rough 模式；flat 模式在 go1/env_cfgs.py 中移除）---
        "height_scan": ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": "terrain_scan"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            scale=1 / terrain_scan.max_distance,
        ),
    }

    # critic = actor 的超集 + 特权信息（无噪声）
    critic_terms = {
        **actor_terms,
        "height_scan": ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": "terrain_scan"},
            scale=1 / terrain_scan.max_distance,
        ),
        # 特权观测（仅 critic 可见）
        "foot_height": ObservationTermCfg(
            func=mdp.foot_height, params={"sensor_name": "foot_height_scan"}
        ),
        "foot_air_time": ObservationTermCfg(
            func=mdp.foot_air_time, params={"sensor_name": "feet_ground_contact"}
        ),
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}
        ),
        "foot_contact_forces": ObservationTermCfg(
            func=mdp.foot_contact_forces, params={"sensor_name": "feet_ground_contact"}
        ),
        "gait_timing_ref": ObservationTermCfg(func=mdp.gait_timing_ref),
        "body_yaw": ObservationTermCfg(func=mdp.body_yaw),
        "ball_pos": ObservationTermCfg(func=mdp.ball_pos_b),
        "ball_vel": ObservationTermCfg(func=mdp.ball_vel_w),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=15,   # 论文 III-A.3：15 步观测历史
            flatten_history_dim=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=15,
            flatten_history_dim=True,
        ),
    }

    ## 指标 ==============================================================
    metrics = {
        "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
    }

    ## 动作 ==============================================================
    # 12 个关节的位置目标，scale=0.5（Go1 配置覆盖为 GO1_ACTION_SCALE）
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
            use_default_offset=True,
        )
    }

    ## 指令 ==============================================================
    # 球速指令 (vx, vy) 在世界坐标系中指定
    # 每 3-8 秒随机重采样；10% 环境接收零指令（球不动）
    commands: dict[str, CommandTermCfg] = {
        "ball_vel": BallVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.1,
            debug_vis=True,
            ranges=BallVelocityCommandCfg.Ranges(
                ball_vel_x=(-1.5, 1.5),
                ball_vel_y=(-1.5, 1.5),
            ),
        )
    }

    ## 事件 ==============================================================
    events = {
        # ---- 重置事件 (mode="reset") ----
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5), "y": (-0.5, 0.5),
                    "z": (0.01, 0.05), "yaw": (-3.14, 3.14),
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
        # 球初始位置随机化（论文 III-A.1：2m 内随机）
        # 必须在 reset_base 之后，否则球会放在旧机器人位置上
        "init_ball_position": EventTermCfg(
            func=mdp.init_ball_position,
            mode="reset",
            params={"init_radius": 2.0},
        ),
        # 随机推机器人（每 1-3 秒）
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
                },
            },
        ),

        # ---- 域随机化事件 (mode="startup") ----
        # 足部摩擦随机化（Go1 配置覆盖为逐轴版本）
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),
                "operation": "abs",
                "ranges": (0.40, 1.00),  # 论文 Table IV
                "shared_random": True,
            },
        ),
        # 编码器偏置 [-0.02, 0.02] rad（论文 Table IV）
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.02, 0.02),
            },
        ),
        # 质心偏移 [-0.15, 0.15] m（论文 Table IV）
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),
                "operation": "add",
                "ranges": {0: (-0.15, 0.15), 1: (-0.15, 0.15), 2: (-0.15, 0.15)},
            },
        ),
        # 负载质量 [-1.0, 3.0] kg（论文 Table IV）
        "payload_mass": EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),
                "operation": "add",
                "distribution": "uniform",
                "ranges": (-1.0, 3.0),
            },
        ),
        # 电机力矩 [90, 110]%（论文 Table IV）
        "motor_strength": EventTermCfg(
            mode="startup",
            func=dr.effort_limits,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "distribution": "uniform",
                "effort_limit_range": (0.9, 1.1),
            },
        ),

        # ---- 步态时钟 (mode="step") ----
        # 每步更新 trot 步态相位，生成 desired_contact 和 timing_ref
        "gait_clock": EventTermCfg(
            func=mdp.update_gait_clock,
            mode="step",
            params={},
        ),

        # ---- 球阻力模型 (mode="startup" + "step") ----
        # 论文 III-B.4：FD = CD * v^2，CD ∈ [0, 1.5]
        "init_ball_drag_cd": EventTermCfg(
            func=mdp.init_ball_drag_cd,
            mode="startup",
            params={"cd_range": (0.0, 1.5)},
        ),
        "apply_ball_drag": EventTermCfg(
            func=mdp.apply_ball_drag,
            mode="step",
        ),

        # ---- 球传送事件 ----
        # 论文 Section VIII：每 ~7s 传送球到 1m 内
        "periodic_ball_teleport": EventTermCfg(
            func=mdp.periodic_ball_teleport,
            mode="interval",
            interval_range_s=(6.0, 8.0),
            params={"teleport_radius": 1.0},
        ),
        # 安全网：距离 >4m 立即传送
        "teleport_ball_if_far": EventTermCfg(
            func=mdp.teleport_ball_if_far,
            mode="step",
            params={"max_distance": 4.0, "teleport_radius": 1.0},
        ),

        # ---- 球域随机化 ----
        # 球质量 [0.159, 0.254] kg（论文 Table IV）
        "ball_mass": EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("ball", body_names=("ball_body",)),
                "operation": "abs",
                "distribution": "uniform",
                "ranges": (0.159, 0.254),
            },
        ),
        # 球速度扰动 [0.0, 0.3] m/s（论文 Table IV）
        "ball_push": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 8.0),
            params={
                "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
                "asset_cfg": SceneEntityCfg("ball"),
            },
        ),
    }

    ## 奖励 ==============================================================
    # 论文使用乘法结构：r_t = r_pos * exp(r_neg)
    # DribblingTotalReward 将正/负奖励分别累加后组合
    #
    # r_pos (7 项)：track_ball_velocity(0.5), robot_ball_distance(4.0),
    #   robot_ball_yaw(4.0), ball_vel_norm(4.0), ball_vel_angle(4.0),
    #   swing_phase(4.0), stance_phase(4.0)
    #
    # r_neg (7 项基础 + Go1 配置追加碰撞项)：
    #   joint_pos_limits(-10.0), joint_torques(-0.0001), joint_vel(-0.0001),
    #   joint_acc(-2.5e-7), flat_orientation(-5.0),
    #   action_rate(-0.1), action_acc(-0.1)

    # 注册所有奖励函数到全局 registry
    mdp.register_reward_func("track_ball_velocity", mdp.track_ball_velocity)
    mdp.register_reward_func("robot_ball_distance", mdp.robot_ball_distance)
    mdp.register_reward_func("robot_ball_yaw", mdp.robot_ball_yaw)
    mdp.register_reward_func("ball_vel_norm", mdp.ball_vel_norm)
    mdp.register_reward_func("ball_vel_angle", mdp.ball_vel_angle)
    mdp.register_reward_func("swing_phase", mdp.swing_phase)
    mdp.register_reward_func("stance_phase", mdp.stance_phase)
    mdp.register_reward_func("joint_pos_limits", mdp.joint_pos_limits_binary)
    mdp.register_reward_func("joint_torques_l2", mdp.joint_torques_l2)
    mdp.register_reward_func("joint_vel_l1", mdp.joint_vel_l1)
    mdp.register_reward_func("joint_acc_l1", mdp.joint_acc_l1)
    mdp.register_reward_func("flat_orientation_l2", mdp.flat_orientation_l2)
    mdp.register_reward_func("action_rate_l2", mdp.action_rate_l2)
    mdp.register_reward_func("action_acc_l2", mdp.action_acc_l2)
    mdp.register_reward_func("self_collision_cost", mdp.self_collision_cost)
    mdp.register_reward_func("ball_under_body", mdp.ball_under_body)

    pos_terms = [
        {"name": "track_ball_velocity", "func": "track_ball_velocity", "weight": 0.5,
         "params": {"command_name": "ball_vel", "std": math.sqrt(0.25)}},
        {"name": "robot_ball_distance", "func": "robot_ball_distance", "weight": 4.0,
         "params": {"std": math.sqrt(0.25)}},
        {"name": "robot_ball_yaw", "func": "robot_ball_yaw", "weight": 4.0,
         "params": {"command_name": "ball_vel", "std": math.sqrt(0.5)}},
        {"name": "ball_vel_norm", "func": "ball_vel_norm", "weight": 4.0,
         "params": {"command_name": "ball_vel", "std": math.sqrt(0.5)}},
        {"name": "ball_vel_angle", "func": "ball_vel_angle", "weight": 4.0,
         "params": {"command_name": "ball_vel"}},
        {"name": "swing_phase", "func": "swing_phase", "weight": 4.0,
         "params": {"sensor_name": "feet_ground_contact", "force_sigma": 50.0}},
        {"name": "stance_phase", "func": "stance_phase", "weight": 4.0,
         "params": {"sensor_name": "feet_ground_contact", "vel_sigma": 0.5,
                    "asset_cfg": SceneEntityCfg("robot", site_names=())}},
    ]

    neg_terms = [
        {"name": "joint_pos_limits", "func": "joint_pos_limits", "weight": -10.0, "params": {}},
        {"name": "joint_torques", "func": "joint_torques_l2", "weight": -0.0001, "params": {}},
        {"name": "joint_vel", "func": "joint_vel_l1", "weight": -0.0001, "params": {}},
        {"name": "joint_acc", "func": "joint_acc_l1", "weight": -2.5e-7, "params": {}},
        {"name": "flat_orientation", "func": "flat_orientation_l2", "weight": -5.0,
         "params": {"asset_cfg": SceneEntityCfg("robot", body_names=())}},
        {"name": "action_rate", "func": "action_rate_l2", "weight": -0.1, "params": {}},
        {"name": "action_acc", "func": "action_acc_l2", "weight": -0.1, "params": {}},
        {"name": "ball_under_body", "func": "ball_under_body", "weight": -5.0,
         "params": {"std": 0.05,
                    "base_x_range": (-0.15, 0.22),
                    "base_y_range": (-0.10, 0.10)}},
    ]

    rewards = {
        "total": RewardTermCfg(
            func=mdp.DribblingTotalReward,
            weight=1.0,
            params={"pos_terms": pos_terms, "neg_terms": neg_terms},
        ),
    }

    ## 终止条件 ==========================================================
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "base_too_low": TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.20},
        ),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
        "out_of_terrain_bounds": TerminationTermCfg(
            func=mdp.out_of_terrain_bounds,
            time_out=True,
        ),
    }

    ## 课程学习 ==========================================================
    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={"command_name": "ball_vel"},
        ),
    }

    ## 组装 ==============================================================
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=replace(ROUGH_TERRAINS_CFG),
                max_init_terrain_level=5,
            ),
            sensors=(terrain_scan, foot_height_scan),
            num_envs=4096,   # 论文值
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
            body_name="",
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=35,
            njmax=3000,
            mujoco=MujocoCfg(
                timestep=0.005,      # 200 Hz 物理频率
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,          # RL step = 4 物理子步 -> 50 Hz 控制频率
        episode_length_s=40.0,  # 论文 III-A.1：每个 episode 持续 40 秒
    )
