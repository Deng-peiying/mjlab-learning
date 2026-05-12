"""Unitree Go1 运球环境配置。

提供两个配置函数：
  unitree_go1_rough_env_cfg()  — 崎岖地形（程序化生成 + 课程学习）
  unitree_go1_flat_env_cfg()    — 平坦地形（从 rough 配置继承并简化）

每个函数调用 make_dribbling_env_cfg() 获取基础配置，然后：
  1. 添加 Go1 机器人实体
  2. 添加球实体（MuJoCo sphere + freejoint, 半径 0.09m）
  3. 配置接触传感器（足端、大腿、小腿、躯干、自碰撞）
  4. 配置射线传感器坐标系（trunk 参考）
  5. 定制奖励项参数（flat_orientation body, stance_phase site_names）
  6. 追加 Go1 特定的碰撞惩罚到负向奖励列表
  7. 替换 foot_friction 为 condim=6 逐轴版本

play=True 模式：
  - 无限长 episode
  - 关闭观测噪声
  - 移除 push_robot 扰动
  - 随机化地形（代替课程学习）
"""

import math
from typing import Literal

from mjlab.asset_zoo.robots import (
    GO1_ACTION_SCALE,
    get_go1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.dribbling import mdp
from mjlab.tasks.dribbling.mdp.ball_command import BallVelocityCommandCfg
from mjlab.tasks.dribbling.dribbling_env_cfg import make_dribbling_env_cfg
import mujoco
from mjlab.entity.entity import EntityCfg

TerrainType = Literal["rough", "obstacles"]


def unitree_go1_rough_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """创建 Unitree Go1 崎岖地形dribbling配置。"""
    cfg = make_dribbling_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.mujoco.impratio = 10
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.contact_sensor_maxmatch = 500

    cfg.scene.entities = {"robot": get_go1_robot_cfg()}

    # ---- 新增：球实体 ----
    def _make_ball_spec() -> mujoco.MjSpec:
        """创建足球的 MuJoCo 模型：一个自由浮动的球体。"""
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="ball_body")
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.09,) * 3,       # 半径 0.09m（3 号足球，论文值）
            mass=0.25,               # 质量 0.25kg
            friction=(0.7, 0.05, 0.10),   # 高滚动/扭转阻力，球有自然衰减
        )
        body.add_freejoint()         # 自由关节：球可以滚动/飞行
        return spec

    ball_cfg = EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.09),   # 初始在机器人前方 0.5m，离地 0.09m（球半径）
            joint_pos={},          # 球没有关节
            joint_vel={},
            lin_vel=(0.0, 0.0, 0.0),
        ),
        spec_fn=_make_ball_spec,
    )
    cfg.scene.entities["ball"] = ball_cfg


    # 将射线传感器坐标系设置为 Go1 的躯干。
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "trunk"

    foot_names = ("FR", "FL", "RR", "RL")
    site_names = ("FR", "FL", "RR", "RL")
    geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

    # 将足部高度扫描连接到各脚站点。
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot") for s in site_names
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    thigh_geom_names = tuple(
        f"{leg}_thigh_collision{i}" for leg in foot_names for i in (1, 2, 3)
    )
    thigh_ground_cfg = ContactSensorCfg(
        name="thigh_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=thigh_geom_names,
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    calf_geom_names = tuple(
        f"{leg}_calf_collision{i}" for leg in foot_names for i in (1, 2)
    )
    shank_ground_cfg = ContactSensorCfg(
        name="shank_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=calf_geom_names,
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    trunk_head_ground_cfg = ContactSensorCfg(
        name="trunk_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=("trunk_collision", "head_collision"),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
        thigh_ground_cfg,
        shank_ground_cfg,
        trunk_head_ground_cfg,
    )

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = GO1_ACTION_SCALE

    cfg.viewer.body_name = "trunk"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    # 将基础的 foot_friction 替换为逐轴摩擦事件以适配 condim 6。
    del cfg.events["foot_friction"]
    cfg.events["foot_friction_slide"] = EventTermCfg(
        mode="startup",
        func=envs_mdp.dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
            "operation": "abs",
            "axes": [0],
            "ranges": (0.40, 1.00),  # paper Table IV
            "shared_random": True,
        },
    )
    cfg.events["foot_friction_spin"] = EventTermCfg(
        mode="startup",
        func=envs_mdp.dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
            "operation": "abs",
            "distribution": "log_uniform",
            "axes": [1],
            "ranges": (1e-4, 2e-2),
            "shared_random": True,
        },
    )
    cfg.events["foot_friction_roll"] = EventTermCfg(
        mode="startup",
        func=envs_mdp.dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
            "operation": "abs",
            "distribution": "log_uniform",
            "axes": [2],
            "ranges": (1e-5, 5e-3),
            "shared_random": True,
        },
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)
    cfg.events["payload_mass"].params["asset_cfg"].body_names = ("trunk",)

    # ---- 奖励定制：修改乘法奖励中的子奖励项参数 ----
    total_params = cfg.rewards["total"].params

    # flat_orientation: 从 trunk 读取重力向量
    for term in total_params["neg_terms"]:
        if term["name"] == "flat_orientation":
            term["params"]["asset_cfg"].body_names = ("trunk",)
            break

    # stance_phase: 设置具体的足端 site 名称
    for term in total_params["pos_terms"]:
        if term["name"] == "stance_phase":
            term["params"]["asset_cfg"].site_names = site_names
            break

    # 碰撞惩罚：追加到 neg_terms
    total_params["neg_terms"].extend([
        {"name": "self_collisions", "func": "self_collision_cost", "weight": -5.0,
         "params": {"sensor_name": self_collision_cfg.name}},
        {"name": "shank_collision", "func": "self_collision_cost", "weight": -5.0,
         "params": {"sensor_name": shank_ground_cfg.name}},
        {"name": "trunk_head_collision", "func": "self_collision_cost", "weight": -5.0,
         "params": {"sensor_name": trunk_head_ground_cfg.name}},
    ])

    # 在崎岖地形上四足机器人会大幅倾斜；不要仅因朝向就终止 episode。
    # 让 out_of_terrain_bounds 来处理重置。
    cfg.terminations.pop("fell_over", None)

    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": thigh_ground_cfg.name},
    )

    # 应用 play 模式覆盖。
    if play:
        # 实际上无限长的 episode。
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def unitree_go1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """创建 Unitree Go1 平坦地形dribbling配置。"""
    cfg = unitree_go1_rough_env_cfg(play=play)

    cfg.sim.njmax = 500
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    # 切换为平坦地形。
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # 移除平坦地形上不需要的射线传感器。
    remove_sensors = {
        "terrain_scan",
        "thigh_ground_touch",
    }
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
    )
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    # 平坦地形上不需要 terrain_sensor_names 参数
    for term in cfg.rewards["total"].params["neg_terms"]:
        if term["name"] == "flat_orientation":
            term["params"].pop("terrain_sensor_names", None)
            break

    # 在平坦地形上 thighs 接地视为摔倒。保留 shank/trunk/self 碰撞检测。
    cfg.terminations.pop("illegal_contact", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    )

    # 禁用地形课程学习（在 play 模式下 rough 函数已清除所有课程项，因此此处无需处理）。
    cfg.curriculum.pop("terrain_levels", None)

    if play:
        ball_vel_cmd = cfg.commands["ball_vel"]
        assert isinstance(ball_vel_cmd, BallVelocityCommandCfg)
        ball_vel_cmd.ranges.ball_vel_x = (-1.5, 1.5)
        ball_vel_cmd.ranges.ball_vel_y = (-1.5, 1.5)

    return cfg
