"""
领域随机化（Domain Randomization）—— 关节与自由度（DOF）相关字段。

本模块提供了一系列用于随机化机器人关节物理参数和传感器误差的函数，
通过在训练时引入物理参数的随机扰动，提高强化学习策略从仿真到现实（Sim-to-Real）的迁移能力。

支持的随机化类型包括：
    - 关节阻尼（joint damping）
    - 关节电枢惯性（joint armature）
    - 关节摩擦损耗（joint friction loss）
    - 关节刚度（joint stiffness）
    - 关节限位（joint limits）
    - 关节默认位置（default joint positions）
    - 编码器偏置（encoder bias，用于模拟编码器标定误差）

每个随机化函数的参数签名高度统一，核心随机化逻辑由 ``_randomize_model_field`` 统一处理。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

from ._core import (
    _DEFAULT_ASSET_CFG,
    Ranges,
    _randomize_model_field,
)
from ._types import Distribution, Operation

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("dof_damping")
def joint_damping(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ranges: Ranges,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    distribution: Distribution | str = "uniform",
    operation: Operation | str = "abs",
    axes: list[int] | None = None,
    shared_random: bool = False,
) -> None:
    """
    随机化关节阻尼系数（dof_damping）。

    阻尼系数影响关节运动的能量耗散速率，其物理含义类似于粘性摩擦：
    τ_damping = -k_damping * \dot{q}，即阻尼力矩与关节速度成正比。

    在域随机化中随机化阻尼系数，可以使策略对关节摩擦特性的变化具有鲁棒性，
    有助于克服仿真模型与真实硬件之间的动力学差异。

    参数：
        env:            基于管理器（Manager-Based）的强化学习环境实例。
        env_ids:        需要施加随机化的环境索引张量，若为 None 则作用于全部环境。
        ranges:         随机化范围（Ranges 对象），定义参数采样的上下界。
        asset_cfg:      场景实体的配置信息，用于指定目标机器人及关节子集。
        distribution:   随机分布类型，支持 "uniform"（均匀分布）、"normal"（正态分布）等。
        operation:       随机化操作模式，"abs"（绝对赋值）或 "add"（偏移叠加）。
        axes:            指定需要随机化的自由度轴索引列表，None 表示作用于所有自由度。
        shared_random:   若为 True，同一批次内所有环境共享相同的随机采样值；
                         若为 False，每个环境独立采样，增加训练的多样性。

    返回值：
        None，函数直接在物理模型上原地修改参数。
    """
    _randomize_model_field(
        env,
        env_ids,
        "dof_damping",
        entity_type="dof",
        ranges=ranges,
        distribution=distribution,
        operation=operation,
        asset_cfg=asset_cfg,
        axes=axes,
        shared_random=shared_random,
        use_address=True,
    )


# 原始别名，允许使用更贴近 MuJoCo 物理模型字段名的调用方式。
dof_damping = joint_damping


@requires_model_fields("dof_armature", recompute=RecomputeLevel.set_const_0)
def joint_armature(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ranges: Ranges,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    distribution: Distribution | str = "uniform",
    operation: Operation | str = "abs",
    axes: list[int] | None = None,
    shared_random: bool = False,
) -> None:
    """
    随机化关节电枢惯性（dof_armature）。

    电枢惯性（armature）表示关节驱动电机转子的等效转动惯量，
    在 MuJoCo 的惯量矩阵中体现为对角元上的附加项：M_effective = M_link + diag(armature)。

    随机化电枢惯性可以模拟不同型号电机转子惯量的差异，
    或在 Sim-to-Real 过程中补偿未建模的转子动力学。

    .. note::
        该操作会触发 ``RecomputeLevel.set_const_0`` 级别的物理量重计算，
        因为电枢惯量的改变会影响整个惯量矩阵，需要重新组装运动方程。

    参数：
        env:            基于管理器的强化学习环境实例。
        env_ids:        需要施加随机化的环境索引张量，若为 None 则作用于全部环境。
        ranges:         随机化范围（Ranges 对象），定义参数采样的上下界。
        asset_cfg:      场景实体的配置信息，用于指定目标机器人及关节子集。
        distribution:   随机分布类型，支持 "uniform"（均匀分布）、"normal"（正态分布）等。
        operation:       随机化操作模式，"abs"（绝对赋值）或 "add"（偏移叠加）。
        axes:            指定需要随机化的自由度轴索引列表，None 表示作用于所有自由度。
        shared_random:   若为 True，同一批次内所有环境共享相同的随机采样值；
                         若为 False，每个环境独立采样。

    返回值：
        None，函数直接在物理模型上原地修改参数。
    """
    _randomize_model_field(
        env,
        env_ids,
        "dof_armature",
        entity_type="dof",
        ranges=ranges,
        distribution=distribution,
        operation=operation,
        asset_cfg=asset_cfg,
        axes=axes,
        shared_random=shared_random,
        use_address=True,
    )


# 原始别名，允许使用更贴近 MuJoCo 物理模型字段名的调用方式。
dof_armature = joint_armature


@requires_model_fields("dof_frictionloss")
def joint_friction(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ranges: Ranges,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    distribution: Distribution | str = "uniform",
    operation: Operation | str = "abs",
    axes: list[int] | None = None,
    shared_random: bool = False,
) -> None:
    """
    随机化关节摩擦损耗（dof_frictionloss）。

    摩擦损耗（frictionloss）是 MuJoCo 关节模型中用于描述库仑摩擦（Coulomb friction）的参数，
    其产生的摩擦力矩为：τ_friction = -frictionloss * sign(\dot{q})，
    即摩擦力矩大小恒定，方向与关节运动方向相反。

    在域随机化中随机化摩擦损耗，可以使策略对不同润滑条件、
    机械磨损程度下的真实关节摩擦特性具有鲁棒性。

    参数：
        env:            基于管理器的强化学习环境实例。
        env_ids:        需要施加随机化的环境索引张量，若为 None 则作用于全部环境。
        ranges:         随机化范围（Ranges 对象），定义参数采样的上下界。
        asset_cfg:      场景实体的配置信息，用于指定目标机器人及关节子集。
        distribution:   随机分布类型，支持 "uniform"（均匀分布）、"normal"（正态分布）等。
        operation:       随机化操作模式，"abs"（绝对赋值）或 "add"（偏移叠加）。
        axes:            指定需要随机化的自由度轴索引列表，None 表示作用于所有自由度。
        shared_random:   若为 True，同一批次内所有环境共享相同的随机采样值；
                         若为 False，每个环境独立采样。

    返回值：
        None，函数直接在物理模型上原地修改参数。
    """
    _randomize_model_field(
        env,
        env_ids,
        "dof_frictionloss",
        entity_type="dof",
        ranges=ranges,
        distribution=distribution,
        operation=operation,
        asset_cfg=asset_cfg,
        axes=axes,
        shared_random=shared_random,
        use_address=True,
    )


# 原始别名，允许使用更贴近 MuJoCo 物理模型字段名的调用方式。
dof_frictionloss = joint_friction


@requires_model_fields("jnt_stiffness")
def joint_stiffness(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ranges: Ranges,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    distribution: Distribution | str = "uniform",
    operation: Operation | str = "abs",
    axes: list[int] | None = None,
    shared_random: bool = False,
) -> None:
    """
    随机化关节刚度（jnt_stiffness）。

    关节刚度（stiffness）决定了关节在外力作用下抵抗形变的能力，
    产生的弹性恢复力矩为：τ_stiffness = -stiffness * (q - q0)，
    即恢复力矩与关节偏离参考位置的角度成正比。

    在仿真中，关节刚度通常用于建模弹性驱动器（Series Elastic Actuator, SEA）
    或柔性关节机器人。随机化刚度有助于策略适应不同刚度的关节结构，
    提高对柔性传动系统的泛化能力。

    参数：
        env:            基于管理器的强化学习环境实例。
        env_ids:        需要施加随机化的环境索引张量，若为 None 则作用于全部环境。
        ranges:         随机化范围（Ranges 对象），定义参数采样的上下界。
        asset_cfg:      场景实体的配置信息，用于指定目标机器人及关节子集。
        distribution:   随机分布类型，支持 "uniform"（均匀分布）、"normal"（正态分布）等。
        operation:       随机化操作模式，"abs"（绝对赋值）或 "add"（偏移叠加）。
        axes:            指定需要随机化的自由度轴索引列表，None 表示作用于所有自由度。
        shared_random:   若为 True，同一批次内所有环境共享相同的随机采样值；
                         若为 False，每个环境独立采样。

    返回值：
        None，函数直接在物理模型上原地修改参数。
    """
    _randomize_model_field(
        env,
        env_ids,
        "jnt_stiffness",
        entity_type="joint",
        ranges=ranges,
        distribution=distribution,
        operation=operation,
        asset_cfg=asset_cfg,
        axes=axes,
        shared_random=shared_random,
    )


# 原始别名，允许使用更贴近 MuJoCo 物理模型字段名的调用方式。
jnt_stiffness = joint_stiffness


@requires_model_fields("jnt_range")
def joint_limits(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ranges: Ranges,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    distribution: Distribution | str = "uniform",
    operation: Operation | str = "abs",
    axes: list[int] | None = None,
    shared_random: bool = False,
) -> None:
    """
    随机化关节限位范围（jnt_range）。

    关节限位（joint limits / jnt_range）定义了关节运动的最小和最大角度范围。
    在 MuJoCo 中，jnt_range 是一个 [lower, upper] 的区间，
    物理引擎会在关节角度到达限位边界时施加约束力。

    随机化关节限位可以模拟：
        - 机械硬限位（hard stop）的制造公差
        - 不同机器人型号的工作空间差异
        - 限位挡块位置的不确定性

    策略在限位随机化下训练后，能够更安全地在接近关节限位处运行，
    避免在真实硬件上因限位位置偏差而导致的机械碰撞或电机堵转。

    参数：
        env:            基于管理器的强化学习环境实例。
        env_ids:        需要施加随机化的环境索引张量，若为 None 则作用于全部环境。
        ranges:         随机化范围（Ranges 对象），定义参数采样的上下界。
        asset_cfg:      场景实体的配置信息，用于指定目标机器人及关节子集。
        distribution:   随机分布类型，支持 "uniform"（均匀分布）、"normal"（正态分布）等。
        operation:       随机化操作模式，"abs"（绝对赋值）或 "add"（偏移叠加）。
        axes:            指定需要随机化的自由度轴索引列表，None 表示作用于所有自由度。
        shared_random:   若为 True，同一批次内所有环境共享相同的随机采样值；
                         若为 False，每个环境独立采样。

    返回值：
        None，函数直接在物理模型上原地修改参数。
    """
    _randomize_model_field(
        env,
        env_ids,
        "jnt_range",
        entity_type="joint",
        ranges=ranges,
        distribution=distribution,
        operation=operation,
        asset_cfg=asset_cfg,
        axes=axes,
        shared_random=shared_random,
    )


# 原始别名，允许使用更贴近 MuJoCo 物理模型字段名的调用方式。
jnt_range = joint_limits


@requires_model_fields("qpos0", recompute=RecomputeLevel.set_const_0)
def joint_default_pos(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ranges: Ranges,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    distribution: Distribution | str = "uniform",
    operation: Operation | str = "add",
    axes: list[int] | None = None,
    shared_random: bool = False,
) -> None:
    """
    随机化关节默认位置（qpos0）。

    qpos0 是 MuJoCo 中每个关节的初始/默认广义坐标位置。
    在环境重置（reset）时，机器人的关节角度会被设定为 qpos0 加上可能的扰动。

    随机化默认关节位置（operation 默认为 "add"，即叠加偏移）可以：
        - 打破训练初期策略对固定初始姿态的过拟合
        - 模拟机器人在真实部署中可能遇到的任意起始构型
        - 提高策略在整个状态空间中的覆盖率

    .. note::
        该操作会触发 ``RecomputeLevel.set_const_0`` 级别的物理量重计算，
        因为 qpos0 的变化会改变初始运动学链，所有依赖 qpos0 的派生量
        （如初始笛卡尔坐标、雅可比矩阵等）都需要重新计算。

    参数：
        env:            基于管理器的强化学习环境实例。
        env_ids:        需要施加随机化的环境索引张量，若为 None 则作用于全部环境。
        ranges:         随机化范围（Ranges 对象），定义参数采样的上下界。
        asset_cfg:      场景实体的配置信息，用于指定目标机器人及关节子集。
        distribution:   随机分布类型，支持 "uniform"（均匀分布）、"normal"（正态分布）等。
        operation:       随机化操作模式，默认为 "add"（偏移叠加），
                         也可使用 "abs"（绝对赋值）。
        axes:            指定需要随机化的自由度轴索引列表，None 表示作用于所有自由度。
        shared_random:   若为 True，同一批次内所有环境共享相同的随机采样值；
                         若为 False，每个环境独立采样。

    返回值：
        None，函数直接在物理模型上原地修改参数。
    """
    _randomize_model_field(
        env,
        env_ids,
        "qpos0",
        entity_type="joint",
        ranges=ranges,
        distribution=distribution,
        operation=operation,
        asset_cfg=asset_cfg,
        axes=axes,
        shared_random=shared_random,
        use_address=True,
    )


# 原始别名，允许使用更贴近 MuJoCo 物理模型字段名的调用方式。
qpos0 = joint_default_pos


def encoder_bias(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    bias_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """
    随机化编码器偏置（encoder bias），用于模拟关节编码器的标定误差。

    在真实机器人系统中，关节角度传感器（编码器）在安装和标定过程中
    可能引入系统性偏置误差。此函数通过在编码器读数上叠加一个随机偏置来模拟这一现象：
    θ_read = θ_true + bias，其中 bias 在 [bias_range[0], bias_range[1]] 内均匀采样。

    与其他随机化函数不同，encoder_bias 直接通过 ``sample_uniform`` 生成随机样本
    并赋值给资产的 ``encoder_bias`` 数据字段，而不经过 ``_randomize_model_field``。
    这是因为编码器偏置属于传感器级参数而非物理模型参数。

    参数：
        env:            基于管理器的强化学习环境实例。
        env_ids:        需要施加随机化的环境索引张量，若为 None 则作用于全部环境。
        bias_range:     偏置采样范围元组 (min, max)，单位取决于具体关节类型
                        （旋转关节为弧度，平移关节为米）。
        asset_cfg:      场景实体的配置信息，用于指定目标机器人及关节子集。

    返回值：
        None，函数直接在资产的 encoder_bias 属性上原地修改。

    实现细节：
        - 支持通过 slice 或列表指定目标关节子集
        - 每个环境 × 每个关节独立采样，保证多样性
        - 使用 ``sample_uniform`` 工具函数进行均匀采样，确保设备一致性和效率
    """
    # 获取目标场景实体（机器人资产）。
    asset: Entity = env.scene[asset_cfg.name]

    # 若未指定环境 ID，则默认作用于全部并行环境。
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    # 解析目标关节索引：支持 slice（全选）或显式列表两种模式。
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        num_joints = asset.num_joints
        joint_ids_tensor = torch.arange(num_joints, device=env.device)
    else:
        joint_ids_tensor = torch.tensor(joint_ids, device=env.device)

    num_joints = len(joint_ids_tensor)
    # 在指定范围内均匀采样偏置值，形状为 (num_envs, num_joints)。
    bias_samples = sample_uniform(
        torch.tensor(bias_range[0], device=env.device),
        torch.tensor(bias_range[1], device=env.device),
        (len(env_ids), num_joints),
        env.device,
    )

    # 根据关节索引的类型采用不同的赋值路径：
    # - slice 模式：直接按环境维度赋值（覆盖全部关节）
    # - 列表模式：按 (env, joint) 二维索引赋值（仅覆盖指定关节）
    if isinstance(joint_ids, slice):
        asset.data.encoder_bias[env_ids] = bias_samples
    else:
        asset.data.encoder_bias[env_ids[:, None], joint_ids_tensor] = bias_samples
