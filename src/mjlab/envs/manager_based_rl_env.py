import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import torch
import warp as wp
from prettytable import PrettyTable

from mjlab.envs import types
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.action_manager import ActionManager, ActionTermCfg
from mjlab.managers.command_manager import (
    CommandManager,
    CommandTermCfg,
    NullCommandManager,
)
from mjlab.managers.curriculum_manager import (
    CurriculumManager,
    CurriculumTermCfg,
    NullCurriculumManager,
)
from mjlab.managers.event_manager import EventManager, EventTermCfg
from mjlab.managers.metrics_manager import (
    MetricsManager,
    MetricsTermCfg,
    NullMetricsManager,
)
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationManager
from mjlab.managers.recorder_manager import (
    NullRecorderManager,
    RecorderManager,
    RecorderTermCfg,
)
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from mjlab.scene import Scene
from mjlab.scene.scene import SceneCfg
from mjlab.sim import SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.utils import random as random_utils
from mjlab.utils.logging import print_info
from mjlab.utils.spaces import Box
from mjlab.utils.spaces import Dict as DictSpace
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


@dataclass(kw_only=True)
class ManagerBasedRlEnvCfg:
    """基于管理器的强化学习（RL）环境配置。

    该配置定义 RL 环境的全部组成：物理场景、观测、动作、奖励、终止条件，
    以及命令与课程学习等可选特性。

    环境步长为 ``sim.mujoco.timestep * decimation``。例如，物理仿真步长为 2ms 且
    decimation=10 时，环境以 50Hz 运行。
    """

    # 基础环境配置。

    decimation: int
    """每个环境步对应的物理仿真子步数。数值越大，控制频率越低。
    环境步持续时间 = physics_dt * decimation。"""

    scene: SceneCfg
    """场景配置，用于定义地形、实体与传感器。该场景会指定 ``num_envs``，
    即并行环境的数量。"""

    observations: dict[str, ObservationGroupCfg] = field(default_factory=dict)
    """观测组配置。每个组（例如 "actor"、"critic"）包含会被拼接的观测项（term）。
    不同组可以有不同的噪声、历史与延迟设置。"""

    actions: dict[str, ActionTermCfg] = field(default_factory=dict)
    """动作项（term）配置。每个项控制一个特定实体/方面（例如关节位置）。
    动作维度会在各项之间拼接。"""

    events: dict[str, EventTermCfg] = field(
        default_factory=lambda: {
            "reset_scene_to_default": EventTermCfg(
                func=reset_scene_to_default,
                mode="reset",
            )
        }
    )
    """用于领域随机化与状态重置的事件项（term）。默认包含 ``reset_scene_to_default``，
    它会将实体重置到初始状态。可设为空以禁用所有事件（包括默认重置）。"""

    seed: int | None = None
    """用于可复现性的随机种子。若为 None，则使用随机种子。初始化后，实际使用的
    seed 会回写到该字段。"""

    sim: SimulationCfg = field(default_factory=SimulationCfg)
    """仿真配置，包括物理时间步、求解器迭代次数、接触参数与 NaN 防护。"""

    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    """渲染查看器配置（相机位置、分辨率等）。"""

    # RL 专用配置。

    episode_length_s: float = 0.0
    """每个 episode 的持续时间（秒）。

    episode 的步数计算为：
        ceil(episode_length_s / (sim.mujoco.timestep * decimation))
    """

    rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
    """奖励项（term）配置。"""

    terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)
    """终止项（term）配置。若为空，则 episode 永不重置。
    可使用 ``mdp.time_out`` 且 ``time_out=True`` 来设置 episode 的时间上限。"""

    commands: dict[str, CommandTermCfg] = field(default_factory=dict)
    """命令生成项（例如速度目标）。"""

    curriculum: dict[str, CurriculumTermCfg] = field(default_factory=dict)
    """用于自适应难度的课程学习项。"""

    metrics: dict[str, MetricsTermCfg] = field(default_factory=dict)
    """自定义指标项：用于把按步记录的值以 episode 平均值的形式写入日志。"""

    recorders: dict[str, RecorderTermCfg] = field(default_factory=dict)
    """Recorder 项：用于在 rollout 期间记录观测、动作或其他数据。
    若为空，则使用无操作（no-op）管理器且没有额外开销。"""

    is_finite_horizon: bool = False
    """任务是否为有限视界或无限视界。默认 False（无限视界）。

    - **有限视界 (True)**：时间上限定义任务边界。到达上限时，上限之后不再存在
        未来价值，因此智能体接收到 terminal done 信号。
    - **无限视界 (False)**：时间上限只是人为截断。智能体接收到 truncated done 信号，
        用于对"超过上限继续执行"的价值进行 bootstrap。
    """

    auto_reset: bool = True
    """是否自动重置已终止或超时的环境。

    当为 True（默认）时，``step()`` 会就地重置 done 的环境，并返回重置后的观测。
    当为 False 时，``step()`` 返回真实的终止观测；调用方必须在下一次 ``step()``
    之前，对 done 的环境显式调用 ``reset(env_ids=...)``。

    注意：mjlab 自带的 ``train.py`` 通过 rsl_rl 的 ``OnPolicyRunner`` 运行，
    该 runner 不会驱动手动 reset。``auto_reset=False`` 适用于用户自行编写训练循环
    （或使用在 step 之间负责 reset 的封装器）。
    """

    scale_rewards_by_dt: bool = True
    """是否将奖励乘以环境步持续时间（dt）。

    当为 True（默认）时，奖励值会乘以 step_dt，以在不同仿真频率下归一化 episode
    累积奖励。若算法期望未缩放的奖励信号（例如 HER、静态奖励缩放），可将其设为 False。
    """


class ManagerBasedRlEnv:
    """基于管理器的强化学习（RL）环境。"""

    is_vector_env = True
    metadata = {
        "render_modes": [None, "rgb_array"],
        "mujoco_version": mujoco.__version__,
        "warp_version": wp.config.version,
    }
    cfg: ManagerBasedRlEnvCfg

    def __init__(
        self,
        cfg: ManagerBasedRlEnvCfg,
        device: str,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        # 初始化基础环境状态。
        self.cfg = cfg
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed, device=device)
        self._sim_step_counter = 0
        self.extras = {}
        self.obs_buf = {}
        self._manual_reset_pending = torch.zeros(
            self.cfg.scene.num_envs, dtype=torch.bool, device=device
        )

        # 初始化场景与仿真。
        self.scene = Scene(self.cfg.scene, device=device)
        self.sim = Simulation(
            num_envs=self.scene.num_envs,
            cfg=self.cfg.sim,
            spec=self.scene.spec,
            variant_info=self.scene.collect_variant_info(),
            device=device,
        )

        self.scene.initialize(
            mj_model=self.sim.mj_model,
            model=self.sim.model,
            data=self.sim.data,
        )

        # 将传感器上下文接入仿真，用于 sense_graph。
        if self.scene.sensor_context is not None:
            self.sim.set_sensor_context(self.scene.sensor_context)

        # 打印环境信息。
        print_info("")
        table = PrettyTable()
        table.title = "Base Environment"
        table.field_names = ["Property", "Value"]
        table.align["Property"] = "l"
        table.align["Value"] = "l"
        table.add_row(["Number of environments", self.num_envs])
        table.add_row(["Environment device", self.device])
        table.add_row(["Environment seed", self.cfg.seed])
        table.add_row(["Physics step-size", self.physics_dt])
        table.add_row(["Environment step-size", self.step_dt])
        print_info(table.get_string())
        print_info("")

        # 初始化 RL 相关状态。
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(
            cfg.scene.num_envs, device=device, dtype=torch.long
        )
        self.render_mode = render_mode
        self._offline_renderer: OffscreenRenderer | None = None
        if self.render_mode == "rgb_array":
            renderer = OffscreenRenderer(
                model=self.sim.mj_model,
                cfg=self.cfg.viewer,
                scene=self.scene,
                sim_model=self.sim.model,
                expanded_fields=self.sim.expanded_fields,
            )
            renderer.initialize()
            self._offline_renderer = renderer
        self.metadata["render_fps"] = 1.0 / self.step_dt

        # 加载所有管理器。
        self.load_managers()
        self.setup_manager_visualizers()

    # 属性。

    @property
    def num_envs(self) -> int:
        """并行环境数量。"""
        return self.scene.num_envs

    @property
    def physics_dt(self) -> float:
        """物理仿真步长。"""
        return self.cfg.sim.mujoco.timestep

    @property
    def step_dt(self) -> float:
        """环境步长（physics_dt * decimation）。"""
        return self.cfg.sim.mujoco.timestep * self.cfg.decimation

    @property
    def device(self) -> str:
        """计算所用设备。"""
        return self.sim.device

    @property
    def max_episode_length_s(self) -> float:
        """episode 最大时长（秒）。"""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """episode 最大步数。"""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    @property
    def unwrapped(self) -> "ManagerBasedRlEnv":
        """获取未包装（unwrapped）的环境（wrapper 链的基类）。"""
        return self

    # 方法。

    def setup_manager_visualizers(self) -> None:
        self.manager_visualizers = {}
        if getattr(self.command_manager, "active_terms", None):
            self.manager_visualizers["command_manager"] = self.command_manager
        self.manager_visualizers["event_manager"] = self.event_manager
        self.manager_visualizers["reward_manager"] = self.reward_manager

    def load_managers(self) -> None:
        """加载并初始化所有管理器。

        加载顺序很重要！必须先加载事件（event）和命令（command）管理器，
        然后是动作（action）和观测（observation）管理器，最后再加载其他 RL 管理器。
        """
        # 事件管理器（用于领域随机化，必须在其他一切之前初始化）。
        self.event_manager = EventManager(self.cfg.events, self)
        print_info(f"[INFO] {self.event_manager}")

        self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)

        # 命令管理器（必须在观测管理器之前初始化，因为观测
        # 可能会引用命令）。
        if len(self.cfg.commands) > 0:
            self.command_manager = CommandManager(self.cfg.commands, self)
        else:
            self.command_manager = NullCommandManager()
        print_info(f"[INFO] {self.command_manager}")

        # 动作与观测管理器。
        self.action_manager = ActionManager(self.cfg.actions, self)
        print_info(f"[INFO] {self.action_manager}")
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print_info(f"[INFO] {self.observation_manager}")

        # 其他 RL 相关管理器。

        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print_info(f"[INFO] {self.termination_manager}")
        self.reward_manager = RewardManager(
            self.cfg.rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
        )
        print_info(f"[INFO] {self.reward_manager}")
        if len(self.cfg.curriculum) > 0:
            self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        else:
            self.curriculum_manager = NullCurriculumManager()
        print_info(f"[INFO] {self.curriculum_manager}")
        if len(self.cfg.metrics) > 0:
            self.metrics_manager = MetricsManager(self.cfg.metrics, self)
        else:
            self.metrics_manager = NullMetricsManager()
        print_info(f"[INFO] {self.metrics_manager}")
        if len(self.cfg.recorders) > 0:
            self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        else:
            self.recorder_manager = NullRecorderManager()
        print_info(f"[INFO] {self.recorder_manager}")

        # 配置环境的空间（spaces）。
        self._configure_gym_env_spaces()

        # 如果定义了 startup 事件，则在启动时执行。
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def reset(
        self,
        *,
        seed: int | None = None,
        env_ids: torch.Tensor | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[types.VecEnvObs, dict]:
        del options  # 未使用。
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        if seed is not None:
            self.seed(seed)
        self.extras["log"] = dict()
        self._reset_idx(env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.command_manager.compute(dt=0.0)
        self.sim.sense()
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self.recorder_manager.record_post_reset(env_ids)
        return self.obs_buf, self.extras

    def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
        """执行一次环境 step：应用动作、推进仿真、计算 RL 信号。

        当 ``auto_reset=True``（默认）时，已终止或超时的环境会就地被重置，返回的观测
        为重置后的状态。当 ``auto_reset=False`` 时，会跳过重置，返回的观测为终止状态；
        调用方必须在下一次 ``step()`` 之前，对 done 的环境调用 ``reset(env_ids=...)``。

        **forward() 调用位置。** MuJoCo 的 ``mj_step`` 在积分（integration）*之前*
        执行正向运动学，因此在 step 之后，派生量（``xpos``、``xquat``、``site_xpos``、
        ``cvel``、``sensordata``）相对于 ``qpos``/``qvel`` 会滞后一个物理子步。
        与其调用两次 ``sim.forward()``（一次在 decimation 循环后、一次在 reset 逻辑后），
        本方法只在计算观测之前调用 **一次**。这一次调用会为 *所有* 环境刷新派生量：
        未重置环境获得 decimation 后的运动学结果，已重置环境获得 reset 后的运动学结果。

        代价是终止与奖励管理器看到的派生量会滞后一个物理子步（最后一次 ``mj_step``
        是从积分前的 ``qpos`` 执行 ``mj_forward``）。在实践中，这种滞后对奖励 shaping
        与终止检查的影响通常可以忽略。更关键的是，这种滞后是*一致的*：每个环境、每一步
        都具有同样的滞后，因此 MDP 定义良好，价值函数可以学习到正确的映射关系。

        .. note::

            事件与命令的作者不需要自行调用 ``sim.forward()``，本方法会处理。
            唯一约束是：不要在同一个既写入状态（``write_root_state_to_sim``、
            ``write_joint_state_to_sim`` 等）又读取派生量（``root_link_pose_w``、
            ``body_link_vel_w`` 等）的函数中读取派生量。
            详情参见 :ref:`faq`。
        """
        if not self.cfg.auto_reset and torch.any(self._manual_reset_pending):
            pending_ids = self._manual_reset_pending.nonzero(as_tuple=False).squeeze(-1)
            raise RuntimeError(
                f"Environments {pending_ids.cpu().tolist()} must be reset via "
                "reset(env_ids=...) before calling step() again when auto_reset=False."
            )

        self.extras["log"] = dict()
        self.action_manager.process_action(action.to(self.device))

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=self.physics_dt)
            self.metrics_manager.compute_substep()

        # 更新环境计数器。
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # 检查终止条件并计算奖励。
        # NOTE：此处派生量（xpos、xquat 等）会滞后一个物理子步。
        # 为什么这可以接受，请见上面的 docstring。
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self.metrics_manager.compute()

        # 重置已终止/超时的环境，并记录 episode 信息。
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if self.cfg.auto_reset and len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()

        # 仅调用一次 forward()：基于当前 qpos/qvel 为所有环境重算派生量。
        # 对未重置环境，这会消除 mj_step 留下的一个子步滞后；对已重置环境，
        # 则会读取刚写入的重置状态。
        self.sim.forward()

        self.command_manager.compute(dt=self.step_dt)

        if "step" in self.event_manager.available_modes:
            self.event_manager.apply(mode="step", dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.sim.sense()
        self.obs_buf = self.observation_manager.compute(update_history=True)

        if self.cfg.auto_reset and len(reset_env_ids) > 0:
            self.recorder_manager.record_post_reset(reset_env_ids)
        elif len(reset_env_ids) > 0:
            self._manual_reset_pending[reset_env_ids] = True

        self.recorder_manager.record_post_step()

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            if self._offline_renderer is None:
                raise ValueError("Offline renderer not initialized")
            debug_callback = (
                self.update_visualizers if hasattr(self, "update_visualizers") else None
            )
            self._offline_renderer.update(
                self.sim.data, debug_vis_callback=debug_callback
            )
            return self._offline_renderer.render()
        else:
            raise NotImplementedError(
                f"Render mode {self.render_mode} is not supported. "
                f"Please use: {self.metadata['render_modes']}."
            )

    def close(self) -> None:
        if self._offline_renderer is not None:
            self._offline_renderer.close()
        self.recorder_manager.close()

    def seed(self, seed: int = -1, device: str | torch.device | None = None) -> int:
        if seed == -1:
            seed = np.random.randint(0, 10_000)
        print_info(f"Setting seed: {seed}")
        random_utils.seed_rng(
            seed, device=device if device is not None else self.device
        )
        return seed

    def update_visualizers(self, visualizer: DebugVisualizer) -> None:
        for mod in self.manager_visualizers.values():
            mod.debug_vis(visualizer)
        for sensor in self.scene.sensors.values():
            sensor.debug_vis(visualizer)

    # 私有方法。

    def _configure_gym_env_spaces(self) -> None:
        from mjlab.utils.spaces import batch_space

        self.single_observation_space = DictSpace()
        for (
            group_name,
            group_term_names,
        ) in self.observation_manager.active_terms.items():
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[
                group_name
            ]
            group_dim = self.observation_manager.group_obs_dim[group_name]
            if has_concatenated_obs:
                assert isinstance(group_dim, tuple)
                self.single_observation_space.spaces[group_name] = Box(
                    shape=group_dim, low=-math.inf, high=math.inf
                )
            else:
                assert not isinstance(group_dim, tuple)
                group_term_cfgs = self.observation_manager._group_obs_term_cfgs[
                    group_name
                ]
                # 为该组创建一个嵌套字典。
                group_space = DictSpace()
                for term_name, term_dim, _term_cfg in zip(
                    group_term_names, group_dim, group_term_cfgs, strict=False
                ):
                    group_space.spaces[term_name] = Box(
                        shape=term_dim, low=-math.inf, high=math.inf
                    )
                self.single_observation_space.spaces[group_name] = group_space

        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = Box(
            shape=(action_dim,), low=-math.inf, high=math.inf
        )

        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        self.curriculum_manager.compute(env_ids=env_ids)
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)

        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(
                mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
            )

        # NOTE：这里对顺序敏感。
        # 观测管理器。
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # 动作管理器。
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # 奖励管理器。
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # 指标管理器。
        info = self.metrics_manager.reset(env_ids)
        self.extras["log"].update(info)
        # 课程学习管理器。
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # 命令管理器。
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # 事件管理器。
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # 终止管理器。
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        # 重置 episode 长度缓冲区。
        self.episode_length_buf[env_ids] = 0
        self._manual_reset_pending[env_ids] = False
