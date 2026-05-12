"""
在 global frame（世界坐标系）采样目标球速 (vx, vy)
每个 episode 开始时随机采样，每 3-8 秒重新采样
不需要 heading command（球速指令不涉及机器人朝向）
保留 standing envs（零指令让机器人静止）和 world envs（世界坐标系指令）
"""
#让类型注解中的 `list[int]`、`dict[str, float]` 等新式泛型语法在 Python 3.9 也能用
from __future__ import annotations

from collections.abc import Callable
# @dataclass: 把类变成数据容器，自动生成 __init__/__repr__/__eq__
# field(): 给 dataclass 字段设默认值（如 field(default_factory=VizCfg)）
from dataclasses import dataclass, field
# TYPE_CHECKING: 运行时永远是 False，只在 IDE/mypy 类型检查时为 True
# 用来包裹"只用于类型注解"的 import，避免循环导入和运行时加载不需要的模块
from typing import TYPE_CHECKING, Any

import torch  # 张量计算，指令用 torch tensor 存在 GPU 上
import numpy as np  # 数值计算，调试可视化时用来处理 numpy 数组

from mjlab.entity import Entity # 场景中的实体（机器人、球），可以读取位置/速度/姿态数据
# CommandTerm: 所有指令的基类，定义 resample/update/compute 生命周期
# CommandTermCfg: 指令配置的基类（dataclass），定义通用参数如 resampling_time_range
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
# wrap_to_pi: 把角度值 wrap 到 [-π, π]，ball_command 其实不需要这个（只有 heading 指令才用）
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
    import viser # Web 3D 可视化服务器，用于创建 GUI 控制面板（摇杆滑块）

    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv  # 主环境类，CommandTerm.__init__ 需要 env 参数
    from mjlab.viewer.debug_visualizer import DebugVisualizer  # 调试可视化器，用于画 3D 箭头

class BallVelocityCommand(CommandTerm):
    """
    继承 CommandTerm，获得 _resample_command/_update_command/_update_metrics等生命周期钩子
    
    球速指令：在全局坐标系中采样目标球速度 (vx, vy)。

    指令是世界坐标系中的 2D 线速度。策略学习踢球使球的速度匹配指令，
    而不直接控制机器人身体的速度。
    """
    cfg: BallVelocityCommandCfg

    def __init__(self, cfg:BallVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]  # 机器人实体，用于读取球位置/速度等数据

        # ball_vel_cmd: [B, 2]，每个环境的目标球速度 (vx, vy)，在 GPU 上的 torch tensor
        self.ball_vel_cmd = torch.zeros(self.num_envs, 2, device=env.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=env.device)  # 是否是 standing envs（零指令）

        #误差评估函数初始化        
        self.metrics["error_ball_vel"] = torch.zeros(self.num_envs, device=self.device)

        # GUI 控件（Viser 查看器激活时使用）
        self._joystick_enabled: viser.GuiCheckboxHandle | None = None
        self._joystick_sliders: list[viser.GuiSliderHandle] = []
        self._joystick_get_env_idx: Callable[[], int] | None = None
        self._on_change: Callable[[], None] | None = None

    @property
    def command(self) -> torch.Tensor:
        """返回指令：[B, 2] 世界坐标系中的目标球速度 (vx, vy)。"""
        return self.ball_vel_cmd
    
    def _update_metrics(self) -> None:
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # 球速度跟踪误差（需要 ball 实体存在） scene 没有get()方法，所以直接访问 _entities 字典
        ball_entity = self._env.scene["ball"] if "ball" in self._env.scene._entities else None
        if ball_entity is not None:
            ball_lin_vel_w = ball_entity.data.root_link_lin_vel_w[:, :2]
            self.metrics["error_ball_vel"] += (
                torch.norm(self.ball_vel_cmd - ball_lin_vel_w, dim=-1) / max_command_step)
                # 每步累积误差 += ‖ 指令球速 - 实际球速 ‖ / 指令周期步数
                # 其中 ‖·‖ 是 L2 范数：√((vx_cmd - vx)² + (vy_cmd - vy)²)



    def _resample_command(self, env_ids: torch.Tensor) -> None:
        r = torch.empty(len(env_ids), device=self.device)
        # 在世界坐标系中采样球速度
        self.ball_vel_cmd[env_ids, 0] = r.uniform_(*self.cfg.ranges.ball_vel_x)
        self.ball_vel_cmd[env_ids, 1] = r.uniform_(*self.cfg.ranges.ball_vel_y)

        # standing envs: 零指令，机器人静止不踢球
        self.is_standing_env[env_ids] = (
            r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        )        
    
    def _update_command(self) -> None:
        # standing envs: 将指令清零
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.ball_vel_cmd[standing_env_ids, :] = 0.0
    # ---- GUI ----

    def create_gui(
        self,
        name: str,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        from viser import Icon

        ranges = self.cfg.ranges

        with server.gui.add_folder(name.capitalize()):
            enabled = server.gui.add_checkbox("Enable", initial_value=False)

            max_vx = server.gui.add_slider(
                "Max ball_vx", initial_value=ranges.ball_vel_x[1],
                step=0.1, min=0.1, max=5.0,
            )
            max_vy = server.gui.add_slider(
                "Max ball_vy", initial_value=ranges.ball_vel_y[1],
                step=0.1, min=0.1, max=5.0,
            )
            slider_vx = server.gui.add_slider(
                "ball_vx", min=-ranges.ball_vel_x[1],
                max=ranges.ball_vel_x[1], step=0.05, initial_value=0.0,
            )
            slider_vy = server.gui.add_slider(
                "ball_vy", min=-ranges.ball_vel_y[1],
                max=ranges.ball_vel_y[1], step=0.05, initial_value=0.0,
            )

            @max_vx.on_update
            def _(_ev, _s=slider_vx, _m=max_vx) -> None:
                _s.min = -_m.value
                _s.max = _m.value

            @max_vy.on_update
            def _(_ev, _s=slider_vy, _m=max_vy) -> None:
                _s.min = -_m.value
                _s.max = _m.value

            zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

            @zero_btn.on_click
            def _(_) -> None:
                slider_vx.value = 0.0
                slider_vy.value = 0.0

        self._joystick_enabled = enabled
        self._joystick_sliders = [slider_vx, slider_vy]
        self._joystick_get_env_idx = get_env_idx
        self._on_change = on_change

    def compute(self, dt: float) -> None:
        super().compute(dt)
        if self._joystick_enabled is not None and self._joystick_enabled.value:
            assert self._joystick_get_env_idx is not None
            idx = self._joystick_get_env_idx()
            for i, s in enumerate(self._joystick_sliders):
                self.ball_vel_cmd[idx, i] = s.value

    # ---- Visualization ----
    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """在世界坐标系中绘制目标球速度箭头。"""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        from mjlab.utils.lab_api.math import matrix_from_quat

        cmds = self.command.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        base_quat_w = self.robot.data.root_link_quat_w
        base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()

        scale = self.cfg.viz.scale
        z_offset = self.cfg.viz.z_offset

        for batch in env_indices:
            base_pos_w = base_pos_ws[batch]
            base_mat_w = base_mat_ws[batch]
            cmd = cmds[batch]
            if abs(cmd[0]) < 1e-6 and abs(cmd[1]) < 1e-6:
                continue

            # 世界坐标系球速箭头（橙色）
            cmd_from = base_pos_w + base_mat_w @ np.array([0, 0, z_offset]) * scale
            cmd_to = cmd_from + np.array([cmd[0], cmd[1], 0]) * scale
            visualizer.add_arrow(
                cmd_from, cmd_to, color=(1.0, 0.5, 0.0, 0.8), width=0.015
            )

@dataclass(kw_only=True)
class BallVelocityCommandCfg(CommandTermCfg):
    """球速指令的配置数据类。"""

    entity_name: str
    """场景实体名称（用于获取机器人位置以进行可视化）。"""

    rel_standing_envs: float = 0.0
    """接收零指令（站立不动）的环境比例。"""

    @dataclass
    class Ranges:
        ball_vel_x: tuple[float, float]
        """世界坐标系 X 方向球速范围 (min, max) [m/s]。"""
        ball_vel_y: tuple[float, float]
        """世界坐标系 Y 方向球速范围 (min, max) [m/s]。"""

    ranges: Ranges

    @dataclass
    class VizCfg:
        z_offset: float = 0.2
        scale: float = 0.5

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> BallVelocityCommand:
        return BallVelocityCommand(self, env)