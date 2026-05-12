"""传感器基类接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import mujoco
import mujoco_warp as mjwarp
import torch

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.viewer.debug_visualizer import DebugVisualizer


T = TypeVar("T")


@dataclass
class SensorCfg(ABC):
    """传感器的基础配置。"""

    name: str

    @property
    def prefixed_name(self) -> str:
        """传感器在 MuJoCo 模型中的名称。"""
        return self.name

    @abstractmethod
    def build(self) -> Sensor[Any]:
        """从该配置构建传感器实例。"""
        raise NotImplementedError


class Sensor(ABC, Generic[T]):
    """带有类型化数据和逐步缓存的传感器基类接口。

    类型参数 T 指定传感器返回的数据类型。例如：
    - Sensor[torch.Tensor] 用于返回原始张量的传感器
    - Sensor[ContactData] 用于返回结构化接触数据的传感器

    子类应注意：
    - 在 ``__init__`` 方法中调用 ``super().__init__()``
    - 如果重写 ``reset()`` 或 ``update()``，先调用 ``super()`` 以失效缓存
    """

    requires_sensor_context: bool = False
    """此传感器是否需要 SensorContext（渲染上下文）。"""

    def __init__(self) -> None:
        self._cached_data: T | None = None
        self._cache_valid: bool = False

    @abstractmethod
    def edit_spec(
        self,
        scene_spec: mujoco.MjSpec,
        entities: dict[str, Entity],
    ) -> None:
        """编辑场景 spec 以添加此传感器。

        在场景构建期间调用，用于向 MjSpec 添加传感器元素。

        Args:
          scene_spec: 要编辑的场景 MjSpec。
          entities: 场景中的实体字典，以名称为键。
        """
        raise NotImplementedError

    @abstractmethod
    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        """在模型编译后初始化传感器。

        在 MjSpec 编译为 MjModel 且仿真准备运行后调用。
        用于缓存传感器索引、分配缓冲区等。

        Args:
          mj_model: 编译后的 MuJoCo 模型。
          model: mjwarp 模型封装。
          data: mjwarp 数据数组。
          device: 张量运算的设备（例如 "cuda"、"cpu"）。
        """
        raise NotImplementedError

    @property
    def data(self) -> T:
        """获取当前传感器数据，如果缓存可用则使用缓存值。

        此属性以特定类型返回传感器的当前数据。
        数据类型由类型参数 T 指定。数据按步缓存，
        仅在缓存失效后（调用 ``reset()`` 或 ``update()`` 后）重新计算。

        Returns:
          格式由类型参数 T 指定的传感器数据。
        """
        if not self._cache_valid:
            self._cached_data = self._compute_data()
            self._cache_valid = True
        assert self._cached_data is not None
        return self._cached_data

    @abstractmethod
    def _compute_data(self) -> T:
        """计算并返回传感器数据。

        子类必须实现此方法来计算传感器数据。
        当缓存失效时，由 ``data`` 属性调用此方法。

        Returns:
          计算得到的传感器数据。
        """
        raise NotImplementedError

    def _invalidate_cache(self) -> None:
        """使缓存数据失效，强制在下次访问时重新计算。"""
        self._cache_valid = False

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """重置指定环境的传感器状态。

        使数据缓存失效。在维护内部状态的子类中重写，
        但**先**调用 ``super().reset(env_ids)``。

        Args:
          env_ids: 要重置的环境索引。若为 None，则重置所有环境。
        """
        del env_ids  # 未使用。
        self._invalidate_cache()

    def update(self, dt: float) -> None:
        """在仿真步之后更新传感器状态。

        使数据缓存失效。在需要逐步更新的子类中重写，
        但**先**调用 ``super().update(dt)``。

        Args:
          dt: 时间步长（秒）。
        """
        del dt  # 未使用。
        self._invalidate_cache()

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        """可视化传感器数据用于调试。

        基类实现不做任何操作。在支持调试可视化的子类中重写。

        Args:
          visualizer: 要绘制到的调试可视化器。
        """
        del visualizer  # 未使用。
