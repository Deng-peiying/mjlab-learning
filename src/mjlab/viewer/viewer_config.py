import enum
from dataclasses import dataclass


@dataclass
class ViewerConfig:
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    distance: float = 5.0
    fovy: float | None = None
    elevation: float = -45.0
    azimuth: float = 90.0

    class OriginType(enum.Enum):
        """相机位置和目标的参考坐标系。"""

        AUTO = enum.auto()
        """跟踪第一个非固定基座的物体，若无则回退为自由相机。"""
        WORLD = enum.auto()
        """位于所配置观察点的自由相机。"""
        ASSET_ROOT = enum.auto()
        """跟踪由 entity_name 指定的资产的根刚体。"""
        ASSET_BODY = enum.auto()
        """跟踪由 entity_name 指定的资产中由 body_name 指定的刚体。"""

    origin_type: OriginType = OriginType.AUTO
    entity_name: str | None = None
    body_name: str | None = None
    env_idx: int = 0
    max_extra_envs: int = 2
    """在 ``env_idx`` 周围渲染的相邻环境数量。"""
    enable_reflections: bool = True
    enable_shadows: bool = True
    height: int = 240
    width: int = 320
