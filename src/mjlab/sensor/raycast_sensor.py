"""地形与障碍物检测的射线传感器。

提供 :class:`RayCastSensor` 和 :class:`RayCastSensorCfg`，基于 BVH 加速的
射线检测，支持栅格、针孔相机、环形三种采样模式。支持多帧挂载、
可配置的射线对齐方式以及 geom 组过滤。

使用指南和示例请参见 :doc:`/sensors/raycast_sensor`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from mujoco_warp import rays

from mjlab.entity import Entity
from mjlab.sensor.builtin_sensor import ObjRef
from mjlab.sensor.sensor import Sensor, SensorCfg
from mjlab.utils.lab_api.math import quat_from_matrix

if TYPE_CHECKING:
    from mjlab.sensor.sensor_context import SensorContext
    from mjlab.viewer.debug_visualizer import DebugVisualizer

RayAlignment = Literal["base", "yaw", "world"]

# 注：mujoco_warp 未公开暴露此类型。
_vec6 = wp.types.vector(length=6, dtype=float)
_ALL_GROUPS = _vec6(-1, -1, -1, -1, -1, -1)


def _geom_groups_to_vec6(groups: tuple[int, ...] | None):  # -> _vec6
    """将 geom 组元组转换为 mujoco_warp 的 vec6 格式。

    在 vec6 格式中，-1 表示包含，0 表示排除。
    """
    if groups is None:
        return _ALL_GROUPS
    out = [0, 0, 0, 0, 0, 0]
    for g in groups:
        if 0 <= g <= 5:
            out[g] = -1
    return _vec6(*out)


@dataclass
class GridPatternCfg:
    """栅格模式 - 2D 网格中的平行射线。"""

    size: tuple[float, float] = (1.0, 1.0)
    """栅格尺寸（长, 宽），单位米。"""

    resolution: float = 0.1
    """射线间距，单位米。"""

    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    """帧局部坐标系中的射线方向。"""

    def generate_rays(
        self, mj_model: mujoco.MjModel | None, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """生成射线模式。

        Args:
          mj_model: MuJoCo 模型（栅格模式下不使用）。
          device: 张量运算设备。

        Returns:
          (local_offsets [N, 3], local_directions [N, 3]) 元组。
        """
        del mj_model  # 栅格模式下不使用
        size_x, size_y = self.size
        res = self.resolution

        x = torch.arange(
            -size_x / 2, size_x / 2 + res * 0.5, res, device=device, dtype=torch.float32
        )
        y = torch.arange(
            -size_y / 2, size_y / 2 + res * 0.5, res, device=device, dtype=torch.float32
        )
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")

        num_rays = grid_x.numel()
        local_offsets = torch.zeros((num_rays, 3), device=device, dtype=torch.float32)
        local_offsets[:, 0] = grid_x.flatten()
        local_offsets[:, 1] = grid_y.flatten()

        # 栅格模式下所有射线共享同一方向。
        direction = torch.tensor(self.direction, device=device, dtype=torch.float32)
        direction = direction / direction.norm()
        local_directions = direction.unsqueeze(0).expand(num_rays, 3).clone()

        return local_offsets, local_directions


@dataclass
class PinholeCameraPatternCfg:
    """针孔相机模式 - 射线从原点发散，类似相机成像。

    可通过显式参数（width, height, fovy）配置，也可通过工厂方法
    from_mujoco_camera() 或 from_intrinsic_matrix() 创建。
    """

    width: int = 16
    """图像宽度（像素）。"""

    height: int = 12
    """图像高度（像素）。"""

    fovy: float = 45.0
    """垂直视场角（度），与 MuJoCo 约定一致。"""

    _camera_name: str | None = field(default=None, repr=False)
    """内部：MuJoCo 相机名称，用于延迟参数解析。"""

    @classmethod
    def from_mujoco_camera(cls, camera_name: str) -> PinholeCameraPatternCfg:
        """创建引用 MuJoCo 相机的配置。

        相机参数（分辨率、视场角）在运行时从模型中解析。

        Args:
          camera_name: 要引用的 MuJoCo 相机名称。

        Returns:
          将从 MuJoCo 相机解析参数的配置。
        """
        # 占位值；实际值在 generate_rays() 中解析。
        return cls(width=0, height=0, fovy=0.0, _camera_name=camera_name)

    @classmethod
    def from_intrinsic_matrix(
        cls, intrinsic_matrix: list[float], width: int, height: int
    ) -> PinholeCameraPatternCfg:
        """从 3x3 内参矩阵 [fx, 0, cx, 0, fy, cy, 0, 0, 1] 创建。

        Args:
          intrinsic_matrix: 展平的 3x3 内参矩阵。
          width: 图像宽度（像素）。
          height: 图像高度（像素）。

        Returns:
          由内参矩阵计算 fovy 的配置。
        """
        fy = intrinsic_matrix[4]  # fy 在矩阵中的位置为 [1,1]
        fovy = 2 * math.atan(height / (2 * fy)) * 180 / math.pi
        return cls(width=width, height=height, fovy=fovy)

    def generate_rays(
        self, mj_model: mujoco.MjModel | None, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """生成射线模式。

        Args:
          mj_model: MuJoCo 模型（使用 from_mujoco_camera 时必需）。
          device: 张量运算设备。

        Returns:
          (local_offsets [N, 3], local_directions [N, 3]) 元组。
        """
        # 解析相机参数。
        if self._camera_name is not None:
            if mj_model is None:
                raise ValueError(
                    "MuJoCo model required when using from_mujoco_camera()"
                )
            # 从 MuJoCo 相机获取参数。
            cam_id = mj_model.camera(self._camera_name).id
            width, height = mj_model.cam_resolution[cam_id]

            # MuJoCo 有两种相机模式：
            # 1. fovy 模式：sensorsize 为零，直接使用 cam_fovy
            # 2. 物理传感器模式：sensorsize > 0，从 focal/sensorsize 计算
            sensorsize = mj_model.cam_sensorsize[cam_id]
            if sensorsize[0] > 0 and sensorsize[1] > 0:
                # 物理传感器模型。
                intrinsic = mj_model.cam_intrinsic[cam_id]  # [fx, fy, cx, cy]
                focal = intrinsic[:2]  # [fx, fy]
                h_fov_rad = 2 * math.atan(sensorsize[0] / (2 * focal[0]))
                v_fov_rad = 2 * math.atan(sensorsize[1] / (2 * focal[1]))
            else:
                # 直接从 MuJoCo 读取垂直视场角。
                v_fov_rad = math.radians(mj_model.cam_fovy[cam_id])
                aspect = width / height
                h_fov_rad = 2 * math.atan(math.tan(v_fov_rad / 2) * aspect)
        else:
            # 使用显式参数。
            width = self.width
            height = self.height
            v_fov_rad = math.radians(self.fovy)
            aspect = width / height
            h_fov_rad = 2 * math.atan(math.tan(v_fov_rad / 2) * aspect)

        # 创建归一化像素坐标 [-1, 1]。
        u = torch.linspace(-1, 1, width, device=device, dtype=torch.float32)
        v = torch.linspace(-1, 1, height, device=device, dtype=torch.float32)
        grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")

        # 转换为射线方向（MuJoCo 相机：-Z 前, +X 右, +Y 下）。
        ray_x = grid_u.flatten() * math.tan(h_fov_rad / 2)
        ray_y = grid_v.flatten() * math.tan(v_fov_rad / 2)
        ray_z = -torch.ones_like(ray_x)  # MuJoCo 相机前方为负 Z

        num_rays = width * height
        local_offsets = torch.zeros((num_rays, 3), device=device)
        local_directions = torch.stack([ray_x, ray_y, ray_z], dim=1)
        local_directions = local_directions / local_directions.norm(dim=1, keepdim=True)

        return local_offsets, local_directions


@dataclass
class RingPatternCfg:
    """环形模式 - 围绕原点的同心圆环射线。

    适用于逐点高度感知，在每个挂载帧周围采样地形。
    """

    @dataclass
    class Ring:
        """模式中的单个圆环。"""

        radius: float
        """圆环半径，单位米。"""

        num_samples: int
        """圆环上均匀分布的采样点数量。"""

    rings: tuple[Ring, ...]
    """圆环定义。多个圆环构成同心模式。"""

    include_center: bool = True
    """是否在中心包含一条射线（零偏移）。"""

    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    """帧局部坐标系中的射线方向。"""

    @classmethod
    def single_ring(
        cls,
        radius: float = 0.1,
        num_samples: int = 8,
        include_center: bool = True,
        direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
    ) -> RingPatternCfg:
        """创建单环模式。

        Args:
          radius: 圆环半径，单位米。
          num_samples: 圆环上均匀分布的采样点数量。
          include_center: 是否包含中心射线。
          direction: 帧局部坐标系中的射线方向。

        Returns:
          包含一个圆环的 RingPatternCfg。
        """
        return cls(
            rings=(cls.Ring(radius, num_samples),),
            include_center=include_center,
            direction=direction,
        )

    def generate_rays(
        self, mj_model: mujoco.MjModel | None, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """生成射线模式。

        Args:
          mj_model: MuJoCo 模型（环形模式下不使用）。
          device: 张量运算设备。

        Returns:
          (local_offsets [N, 3], local_directions [N, 3]) 元组。
        """
        del mj_model
        offsets: list[torch.Tensor] = []

        if self.include_center:
            offsets.append(torch.zeros(3, device=device, dtype=torch.float32))

        for ring in self.rings:
            for i in range(ring.num_samples):
                angle = 2.0 * math.pi * i / ring.num_samples
                offsets.append(
                    torch.tensor(
                        [
                            ring.radius * math.cos(angle),
                            ring.radius * math.sin(angle),
                            0.0,
                        ],
                        device=device,
                        dtype=torch.float32,
                    )
                )

        local_offsets = torch.stack(offsets)  # [N, 3]
        num_rays = local_offsets.shape[0]

        direction = torch.tensor(self.direction, device=device, dtype=torch.float32)
        direction = direction / direction.norm()
        local_directions = direction.unsqueeze(0).expand(num_rays, 3).clone()

        return local_offsets, local_directions


PatternCfg = GridPatternCfg | PinholeCameraPatternCfg | RingPatternCfg


@dataclass
class RayCastData:
    """射线传感器输出数据。

    Note:
      字段为 GPU 缓冲区的视图，在下一次 ``sense()`` 调用之前有效。
    """

    distances: torch.Tensor
    """[B, N] 到命中点的距离。未命中时为 -1。

  N = num_frames * num_rays_per_frame。
  """

    normals_w: torch.Tensor
    """[B, N, 3] 命中点表面法线（世界坐标系）。未命中时为零向量。"""

    hit_pos_w: torch.Tensor
    """[B, N, 3] 命中位置（世界坐标系）。未命中时为射线原点。"""

    pos_w: torch.Tensor
    """[B, 3] 第一帧在世界坐标系中的位置。"""

    quat_w: torch.Tensor
    """[B, 4] 第一帧姿态四元数 (w, x, y, z)。"""

    frame_pos_w: torch.Tensor
    """[B, F, 3] 所有帧在世界坐标系中的位置。"""

    frame_quat_w: torch.Tensor
    """[B, F, 4] 所有帧姿态四元数 (w, x, y, z)。"""


@dataclass
class RayCastSensorCfg(SensorCfg):
    """射线传感器配置。

    支持多种射线模式（栅格、针孔相机、环形）和对齐方式。
    """

    @dataclass
    class VizCfg:
        """调试渲染的可视化设置。"""

        hit_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.8)
        """命中表面的射线 RGBA 颜色。"""

        miss_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.4)
        """未命中的射线 RGBA 颜色。"""

        hit_sphere_color: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 1.0)
        """命中点处绘制的球体 RGBA 颜色。"""

        hit_sphere_radius: float = 0.5
        """命中点处绘制球体的半径（meansize 的倍数）。"""

        show_rays: bool = False
        """是否绘制射线箭头。"""

        show_normals: bool = False
        """是否在命中点处绘制表面法线。"""

        normal_color: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 1.0)
        """表面法线箭头的 RGBA 颜色。"""

        normal_length: float = 5.0
        """表面法线箭头的长度（meansize 的倍数）。"""

    frame: ObjRef | tuple[ObjRef, ...]
    """挂载射线的 body、site 或 geom。

  传入单个 ``ObjRef`` 用于单帧，或传入元组用于多帧感知
  （例如逐足高度传感器）。当 ``exclude_parent_body`` 为 True 时，
  每帧的父刚体独立排除。
  """

    pattern: PatternCfg = field(default_factory=GridPatternCfg)
    """射线模式配置。默认为 GridPatternCfg。"""

    ray_alignment: RayAlignment = "base"
    """射线相对于帧的朝向。仅控制方向；
  射线原点始终为物理帧位置（``site_xpos`` / ``geom_xpos`` / ``body_xpos``）。

  - "base"：完整旋转（默认）。射线随刚体旋转。
  - "yaw"：仅偏航，忽略俯仰/滚动（适用于高度图）。
  - "world"：固定在世界坐标系中，射线始终指向同一方向。
  """

    max_distance: float = 10.0
    """最大射线距离。超过此距离的射线返回 -1。"""

    exclude_parent_body: bool = True
    """排除父刚体参与射线相交测试。"""

    include_geom_groups: tuple[int, ...] | None = (0, 1, 2)
    """参与射线检测的 geom 组 (0-5)。

  默认为 (0, 1, 2)。设为 None 则包含所有组。
  """

    debug_vis: bool = False
    """启用调试可视化。"""

    viz: VizCfg = field(default_factory=VizCfg)
    """可视化设置。"""

    def build(self) -> RayCastSensor:
        return RayCastSensor(self)


class RayCastSensor(Sensor[RayCastData]):
    """地形与障碍物检测的射线传感器。"""

    requires_sensor_context = True

    def __init__(self, cfg: RayCastSensorCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self._data: mjwarp.Data | None = None
        self._model: mjwarp.Model | None = None
        self._mj_model: mujoco.MjModel | None = None
        self._device: str | None = None
        self._wp_device: wp.Device | None = None

        # 逐帧信息：列表元素为 (帧类型, 对象id, 刚体id)。
        self._frame_infos: list[tuple[Literal["body", "site", "geom"], int, int]] = []
        self._num_frames: int = 0
        self._num_rays_per_frame: int = 0

        self._local_offsets: torch.Tensor | None = None
        self._local_directions: torch.Tensor | None = None  # [rays_per_frame, 3]
        self._num_rays: int = 0

        self._ray_pnt: wp.array | None = None
        self._ray_vec: wp.array | None = None
        self._ray_dist: wp.array | None = None
        self._ray_geomid: wp.array | None = None
        self._ray_normal: wp.array | None = None
        self._ray_bodyexclude: wp.array | None = None
        self._geomgroup = _vec6(-1, -1, -1, -1, -1, -1)

        self._distances: torch.Tensor | None = None
        self._normals_w: torch.Tensor | None = None
        self._hit_pos_w: torch.Tensor | None = None
        self._pos_w: torch.Tensor | None = None
        self._quat_w: torch.Tensor | None = None
        self._frame_pos_w: torch.Tensor | None = None
        self._frame_quat_w: torch.Tensor | None = None

        self._cached_world_origins: torch.Tensor | None = None
        self._cached_world_rays: torch.Tensor | None = None
        self._cached_frame_pos: torch.Tensor | None = None
        self._cached_frame_mat: torch.Tensor | None = None

        self._debug_vis_enabled: bool = True
        self._ctx: SensorContext | None = None

    def edit_spec(
        self,
        scene_spec: mujoco.MjSpec,
        entities: dict[str, Entity],
    ) -> None:
        del scene_spec, entities

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        self._data = data
        self._model = model
        self._mj_model = mj_model
        self._device = device
        self._wp_device = wp.get_device(device)
        num_envs = data.nworld

        # 将 frame 规范化为元组。
        frames = self.cfg.frame
        if isinstance(frames, ObjRef):
            frames = (frames,)

        # 解析每帧 ID。
        self._frame_infos = []
        for frame in frames:
            frame_name = frame.prefixed_name()
            info: tuple[Literal["body", "site", "geom"], int, int]
            if frame.type == "body":
                bid = mj_model.body(frame_name).id
                info = ("body", bid, bid)
            elif frame.type == "site":
                sid = mj_model.site(frame_name).id
                info = ("site", sid, int(mj_model.site_bodyid[sid]))
            elif frame.type == "geom":
                gid = mj_model.geom(frame_name).id
                info = ("geom", gid, int(mj_model.geom_bodyid[gid]))
            else:
                raise ValueError(
                    f"RayCastSensor frame must be 'body', 'site', or 'geom', got '{frame.type}'"
                )
            self._frame_infos.append(info)
        self._num_frames = len(self._frame_infos)

        # 生成射线模式。
        pattern = self.cfg.pattern
        self._local_offsets, self._local_directions = pattern.generate_rays(
            mj_model, device
        )
        self._num_rays_per_frame = self._local_offsets.shape[0]
        self._num_rays = self._num_frames * self._num_rays_per_frame

        self._ray_pnt = wp.zeros(
            (num_envs, self._num_rays), dtype=wp.vec3, device=device
        )
        self._ray_vec = wp.zeros(
            (num_envs, self._num_rays), dtype=wp.vec3, device=device
        )
        self._ray_dist = wp.zeros(
            (num_envs, self._num_rays), dtype=float, device=device
        )
        self._ray_geomid = wp.zeros(
            (num_envs, self._num_rays), dtype=int, device=device
        )
        self._ray_normal = wp.zeros(
            (num_envs, self._num_rays), dtype=wp.vec3, device=device
        )

        # 刚体排除：每帧的 body_id 重复 N 次。
        if self.cfg.exclude_parent_body:
            body_excludes: list[int] = []
            for _, _, body_id in self._frame_infos:
                body_excludes.extend([body_id] * self._num_rays_per_frame)
        else:
            body_excludes = [-1] * self._num_rays
        self._ray_bodyexclude = wp.array(body_excludes, dtype=int, device=device)

        self._geomgroup = _geom_groups_to_vec6(self.cfg.include_geom_groups)

        # 预分配输出张量，确保在首次 sense() 调用前形状推导能正常工作。
        F = self._num_frames
        self._distances = torch.zeros(num_envs, self._num_rays, device=device)
        self._normals_w = torch.zeros(num_envs, self._num_rays, 3, device=device)
        self._hit_pos_w = torch.zeros(num_envs, self._num_rays, 3, device=device)
        self._pos_w = torch.zeros(num_envs, 3, device=device)
        self._quat_w = torch.zeros(num_envs, 4, device=device)
        self._frame_pos_w = torch.zeros(num_envs, F, 3, device=device)
        self._frame_quat_w = torch.zeros(num_envs, F, 4, device=device)

        assert self._wp_device is not None

    @property
    def include_geom_groups(self) -> tuple[int, ...] | None:
        return self.cfg.include_geom_groups

    def set_context(self, ctx: SensorContext) -> None:
        """将传感器挂接到 SensorContext，用于 BVH 加速射线检测。"""
        self._ctx = ctx

    def _compute_data(self) -> RayCastData:
        if self._ctx is None:
            raise RuntimeError(
                "RayCastSensor requires a SensorContext. "
                "Ensure the sensor is part of a scene with "
                "sim.sense() calls."
            )
        assert self._distances is not None and self._normals_w is not None
        assert self._hit_pos_w is not None
        assert self._pos_w is not None and self._quat_w is not None
        assert self._frame_pos_w is not None and self._frame_quat_w is not None
        return RayCastData(
            distances=self._distances,
            normals_w=self._normals_w,
            hit_pos_w=self._hit_pos_w,
            pos_w=self._pos_w,
            quat_w=self._quat_w,
            frame_pos_w=self._frame_pos_w,
            frame_quat_w=self._frame_quat_w,
        )

    @property
    def num_rays(self) -> int:
        """总射线数 (num_frames * num_rays_per_frame)。"""
        return self._num_rays

    @property
    def num_frames(self) -> int:
        """挂载帧数量。"""
        return self._num_frames

    @property
    def num_rays_per_frame(self) -> int:
        """每帧射线数量。"""
        return self._num_rays_per_frame

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        if not self.cfg.debug_vis or not self._debug_vis_enabled:
            return
        assert self._data is not None
        assert self._local_offsets is not None
        assert self._local_directions is not None
        assert self._cached_frame_pos is not None
        assert self._cached_frame_mat is not None

        data = self.data
        env_indices = list(visualizer.get_env_indices(data.distances.shape[0]))
        if not env_indices:
            return

        F = self._num_frames
        N = self._num_rays_per_frame

        # 使用缓存的帧姿态 [B, F, 3] / [B, F, 3, 3]。
        frame_pos = self._cached_frame_pos[env_indices]  # [K, F, 3]
        frame_mat = self._cached_frame_mat[env_indices]  # [K, F, 3, 3]

        K_envs = len(env_indices)
        # 计算所有帧的对齐旋转。
        rot_mats = (
            self._compute_alignment_rotation(frame_mat.view(K_envs * F, 3, 3))
            .view(K_envs, F, 3, 3)
            .cpu()
            .numpy()
        )
        origins = frame_pos.cpu().numpy()
        offsets = self._local_offsets.cpu().numpy()
        directions = self._local_directions.cpu().numpy()
        hit_positions = data.hit_pos_w[env_indices].cpu().numpy()
        distances = data.distances[env_indices].cpu().numpy()
        normals = data.normals_w[env_indices].cpu().numpy()

        meansize = visualizer.meansize
        ray_width = 0.1 * meansize
        sphere_radius = self.cfg.viz.hit_sphere_radius * meansize
        normal_length = self.cfg.viz.normal_length * meansize
        normal_width = 0.1 * meansize
        miss_extent = min(0.5, self.cfg.max_distance * 0.05)
        name = self.cfg.name

        for k in range(K_envs):
            for f in range(F):
                rot = rot_mats[k, f]
                for i in range(N):
                    ray_idx = f * N + i
                    origin = origins[k, f] + rot @ offsets[i]
                    hit = distances[k, ray_idx] >= 0

                    if hit:
                        end = hit_positions[k, ray_idx]
                        color = self.cfg.viz.hit_color
                    else:
                        end = origin + rot @ directions[i] * miss_extent
                        color = self.cfg.viz.miss_color

                    if self.cfg.viz.show_rays:
                        visualizer.add_arrow(
                            start=origin,
                            end=end,
                            color=color,
                            width=ray_width,
                            label=f"{name}_ray_{ray_idx}",
                        )

                    if hit:
                        visualizer.add_sphere(
                            center=end,
                            radius=sphere_radius,
                            color=self.cfg.viz.hit_sphere_color,
                            label=f"{name}_hit_{ray_idx}",
                        )
                        if self.cfg.viz.show_normals:
                            normal_end = end + normals[k, ray_idx] * normal_length
                            visualizer.add_arrow(
                                start=end,
                                end=normal_end,
                                color=self.cfg.viz.normal_color,
                                width=normal_width,
                                label=f"{name}_normal_{ray_idx}",
                            )

    # 私有方法。

    def prepare_rays(self) -> None:
        """PRE-GRAPH：将局部射线变换到世界坐标系。

        通过 PyTorch 读取 body/site/geom 姿态，将世界坐标系中的射线
        原点和方向写入 Warp 数组。缓存帧姿态和世界坐标张量供
        postprocess_rays() 使用。
        """
        assert self._data is not None and self._model is not None
        assert self._local_offsets is not None and self._local_directions is not None

        # 收集逐帧姿态：[B, F, 3] 和 [B, F, 3, 3]。
        # 位置始终为物理世界位置。对齐仅影响射线方向
        # （在下文中应用于 frame_mat）。
        pos_list: list[torch.Tensor] = []
        mat_list: list[torch.Tensor] = []
        for frame_type, obj_id, _ in self._frame_infos:
            if frame_type == "body":
                pos_list.append(self._data.xpos[:, obj_id])
                mat_list.append(self._data.xmat[:, obj_id].view(-1, 3, 3))
            elif frame_type == "site":
                pos_list.append(self._data.site_xpos[:, obj_id])
                mat_list.append(self._data.site_xmat[:, obj_id].view(-1, 3, 3))
            else:  # geom
                pos_list.append(self._data.geom_xpos[:, obj_id])
                mat_list.append(self._data.geom_xmat[:, obj_id].view(-1, 3, 3))

        frame_pos = torch.stack(pos_list, dim=1)  # [B, F, 3]
        frame_mat = torch.stack(mat_list, dim=1)  # [B, F, 3, 3]

        B, F = frame_pos.shape[:2]
        N = self._num_rays_per_frame

        # 一次性计算所有帧的对齐旋转。
        rot_mat = self._compute_alignment_rotation(
            frame_mat.reshape(B * F, 3, 3)
        ).reshape(B, F, 3, 3)

        # 计算世界偏移和方向：[B, F, N, 3]。
        world_offsets = torch.einsum("bfij,nj->bfni", rot_mat, self._local_offsets)
        world_origins = frame_pos[:, :, None, :] + world_offsets
        world_rays = torch.einsum("bfij,nj->bfni", rot_mat, self._local_directions)

        # 展平为 [B, F*N, 3] 用于射线检测。
        world_origins_flat = world_origins.reshape(B, F * N, 3)
        world_rays_flat = world_rays.reshape(B, F * N, 3)

        assert self._ray_pnt is not None and self._ray_vec is not None
        pnt_torch = wp.to_torch(self._ray_pnt).view(B, self._num_rays, 3)
        vec_torch = wp.to_torch(self._ray_vec).view(B, self._num_rays, 3)
        pnt_torch.copy_(world_origins_flat)
        vec_torch.copy_(world_rays_flat)

        # 缓存供 postprocess_rays() 和 debug_vis() 使用。
        self._cached_world_origins = world_origins_flat
        self._cached_world_rays = world_rays_flat
        self._cached_frame_pos = frame_pos  # [B, F, 3]
        self._cached_frame_mat = frame_mat  # [B, F, 3, 3]

    def raycast_kernel(self, rc: mjwarp.RenderContext) -> None:
        """IN-GRAPH：执行 BVH 加速的射线检测内核。"""
        assert self._ray_pnt is not None
        assert self._ray_vec is not None
        assert self._ray_bodyexclude is not None
        assert self._ray_dist is not None
        assert self._ray_geomid is not None
        assert self._ray_normal is not None
        rays(
            m=self._model.struct,  # type: ignore[attr-defined]
            d=self._data.struct,  # type: ignore[attr-defined]
            pnt=self._ray_pnt,  # pyright: ignore[reportArgumentType]
            vec=self._ray_vec,  # pyright: ignore[reportArgumentType]
            geomgroup=self._geomgroup,  # pyright: ignore[reportArgumentType]
            flg_static=True,
            bodyexclude=self._ray_bodyexclude,
            dist=self._ray_dist,  # pyright: ignore[reportArgumentType]
            geomid=self._ray_geomid,  # pyright: ignore[reportArgumentType]
            normal=self._ray_normal,  # pyright: ignore[reportArgumentType]
            rc=rc,
        )

    def postprocess_rays(self) -> None:
        """POST-GRAPH：将 Warp 输出转换为 PyTorch 张量，计算命中位置。"""
        assert self._cached_world_origins is not None
        assert self._cached_world_rays is not None
        assert self._cached_frame_pos is not None
        assert self._cached_frame_mat is not None

        B = self._cached_frame_pos.shape[0]
        F = self._num_frames

        assert self._ray_dist is not None and self._ray_normal is not None
        distances = wp.to_torch(self._ray_dist)
        normals_w = wp.to_torch(self._ray_normal).view(B, self._num_rays, 3)
        distances.masked_fill_(distances > self.cfg.max_distance, -1.0)

        hit_mask = distances >= 0
        # ``origin + ray * max(distance, 0)`` 将未命中射线收缩到 ``origin``
        # （clamp 后的距离为 0），无需分支。
        clamped = distances.clamp(min=0.0)
        self._hit_pos_w = (
            self._cached_world_origins + self._cached_world_rays * clamped.unsqueeze(-1)
        )

        # 未命中时将法线置零。
        normals_w.masked_fill_(~hit_mask.unsqueeze(-1), 0.0)
        self._distances = distances
        self._normals_w = normals_w

        # 所有帧：[B, F, 3] / [B, F, 4]。
        self._frame_pos_w = self._cached_frame_pos
        self._frame_quat_w = quat_from_matrix(
            self._cached_frame_mat.reshape(B * F, 3, 3)
        ).reshape(B, F, 4)

        # 第一帧供向后兼容：[B, 3] / [B, 4]。
        assert self._frame_pos_w is not None and self._frame_quat_w is not None
        self._pos_w = self._frame_pos_w[:, 0]
        self._quat_w = self._frame_quat_w[:, 0]

    def _compute_alignment_rotation(self, frame_mat: torch.Tensor) -> torch.Tensor:
        """根据 ray_alignment 设置计算旋转矩阵。"""
        if self.cfg.ray_alignment == "base":
            # 完整旋转。
            return frame_mat
        elif self.cfg.ray_alignment == "yaw":
            # 仅提取偏航角，去除俯仰/滚动。
            return self._extract_yaw_rotation(frame_mat)
        elif self.cfg.ray_alignment == "world":
            # 单位旋转（世界对齐）。
            num_envs = frame_mat.shape[0]
            return (
                torch.eye(3, device=frame_mat.device, dtype=frame_mat.dtype)
                .unsqueeze(0)
                .expand(num_envs, -1, -1)
            )
        else:
            raise ValueError(f"Unknown ray_alignment: {self.cfg.ray_alignment}")

    def _extract_yaw_rotation(self, rot_mat: torch.Tensor) -> torch.Tensor:
        """提取仅偏航旋转矩阵（绕 Z 轴旋转）。

        处理 ±90° 俯仰处的奇异性：当 X 轴在 XY 平面上的投影
        过小时，回退使用 Y 轴。
        """
        batch_size = rot_mat.shape[0]
        device = rot_mat.device
        dtype = rot_mat.dtype

        # 将 X 轴投影到 XY 平面。
        x_axis = rot_mat[:, :, 0]  # 第一列 [B, 3]
        x_proj = x_axis.clone()
        x_proj[:, 2] = 0  # 将 Z 分量置零
        x_norm = x_proj.norm(dim=1)  # [B]

        # 检查奇异性（X 轴接近垂直）。
        threshold = 0.1
        singular = x_norm < threshold  # [B]

        # 对于奇异情况，改用 Y 轴。
        if singular.any():
            y_axis = rot_mat[:, :, 1]  # 第二列 [B, 3]
            y_proj = y_axis.clone()
            y_proj[:, 2] = 0
            y_norm = y_proj.norm(dim=1).clamp(min=1e-6)
            y_proj = y_proj / y_norm.unsqueeze(-1)
            # Y 轴指向左侧；绕 Z 轴旋转 -90° 得到前向。
            # [y_x, y_y] -> [y_y, -y_x]
            x_from_y = torch.zeros_like(y_proj)
            x_from_y[:, 0] = y_proj[:, 1]
            x_from_y[:, 1] = -y_proj[:, 0]
            x_proj[singular] = x_from_y[singular]
            x_norm[singular] = 1.0  # 已经归一化

        # 归一化 X 投影。
        x_norm = x_norm.clamp(min=1e-6)
        x_proj = x_proj / x_norm.unsqueeze(-1)

        # 构建仅偏航旋转矩阵。
        yaw_mat = torch.zeros((batch_size, 3, 3), device=device, dtype=dtype)
        yaw_mat[:, 0, 0] = x_proj[:, 0]
        yaw_mat[:, 1, 0] = x_proj[:, 1]
        yaw_mat[:, 0, 1] = -x_proj[:, 1]
        yaw_mat[:, 1, 1] = x_proj[:, 0]
        yaw_mat[:, 2, 2] = 1
        return yaw_mat
