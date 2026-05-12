from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg
from mjlab.utils import spec_config as spec_cfg


def _proportional_counts(num_envs: int, proportions: np.ndarray) -> np.ndarray:
    """按比例将 *num_envs* 分配到各个桶中。

    当 ``num_envs >= len(proportions)`` 时，每个桶至少分配一个。剩余名额
    通过最大余数法（Largest Remainder Method）进行分配。
    """
    n = len(proportions)
    if num_envs >= n:
        counts = np.ones(n, dtype=int)
        remaining = num_envs - n
    else:
        counts = np.zeros(n, dtype=int)
        remaining = num_envs
    if remaining > 0:
        ideal = proportions * remaining
        floor = np.floor(ideal).astype(int)
        counts += floor
        leftover = remaining - floor.sum()
        if leftover > 0:
            order = np.argsort(-(ideal - floor))
            counts[order[:leftover]] += 1
    return counts


_DEFAULT_SUN_LIGHT = spec_cfg.LightCfg(
    name="sun", pos=(0.0, 0.0, 1.5), type="directional"
)

_DEFAULT_PLANE_TEXTURE = spec_cfg.TextureCfg(
    name="groundplane",
    type="2d",
    builtin="checker",
    mark="edge",
    rgb1=(0.2, 0.3, 0.4),
    rgb2=(0.1, 0.2, 0.3),
    markrgb=(0.8, 0.8, 0.8),
    width=300,
    height=300,
)

_DEFAULT_PLANE_MATERIAL = spec_cfg.MaterialCfg(
    name="groundplane",
    texuniform=True,
    texrepeat=(4.0, 4.0),
    reflectance=0.2,
    texture="groundplane",
    geom_names_expr=("terrain$",),
)


@dataclass
class TerrainEntityCfg(EntityCfg):
    """地形实体配置。"""

    terrain_type: Literal["generator", "plane"] = "plane"
    """要生成的地形类型。"generator" 使用带子地形网格的程序化地形，
    "plane" 创建一个平坦的地面平面。"""
    terrain_generator: TerrainGeneratorCfg | None = None
    """程序化地形生成的配置。当 terrain_type 为 "generator" 时必须提供。"""
    env_spacing: float | None = 2.0
    """使用网格布局时各环境原点之间的间距。对于 "plane" 地形或没有
    子地形原点时必须指定。"""
    max_init_terrain_level: int | None = None
    """课程模式下环境放置的最大初始难度等级（行索引）。
    为 None 时使用所有可用行。"""
    num_envs: int = 1
    """要创建的并行环境数量。如果在场景配置中指定，则会被覆盖。"""
    textures: tuple[spec_cfg.TextureCfg, ...] = field(
        default_factory=lambda: (_DEFAULT_PLANE_TEXTURE,)
    )
    """地面平面的纹理。默认为棋盘格图案。设为 ``()`` 可禁用纹理
    （例如使用 ``dr.geom_rgba`` 时）。"""
    materials: tuple[spec_cfg.MaterialCfg, ...] = field(
        default_factory=lambda: (_DEFAULT_PLANE_MATERIAL,)
    )
    """地面平面的材质。默认为棋盘格材质。设为 ``()`` 可禁用材质
    （例如使用 ``dr.geom_rgba`` 时）。"""
    lights: tuple[spec_cfg.LightCfg, ...] = field(
        default_factory=lambda: (_DEFAULT_SUN_LIGHT,)
    )
    """场景的光源。默认为一个方向性太阳光。"""
    debug_vis: bool = False
    """为环境原点、地形原点和平坦地块添加可视化站点。
    默认为 False。"""

    def build(self) -> TerrainEntity:
        raise TypeError(
            "TerrainEntityCfg.build() requires a device argument. "
            "Use TerrainEntity(cfg, device=...) directly."
        )


class TerrainEntity(Entity):
    """地形实体。

    地形是一个子地形地块网格（num_rows x num_cols），每个地块都有一个
    生成原点。当 num_envs 超过地块数量时，环境原点将从子地形原点中采样。

    .. note::
      程序化地形的环境分配：当 ``TerrainGeneratorCfg`` 可用时，列（地形类型）
      会按**比例**（匹配每个子地形的 ``proportion`` 字段）分配到各环境；
      不可用时则均匀分配。行（难度等级）会随机采样。这意味着即使
      num_envs > num_patches 时，多个环境也可能生成在同一个 (row, col)
      地块上，而其他地块可能未被占用。

    参见 FAQ："env_origins 如何决定机器人布局？"
    """

    cfg: TerrainEntityCfg

    def __init__(self, cfg: TerrainEntityCfg, device: str) -> None:
        self._device = device
        super().__init__(cfg)

    def _build_spec(self) -> None:
        self._spec = mujoco.MjSpec()

        if self.cfg.terrain_type == "generator":
            if self.cfg.terrain_generator is None:
                raise ValueError(
                    "terrain_generator must be specified for terrain_type 'generator'"
                )
            terrain_generator = TerrainGenerator(
                self.cfg.terrain_generator, device=self._device
            )
            terrain_generator.compile(self._spec)
            gen_cfg = self.cfg.terrain_generator
            proportions = np.array(
                [s.proportion for s in gen_cfg.sub_terrains.values()]
            )
            proportions = proportions / proportions.sum()
            self._configure_env_origins(terrain_generator.terrain_origins, proportions)
            self._flat_patches: dict[str, torch.Tensor] = {
                name: torch.from_numpy(arr).to(device=self._device, dtype=torch.float)
                for name, arr in terrain_generator.flat_patches.items()
            }
            self._flat_patch_radii: dict[str, float] = dict(
                terrain_generator.flat_patch_radii
            )
        elif self.cfg.terrain_type == "plane":
            self._import_ground_plane("terrain")
            self._configure_env_origins()
            self._flat_patches: dict[str, torch.Tensor] = {}
            self._flat_patch_radii: dict[str, float] = {}
        else:
            raise ValueError(f"Unknown terrain type: {self.cfg.terrain_type}")

        if self.cfg.debug_vis:
            self._add_env_origin_sites()
            self._add_terrain_origin_sites()
            self._add_flat_patch_sites()

    def _add_initial_state_keyframe(self) -> None:
        pass  # 无关节，无关键帧。

    # 地形专用属性。

    @property
    def flat_patches(self) -> dict[str, torch.Tensor]:
        return self._flat_patches

    @property
    def flat_patch_radii(self) -> dict[str, float]:
        return self._flat_patch_radii

    # 地形原点管理。

    def configure_env_origins(
        self,
        origins: np.ndarray | torch.Tensor | None = None,
        proportions: np.ndarray | None = None,
    ) -> None:
        """根据地形配置环境的原点。"""
        self._configure_env_origins(origins, proportions)

    def _configure_env_origins(
        self,
        origins: np.ndarray | torch.Tensor | None = None,
        proportions: np.ndarray | None = None,
    ) -> None:
        if origins is not None:
            if isinstance(origins, np.ndarray):
                origins = torch.from_numpy(origins)
            else:
                assert isinstance(origins, torch.Tensor)
            self.terrain_origins = origins.to(self._device, dtype=torch.float)
            self.env_origins = self._compute_env_origins_curriculum(
                self.cfg.num_envs, self.terrain_origins, proportions
            )
        else:
            self.terrain_origins = None
            if self.cfg.env_spacing is None:
                raise ValueError(
                    "Environment spacing must be specified for configuring grid-like origins."
                )
            self.env_origins = self._compute_env_origins_grid(
                self.cfg.num_envs, self.cfg.env_spacing
            )

    def update_env_origins(
        self,
        env_ids: torch.Tensor,
        move_up: torch.Tensor,
        move_down: torch.Tensor,
    ) -> None:
        """根据地形等级更新环境原点。"""
        if self.terrain_origins is None:
            return
        assert self.env_origins is not None
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    def randomize_env_origins(self, env_ids: torch.Tensor) -> None:
        """将环境原点随机化到随机的子地形上。"""
        if self.terrain_origins is None:
            return
        assert self.env_origins is not None
        num_rows, num_cols = self.terrain_origins.shape[:2]
        num_envs = len(env_ids)
        self.terrain_levels[env_ids] = torch.randint(
            0, num_rows, (num_envs,), device=self._device
        )
        self.terrain_types[env_ids] = torch.randint(
            0, num_cols, (num_envs,), device=self._device
        )
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    # 私有方法。

    def _import_ground_plane(self, name: str) -> None:
        self._spec.worldbody.add_body(name=name).add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(0, 0, 0.01),
        )

    def _add_env_origin_sites(self) -> None:
        if self.env_origins is None:
            return
        origin_site_radius: float = 0.3
        origin_site_color = (0.2, 0.6, 0.2, 0.3)
        if isinstance(self.env_origins, torch.Tensor):
            env_origins_np = self.env_origins.cpu().numpy()
        else:
            env_origins_np = self.env_origins
        for env_id, origin in enumerate(env_origins_np):
            self._spec.worldbody.add_site(
                name=f"env_origin_{env_id}",
                pos=origin,
                size=(origin_site_radius,) * 3,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                rgba=origin_site_color,
                group=4,
            )

    def _add_terrain_origin_sites(self) -> None:
        if self.terrain_origins is None:
            return
        if isinstance(self.terrain_origins, torch.Tensor):
            terrain_origins_np = self.terrain_origins.cpu().numpy()
        else:
            terrain_origins_np = self.terrain_origins
        terrain_origin_site_radius: float = 0.5
        terrain_origin_site_color = (0.2, 0.2, 0.6, 0.3)
        num_rows, num_cols = terrain_origins_np.shape[:2]
        for row in range(num_rows):
            for col in range(num_cols):
                origin = terrain_origins_np[row, col]
                self._spec.worldbody.add_site(
                    name=f"terrain_origin_{row}_{col}",
                    pos=origin,
                    size=(terrain_origin_site_radius,) * 3,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    rgba=terrain_origin_site_color,
                    group=5,
                )

    def _add_flat_patch_sites(self) -> None:
        if not self._flat_patches:
            return
        site_thickness = 0.02
        site_color = (0.9, 0.6, 0.1, 0.1)
        for name, patches_tensor in self._flat_patches.items():
            radius = self._flat_patch_radii.get(name, 0.5)
            patches_np = patches_tensor.cpu().numpy()
            num_rows, num_cols, num_patches, _ = patches_np.shape
            for row in range(num_rows):
                for col in range(num_cols):
                    for p in range(num_patches):
                        pos = patches_np[row, col, p]
                        self._spec.worldbody.add_site(
                            name=f"flat_patch_{name}_{row}_{col}_{p}",
                            pos=pos,
                            size=(radius, radius, site_thickness),
                            type=mujoco.mjtGeom.mjGEOM_BOX,
                            rgba=site_color,
                            group=3,
                        )

    def _compute_env_origins_curriculum(
        self,
        num_envs: int,
        origins: torch.Tensor,
        proportions: np.ndarray | None = None,
    ) -> torch.Tensor:
        """从子地形原点计算环境原点。

        Args:
          num_envs: 要放置的环境数量。
          origins: 子地形原点，形状 ``[num_rows, num_cols, 3]``。
          proportions: 归一化的每列权重。提供时，机器人按比例分配
            （当 ``num_envs >= num_cols`` 时每列至少一个）。
            ``None`` 表示均匀分配。
        """
        num_rows, num_cols = origins.shape[:2]
        if self.cfg.max_init_terrain_level is None:
            max_init_level = num_rows - 1
        else:
            max_init_level = min(self.cfg.max_init_terrain_level, num_rows - 1)
        self.max_terrain_level = num_rows
        self.terrain_levels = torch.randint(
            0, max_init_level + 1, (num_envs,), device=self._device
        )

        if proportions is not None and len(proportions) == num_cols:
            counts = _proportional_counts(num_envs, proportions)
            self.terrain_types = torch.repeat_interleave(
                torch.arange(num_cols, device=self._device),
                torch.from_numpy(counts).to(self._device),
            )
        else:
            self.terrain_types = torch.div(
                torch.arange(num_envs, device=self._device),
                (num_envs / num_cols),
                rounding_mode="floor",
            ).to(torch.long)

        env_origins = torch.zeros(num_envs, 3, device=self._device)
        env_origins[:] = origins[self.terrain_levels, self.terrain_types]
        return env_origins

    def _compute_env_origins_grid(
        self, num_envs: int, env_spacing: float
    ) -> torch.Tensor:
        env_origins = torch.zeros(num_envs, 3, device=self._device)
        num_rows = np.ceil(num_envs / int(np.sqrt(num_envs)))
        num_cols = np.ceil(num_envs / num_rows)
        ii, jj = torch.meshgrid(
            torch.arange(num_rows, device=self._device),
            torch.arange(num_cols, device=self._device),
            indexing="ij",
        )
        env_origins[:, 0] = (
            -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * env_spacing
        )
        env_origins[:, 1] = (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * env_spacing
        env_origins[:, 2] = 0.0
        return env_origins
