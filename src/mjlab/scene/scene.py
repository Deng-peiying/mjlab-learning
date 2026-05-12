from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.entity.variants import VariantMetadata
from mjlab.sensor import BuiltinSensor, RayCastSensor, Sensor, SensorCfg
from mjlab.sensor.camera_sensor import CameraSensor
from mjlab.sensor.sensor_context import SensorContext
from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg
from mjlab.utils.spec import export_spec, non_default_option_fields

_SCENE_XML = Path(__file__).parent / "scene.xml"


@dataclass(kw_only=True)
class SceneCfg:
    """仿真场景的配置。"""

    num_envs: int = 1
    """并行环境数量。"""

    env_spacing: float = 2.0
    """各环境原点之间的间距（米）。"""

    terrain: TerrainEntityCfg | None = None
    """地形配置。若为 ``None``，则不添加地形。"""

    entities: dict[str, EntityCfg] = field(default_factory=dict)
    """实体名称到其配置的映射。"""

    sensors: tuple[SensorCfg, ...] = field(default_factory=tuple)
    """附加到场景的传感器配置。"""

    extent: float | None = None
    """覆盖 ``mjModel.stat.extent``。若为 ``None``，则由 MuJoCo 自动计算。"""

    spec_fn: Callable[[mujoco.MjSpec], None] | None = None
    """可选回调，在实体和传感器已添加但编译之前修改 ``MjSpec``。"""


class Scene:
    def __init__(self, scene_cfg: SceneCfg, device: str) -> None:
        self._cfg = scene_cfg
        self._device = device
        self._entities: dict[str, Entity] = {}
        self._sensors: dict[str, Sensor] = {}
        self._terrain: TerrainEntity | None = None
        self._default_env_origins: torch.Tensor | None = None
        self._sensor_context: SensorContext | None = None

        self._spec = mujoco.MjSpec.from_file(str(_SCENE_XML))
        if self._cfg.extent is not None:
            self._spec.stat.extent = self._cfg.extent
        self._add_terrain()
        self._add_entities()
        self._add_sensors()
        if self._cfg.spec_fn is not None:
            self._cfg.spec_fn(self._spec)

    def compile(self) -> mujoco.MjModel:
        return self._spec.compile()

    def write(self, output_dir: Path, *, zip: bool = False) -> None:
        """将场景 XML 和网格资产写入目录。

        创建 ``scene.xml`` 和包含场景引用的网格文件的 ``assets/`` 子目录。
        当 *zip* 为 True 时，目录会被压缩为 ``.zip`` 归档文件并删除原目录。
        在 spec 的副本上操作以避免对原数据进行修改。

        Args:
          output_dir: 目标目录（若不存在则创建）。
          zip: 若为 True，则生成 ``<output_dir>.zip`` 而非目录。
        """
        export_spec(self._spec, output_dir, zip=zip)

    def to_zip(self, path: Path) -> None:
        """已弃用。请使用 ``write(output_dir, zip=True)`` 替代。"""
        warnings.warn(
            "Scene.to_zip() is deprecated. Use Scene.write(path, zip=True).",
            DeprecationWarning,
            stacklevel=2,
        )
        self.write(path, zip=True)

    # 属性。

    @property
    def spec(self) -> mujoco.MjSpec:
        return self._spec

    @property
    def env_origins(self) -> torch.Tensor:
        if self._terrain is not None:
            assert self._terrain.env_origins is not None
            return self._terrain.env_origins
        assert self._default_env_origins is not None
        return self._default_env_origins

    @property
    def env_spacing(self) -> float:
        return self._cfg.env_spacing

    @property
    def entities(self) -> dict[str, Entity]:
        return self._entities

    @property
    def sensors(self) -> dict[str, Sensor]:
        return self._sensors

    @property
    def terrain(self) -> TerrainEntity | None:
        return self._terrain

    @property
    def num_envs(self) -> int:
        return self._cfg.num_envs

    @property
    def device(self) -> str:
        return self._device

    def collect_variant_info(
        self,
    ) -> list[tuple[str, VariantMetadata]]:
        """收集具有网格变体的实体的变体元数据。"""
        result: list[tuple[str, VariantMetadata]] = []
        for name, ent in self._entities.items():
            if ent.variant_metadata is not None:
                result.append((f"{name}/", ent.variant_metadata))
        return result

    def __getitem__(self, key: str) -> Any:
        if key in self._sensors:
            return self._sensors[key]
        if key in self._entities:
            return self._entities[key]

        # 未找到，抛出有帮助的错误信息。
        available = list(self._entities.keys()) + list(self._sensors.keys())
        raise KeyError(f"Scene element '{key}' not found. Available: {available}")

    # 方法。

    @property
    def sensor_context(self) -> SensorContext | None:
        """共享的感知资源，若没有相机/射线传感器则为 None。"""
        return self._sensor_context

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
    ):
        self._default_env_origins = torch.zeros(
            (self._cfg.num_envs, 3), device=self._device, dtype=torch.float32
        )
        for ent in self._entities.values():
            ent.initialize(mj_model, model, data, self._device)
        for sensor in self._sensors.values():
            sensor.initialize(mj_model, model, data, self._device)

        # 如果有传感器需要，则创建 SensorContext。
        ctx_sensors = [s for s in self._sensors.values() if s.requires_sensor_context]
        if ctx_sensors:
            camera_sensors = [s for s in ctx_sensors if isinstance(s, CameraSensor)]
            raycast_sensors = [s for s in ctx_sensors if isinstance(s, RayCastSensor)]
            self._sensor_context = SensorContext(
                mj_model=mj_model,
                model=model,
                data=data,
                camera_sensors=camera_sensors,
                raycast_sensors=raycast_sensors,
                device=self._device,
            )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        for ent in self._entities.values():
            ent.reset(env_ids)
        for sensor in self._sensors.values():
            sensor.reset(env_ids)

    def update(self, dt: float) -> None:
        for ent in self._entities.values():
            ent.update(dt)
        for sensor in self._sensors.values():
            sensor.update(dt)

    def write_data_to_sim(self) -> None:
        for ent in self._entities.values():
            ent.write_data_to_sim()

    # 私有方法。

    def _add_entities(self) -> None:
        # 从各实体收集关键帧，合并为单个场景关键帧。
        # 顺序很重要：qpos/ctrl 按实体迭代顺序拼接。
        key_qpos: list[np.ndarray] = []
        key_ctrl: list[np.ndarray] = []
        for ent_name, ent_cfg in self._cfg.entities.items():
            ent = ent_cfg.build()
            self._entities[ent_name] = ent
            # 在 attach 之前提取关键帧（必须在 attach 之前删除以避免损坏）。
            if ent.spec.keys:
                if len(ent.spec.keys) > 1:
                    warnings.warn(
                        f"Entity '{ent_name}' has {len(ent.spec.keys)} keyframes; only the "
                        "first one will be used.",
                        stacklevel=2,
                    )
                key_qpos.append(np.array(ent.spec.keys[0].qpos))
                key_ctrl.append(np.array(ent.spec.keys[0].ctrl))
                ent.spec.delete(ent.spec.keys[0])
            non_default = non_default_option_fields(ent.spec.option)
            if non_default:
                fields = ", ".join(non_default)
                warnings.warn(
                    f"Entity '{ent_name}' has non-default <option> fields ({fields}) that will"
                    " not be propagated by MjSpec.attach(). Use MujocoCfg instead.",
                    stacklevel=2,
                )
            frame = self._spec.worldbody.add_frame()
            self._spec.attach(ent.spec, prefix=f"{ent_name}/", frame=frame)
        # 将合并后的关键帧添加到场景 spec 中。
        if key_qpos:
            combined_qpos = np.concatenate(key_qpos)
            combined_ctrl = np.concatenate(key_ctrl)
            self._spec.add_key(
                name="init_state",
                qpos=combined_qpos.tolist(),
                ctrl=combined_ctrl.tolist(),
            )

    def _add_terrain(self) -> None:
        if self._cfg.terrain is None:
            return
        self._cfg.terrain.num_envs = self._cfg.num_envs
        self._cfg.terrain.env_spacing = self._cfg.env_spacing
        terrain = TerrainEntity(self._cfg.terrain, device=self._device)
        self._terrain = terrain
        self._entities["terrain"] = terrain
        non_default = non_default_option_fields(terrain.spec.option)
        if non_default:
            fields = ", ".join(non_default)
            warnings.warn(
                f"Terrain has non-default <option> fields ({fields}) that will not be"
                " propagated by MjSpec.attach(). Use MujocoCfg instead.",
                stacklevel=2,
            )
        frame = self._spec.worldbody.add_frame()
        self._spec.attach(terrain.spec, prefix="", frame=frame)

    def _add_sensors(self) -> None:
        for sensor_cfg in self._cfg.sensors:
            sns = sensor_cfg.build()
            sns.edit_spec(self._spec, self._entities)
            self._sensors[sensor_cfg.prefixed_name] = sns

        for sns in self._spec.sensors:
            if sns.name not in self._sensors:
                self._sensors[sns.name] = BuiltinSensor.from_existing(sns.name)
