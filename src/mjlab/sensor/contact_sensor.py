"""接触传感器用于跟踪几何体、物体或子树之间的碰撞。

``ContactSensor`` 将正则表达式模式解析为一组 *主要* MuJoCo 元素，以及
（可选）一个用于过滤的 *次要* 元素。每个物理步中，它从 MuJoCo 的原生接触
传感器中提取每个主要元素的接触数据，并打包为批处理的 ``ContactData`` 数据类。

``ContactData`` 上的形状约定：

- ``P`` = 从模式解析出的主要元素数量（参见
  :attr:`ContactSensor.primary_names` 获取索引→名称映射）。
- ``N`` = ``P * num_slots``（逐接触轴，以主要元素为主布局）。
- 逐接触字段（``found``、``force``、``torque``、``dist``、``pos``、
  ``normal``、``tangent``）的形状为 ``[B, N, ...]``。
- 逐主要元素字段（``current_air_time`` 等）的形状为 ``[B, P]``。

大多数用户使用默认的 ``num_slots=1``，此时 ``N == P``，两种形状族一致。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity
from mjlab.sensor.sensor import Sensor, SensorCfg

_CONTACT_DATA_MAP = {
    "found": 0,
    "force": 1,
    "torque": 2,
    "dist": 3,
    "pos": 4,
    "normal": 5,
    "tangent": 6,
}

_CONTACT_DATA_DIMS = {
    "found": 1,
    "force": 3,
    "torque": 3,
    "dist": 1,
    "pos": 3,
    "normal": 3,
    "tangent": 3,
}

_CONTACT_REDUCE_MAP = {
    "none": 0,
    "mindist": 1,
    "maxforce": 2,
    "netforce": 3,
}

_MODE_TO_OBJTYPE = {
    "geom": mujoco.mjtObj.mjOBJ_GEOM,
    "body": mujoco.mjtObj.mjOBJ_BODY,
    "subtree": mujoco.mjtObj.mjOBJ_XBODY,
}


@dataclass
class ContactMatch:
    """指定接触一侧的匹配条件。

    Args:
      mode: 要匹配的 MuJoCo 元素类型（"geom"、"body" 或 "subtree"）。
      pattern: 一个正则表达式（或正则表达式元组），与 ``entity`` 中的元素名称
        进行匹配。如果未设置 ``entity``，该模式将被视为 MuJoCo 字面名称
        （不进行正则展开）。
      entity: 用于限定模式范围的实体名称。如果为 ``None``/``""``，
        则该模式将被视为 MuJoCo 字面名称。
      exclude: 要从匹配中过滤掉的名称。如果条目包含正则元字符，
        则将其视为正则表达式，否则视为精确名称。
    """

    mode: Literal["geom", "body", "subtree"]
    pattern: str | tuple[str, ...]
    entity: str | None = None
    exclude: tuple[str, ...] = ()


@dataclass
class ContactSensorCfg(SensorCfg):
    """用于 :class:`ContactSensor` 的配置。

    接触传感器监视 ``primary`` 元素集与可选的 ``secondary`` 元素之间的接触。
    模式可展开为多个主要元素（例如四足机器人的所有四只脚）；每个主要元素
    成为输出张量逐接触轴的一列。

    参见模块 docstring 了解形状约定（``P``、``N``、主要元素为主布局），
    以及 :attr:`ContactSensor.primary_names` 获取索引→名称查找。

    Args:
      primary: 要测量的元素（例如机器人的脚）。通常是一个可解析为多个
        元素的正则表达式。
      secondary: 对主要元素可能接触的内容的可选过滤（例如地形）。
        ``None`` 表示"与主要元素的任何接触都计入"。
      fields: 要提取的接触量。仅请求的字段会被分配和计算；
        其余字段在 ``ContactData`` 上为 ``None``。

        - ``"found"``: 0 = 无接触；>0 = 匹配到的接触数量。
        - ``"force"``, ``"torque"``: 接触坐标系中的 3D 向量（当
          ``reduce="netforce"`` 或 ``global_frame=True`` 时为全局坐标系）。
        - ``"dist"``: 穿透深度（标量）。
        - ``"pos"``, ``"normal"``, ``"tangent"``: 全局坐标系中的 3D 向量
          （法线指向主要元素 → 次要元素）。

      reduce: 如何将同一主要元素上的同时接触折叠为 ``num_slots`` 个
        代表性接触。

        - ``"none"``: 快速，非确定性排序。
        - ``"mindist"``: 保留最近（最深）的接触。
        - ``"maxforce"``: 保留最强的接触。
        - ``"netforce"``: 将所有接触求和为单个净力/力矩（无论
          ``num_slots`` 如何，每个主要元素始终只产生一个槽位）。

      num_slots: 归约后每个主要元素保留的接触数量。
        几乎总是 ``1``：大多数策略希望每个主要元素有一个代表性接触，
        而模式展开已经提供了多个主要元素。仅当单个主要元素可能有
        多个需要分别检查的不同接触时才增大此值，配合 ``reduce``
        为 ``{"none", "mindist", "maxforce"}``。
        ``reduce="netforce"`` 时忽略此参数。
      secondary_policy: 如何处理解析为多个元素的次要模式：
        ``"first"`` 选取第一个匹配，``"any"`` 完全移除次要过滤，
        ``"error"`` 抛出错误。
      track_air_time: 分配逐主要元素的空中/接触时间累加器
        （对步态奖励有用）。需要 ``fields`` 中包含 ``"found"``。
        主要元素内的槽位归约使用"任何槽位接触即视为接触"规则。
      global_frame: 将 ``force``/``torque`` 从接触坐标系旋转到全局坐标系。
        需要 ``fields`` 中包含 ``"normal"`` 和 ``"tangent"``。
        ``reduce="netforce"`` 时隐式启用。
      history_length: 如果 >0，保留最近 N 个子步的
        ``force``/``torque``/``dist`` 数据的滚动缓冲区。
        设置为你的 decimation 值，使缓冲区恰好覆盖一个策略步；
        对于捕获在子步中途解决的短暂碰撞很有用。``0`` 禁用缓冲区。
      debug: 在将每个 MuJoCo 传感器添加到 spec 时打印。
        有助于检查模式展开是否产生了预期的元素。
    """

    primary: ContactMatch
    secondary: ContactMatch | None = None
    fields: tuple[str, ...] = ("found", "force")
    reduce: Literal["none", "mindist", "maxforce", "netforce"] = "maxforce"
    num_slots: int = 1
    secondary_policy: Literal["first", "any", "error"] = "first"
    track_air_time: bool = False
    global_frame: bool = False
    history_length: int = 0
    debug: bool = False

    def build(self) -> ContactSensor:
        return ContactSensor(self)


@dataclass
class _ContactSlot:
    """将一个 MuJoCo 传感器（一个主要元素，一个字段）映射到其 sensordata 视图。"""

    primary_name: str
    field_name: str
    sensor_name: str
    data_view: torch.Tensor | None = None


@dataclass
class _AirTimeState:
    """跟踪接触处于空中/接触状态的时长。形状: [B, P]。"""

    current_air_time: torch.Tensor
    last_air_time: torch.Tensor
    current_contact_time: torch.Tensor
    last_contact_time: torch.Tensor
    last_time: torch.Tensor


@dataclass
class ContactData:
    """接触传感器输出（仅填充请求的字段）。

    形状约定：P = 主要元素数量；N = P * num_slots（逐接触字段
    以主要元素为主布局）。空中时间字段是逐主要元素的，
    并在槽位间归约（任何槽位接触 → 主要元素处于接触状态）。
    """

    found: torch.Tensor | None = None
    """[B, N] 0=无接触, >0=匹配计数"""
    force: torch.Tensor | None = None
    """[B, N, 3] 接触坐标系（reduce="netforce" 或 global_frame=True 时为全局）"""
    torque: torch.Tensor | None = None
    """[B, N, 3] 接触坐标系（reduce="netforce" 或 global_frame=True 时为全局）"""
    dist: torch.Tensor | None = None
    """[B, N] 穿透深度"""
    pos: torch.Tensor | None = None
    """[B, N, 3] 全局坐标系"""
    normal: torch.Tensor | None = None
    """[B, N, 3] 全局坐标系，主要→次要"""
    tangent: torch.Tensor | None = None
    """[B, N, 3] 全局坐标系"""

    current_air_time: torch.Tensor | None = None
    """[B, P] 每个主要元素的空中时间（如果 track_air_time=True）"""
    last_air_time: torch.Tensor | None = None
    """[B, P] 每个主要元素的上一次空中阶段持续时间（如果 track_air_time=True）"""
    current_contact_time: torch.Tensor | None = None
    """[B, P] 每个主要元素的接触时间（如果 track_air_time=True）"""
    last_contact_time: torch.Tensor | None = None
    """[B, P] 每个主要元素的上一次接触阶段持续时间（如果 track_air_time=True）"""

    force_history: torch.Tensor | None = None
    """[B, N, H, 3] 最近 H 个子步的接触力（索引 0 = 最新）"""
    torque_history: torch.Tensor | None = None
    """[B, N, H, 3] 最近 H 个子步的接触力矩（索引 0 = 最新）"""
    dist_history: torch.Tensor | None = None
    """[B, N, H] 最近 H 个子步的穿透深度（索引 0 = 最新）"""


class ContactSensor(Sensor[ContactData]):
    """跟踪接触，自动将模式展开为多个 MuJoCo 传感器。"""

    def __init__(self, cfg: ContactSensorCfg) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.global_frame and cfg.reduce != "netforce":
            if "normal" not in cfg.fields or "tangent" not in cfg.fields:
                raise ValueError(
                    f"Sensor '{cfg.name}': global_frame=True requires 'normal' and 'tangent' "
                    "in fields (needed to build rotation matrix)"
                )

        self._slots: list[_ContactSlot] = []
        self._data: mjwarp.Data | None = None
        self._device: str | None = None
        self._air_time_state: _AirTimeState | None = None
        self._history_state: dict[str, torch.Tensor] | None = None

    @property
    def primary_names(self) -> list[str]:
        """主要元素名称，按沿逐接触维度的顺序排列。

        逐接触字段（[B, N, ...]）以主要元素为主布局，因此主要元素
        `primary_names[i]` 占据索引 [i * num_slots : (i + 1) * num_slots]。
        逐主要元素字段（[B, P, ...]）在此列表中每个名称对应一个条目。
        """
        return list(dict.fromkeys(slot.primary_name for slot in self._slots))

    def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
        """展开模式并添加 MuJoCo 传感器（每个主要元素 x 字段对一个）。"""
        self._slots.clear()

        primary_names = self._resolve_primary_names(entities, self.cfg.primary)
        if self.cfg.secondary is None or self.cfg.secondary_policy == "any":
            secondary_name = None
        else:
            secondary_name = self._resolve_single_secondary(
                entities, self.cfg.secondary, self.cfg.secondary_policy
            )

        # MuJoCo 允许通过 `dataspec` 位域将多个字段打包到一个接触传感器中，
        # 但我们为每个 (主要元素, 字段) 对注册一个传感器，这样每个传感器的
        # `sensordata` 块布局为 `[B, num_slots * dim]`，`_extract_sensor_data`
        # 可以按字段进行 reshape，而无需在交错的逐槽位布局中计算逐字段偏移。
        for prim in primary_names:
            for field in self.cfg.fields:
                sensor_name = f"{self.cfg.name}_{prim}_{field}"

                self._add_contact_sensor_to_spec(
                    scene_spec, sensor_name, prim, secondary_name, field
                )

                self._slots.append(
                    _ContactSlot(
                        primary_name=prim,
                        field_name=field,
                        sensor_name=sensor_name,
                    )
                )

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        """将传感器映射到 sensordata 缓冲区并分配空中时间状态。"""
        del model

        if not self._slots:
            raise RuntimeError(
                f"There was an error initializing contact sensor '{self.cfg.name}'"
            )

        for slot in self._slots:
            sensor = mj_model.sensor(slot.sensor_name)
            start = sensor.adr[0]
            dim = sensor.dim[0]
            slot.data_view = data.sensordata[:, start : start + dim]

        self._data = data
        self._device = device

        n_primary = len(self.primary_names)

        if self.cfg.track_air_time:
            n_envs = data.time.shape[0]
            self._air_time_state = _AirTimeState(
                current_air_time=torch.zeros((n_envs, n_primary), device=device),
                last_air_time=torch.zeros((n_envs, n_primary), device=device),
                current_contact_time=torch.zeros((n_envs, n_primary), device=device),
                last_contact_time=torch.zeros((n_envs, n_primary), device=device),
                last_time=torch.zeros((n_envs,), device=device),
            )

        if self.cfg.history_length > 0:
            n_envs = data.time.shape[0]
            n_contacts = n_primary * self.cfg.num_slots
            h = self.cfg.history_length
            self._history_state = {}
            if "force" in self.cfg.fields:
                self._history_state["force"] = torch.zeros(
                    (n_envs, n_contacts, h, 3), device=device
                )
            if "torque" in self.cfg.fields:
                self._history_state["torque"] = torch.zeros(
                    (n_envs, n_contacts, h, 3), device=device
                )
            if "dist" in self.cfg.fields:
                self._history_state["dist"] = torch.zeros(
                    (n_envs, n_contacts, h), device=device
                )

    def _compute_data(self) -> ContactData:
        out = self._extract_sensor_data()
        if self._air_time_state is not None:
            out.current_air_time = self._air_time_state.current_air_time
            out.last_air_time = self._air_time_state.last_air_time
            out.current_contact_time = self._air_time_state.current_contact_time
            out.last_contact_time = self._air_time_state.last_contact_time
        if self._history_state is not None:
            out.force_history = self._history_state.get("force")
            out.torque_history = self._history_state.get("torque")
            out.dist_history = self._history_state.get("dist")
        return out

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)

        # 重置指定环境的空中时间状态。
        if self._air_time_state is not None:
            self._air_time_state.current_air_time[env_ids] = 0.0
            self._air_time_state.last_air_time[env_ids] = 0.0
            self._air_time_state.current_contact_time[env_ids] = 0.0
            self._air_time_state.last_contact_time[env_ids] = 0.0
            if self._data is not None:
                self._air_time_state.last_time[env_ids] = self._data.time[env_ids]

        # 重置指定环境的历史状态。
        if self._history_state is not None:
            for buf in self._history_state.values():
                buf[env_ids] = 0.0

    def update(self, dt: float) -> None:
        super().update(dt)
        if self._air_time_state is not None:
            self._update_air_time_tracking()
        if self._history_state is not None:
            self._update_history()

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-6) -> torch.Tensor:
        """返回 [B, P] 布尔张量：在最近 dt 秒内着地的主要元素为 True。"""
        if self._air_time_state is None:
            raise RuntimeError(
                f"Sensor '{self.cfg.name}' must have track_air_time=True "
                "to use compute_first_contact"
            )
        is_in_contact = self._air_time_state.current_contact_time > 0.0
        within_dt = self._air_time_state.current_contact_time < (dt + abs_tol)
        return is_in_contact & within_dt

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-6) -> torch.Tensor:
        """返回 [B, P] 布尔张量：在最近 dt 秒内离地的主要元素为 True。"""
        if self._air_time_state is None:
            raise RuntimeError(
                f"Sensor '{self.cfg.name}' must have track_air_time=True "
                "to use compute_first_air"
            )
        is_in_air = self._air_time_state.current_air_time > 0.0
        within_dt = self._air_time_state.current_air_time < (dt + abs_tol)
        return is_in_air & within_dt

    def _extract_sensor_data(self) -> ContactData:
        if not self._slots:
            raise RuntimeError(f"Sensor '{self.cfg.name}' not initialized")

        field_chunks: dict[str, list[torch.Tensor]] = {f: [] for f in self.cfg.fields}

        for slot in self._slots:
            assert slot.data_view is not None
            field_dim = _CONTACT_DATA_DIMS[slot.field_name]
            raw = slot.data_view.view(
                slot.data_view.size(0), self.cfg.num_slots, field_dim
            )
            field_chunks[slot.field_name].append(raw)

        out = ContactData()
        for field, chunks in field_chunks.items():
            cat = torch.cat(chunks, dim=1)
            if cat.size(-1) == 1:
                cat = cat.squeeze(-1)
            setattr(out, field, cat)

        if self.cfg.global_frame and self.cfg.reduce != "netforce":
            out = self._transform_to_global_frame(out)

        return out

    def _transform_to_global_frame(self, data: ContactData) -> ContactData:
        """将力/力矩从接触坐标系旋转到全局坐标系。"""
        assert data.normal is not None and data.tangent is not None

        normal = data.normal
        tangent = data.tangent
        tangent2 = torch.cross(normal, tangent, dim=-1)
        R = torch.stack([normal, tangent, tangent2], dim=-1)

        has_contact = torch.norm(normal, dim=-1, keepdim=True) > 1e-8

        if data.force is not None:
            force_global = torch.einsum("...ij,...j->...i", R, data.force)
            data.force = torch.where(has_contact, force_global, data.force)

        if data.torque is not None:
            torque_global = torch.einsum("...ij,...j->...i", R, data.torque)
            data.torque = torch.where(has_contact, torque_global, data.torque)

        return data

    def _update_air_time_tracking(self) -> None:
        assert self._air_time_state is not None

        contact_data = self._extract_sensor_data()
        if contact_data.found is None or "found" not in self.cfg.fields:
            return

        assert self._data is not None
        current_time = self._data.time
        elapsed_time = current_time - self._air_time_state.last_time
        elapsed_time = elapsed_time.unsqueeze(-1)

        # 将 `found` 从 [B, P*num_slots] 归约为 [B, P]：如果主要元素的任何
        # 槽位报告了匹配，则该主要元素处于接触状态。空中时间按主要元素跟踪。
        found = contact_data.found
        if self.cfg.num_slots > 1:
            found = found.view(found.size(0), -1, self.cfg.num_slots).any(dim=-1)
        is_contact = found > 0

        state = self._air_time_state
        is_first_contact = (state.current_air_time > 0) & is_contact
        is_first_detached = (state.current_contact_time > 0) & ~is_contact

        state.last_air_time[:] = torch.where(
            is_first_contact,
            state.current_air_time + elapsed_time,
            state.last_air_time,
        )
        state.current_air_time[:] = torch.where(
            ~is_contact,
            state.current_air_time + elapsed_time,
            torch.zeros_like(state.current_air_time),
        )

        state.last_contact_time[:] = torch.where(
            is_first_detached,
            state.current_contact_time + elapsed_time,
            state.last_contact_time,
        )
        state.current_contact_time[:] = torch.where(
            is_contact,
            state.current_contact_time + elapsed_time,
            torch.zeros_like(state.current_contact_time),
        )

        state.last_time[:] = current_time

    def _update_history(self) -> None:
        """滚动历史缓冲区并将当前接触数据插入索引 0。"""
        assert self._history_state is not None

        contact_data = self._extract_sensor_data()

        if "force" in self._history_state and contact_data.force is not None:
            self._history_state["force"] = self._history_state["force"].roll(1, dims=2)
            self._history_state["force"][:, :, 0, :] = contact_data.force

        if "torque" in self._history_state and contact_data.torque is not None:
            self._history_state["torque"] = self._history_state["torque"].roll(
                1, dims=2
            )
            self._history_state["torque"][:, :, 0, :] = contact_data.torque

        if "dist" in self._history_state and contact_data.dist is not None:
            self._history_state["dist"] = self._history_state["dist"].roll(1, dims=2)
            self._history_state["dist"][:, :, 0] = contact_data.dist

    def _resolve_primary_names(
        self, entities: dict[str, Entity], match: ContactMatch
    ) -> list[str]:
        if match.entity in (None, ""):
            result = (
                [match.pattern]
                if isinstance(match.pattern, str)
                else list(match.pattern)
            )
            return result

        if match.entity not in entities:
            raise ValueError(
                f"Primary entity '{match.entity}' not found. Available: {list(entities.keys())}"
            )
        ent = entities[match.entity]

        patterns = [match.pattern] if isinstance(match.pattern, str) else match.pattern

        if match.mode == "geom":
            _, names = ent.find_geoms(patterns)
        elif match.mode == "body":
            _, names = ent.find_bodies(patterns)
        elif match.mode == "subtree":
            _, names = ent.find_bodies(patterns)
            if not names:
                raise ValueError(
                    f"Primary subtree pattern '{match.pattern}' matched no bodies in "
                    f"'{match.entity}'"
                )
        else:
            raise ValueError("Primary mode must be one of {'geom','body','subtree'}")

        excludes = match.exclude
        if excludes:
            exclude_patterns = []
            exclude_exact = set()
            for exc in excludes:
                if any(c in exc for c in r".*+?[]{}()\|^$"):
                    exclude_patterns.append(re.compile(exc))
                else:
                    exclude_exact.add(exc)
            if exclude_exact:
                names = [n for n in names if n not in exclude_exact]
            if exclude_patterns:
                names = [
                    n for n in names if not any(rx.search(n) for rx in exclude_patterns)
                ]

        if not names:
            raise ValueError(
                f"Primary pattern '{match.pattern}' (after excludes) matched "
                f"no names in '{match.entity}'"
            )
        return names

    def _resolve_single_secondary(
        self,
        entities: dict[str, Entity],
        match: ContactMatch,
        policy: Literal["first", "any", "error"],
    ) -> str | None:
        if policy == "any":
            return None

        if isinstance(match.pattern, tuple):
            raise ValueError(
                "Secondary must specify a single name (string). "
                "Use a single exact name or a regex that resolves to one name, "
                "or set secondary_policy='any' if you want no filter."
            )

        if match.entity in (None, ""):
            if match.mode not in {"geom", "body", "subtree"}:
                raise ValueError(
                    "Secondary mode must be one of {'geom','body','subtree'}"
                )
            return match.pattern

        if match.entity not in entities:
            raise ValueError(
                f"Secondary entity '{match.entity}' not found. "
                f"Available: {list(entities.keys())}"
            )
        ent = entities[match.entity]

        if match.mode == "subtree":
            return match.pattern

        if match.mode == "geom":
            _, names = ent.find_geoms(match.pattern)
        elif match.mode == "body":
            _, names = ent.find_bodies(match.pattern)
        else:
            raise ValueError("Secondary mode must be one of {'geom','body','subtree'}")

        if not names:
            raise ValueError(
                f"Secondary pattern '{match.pattern}' matched nothing in '{match.entity}'"
            )

        if len(names) == 1 or policy == "first":
            return names[0]

        raise ValueError(
            f"Secondary pattern '{match.pattern}' matched multiple: {names}. "
            f"Be explicit or set secondary_policy='first' or 'any'."
        )

    def _add_contact_sensor_to_spec(
        self,
        scene_spec: mujoco.MjSpec,
        sensor_name: str,
        primary_name: str,
        secondary_name: str | None,
        field: str,
    ) -> None:
        data_bits = 1 << _CONTACT_DATA_MAP[field]
        reduce_mode = _CONTACT_REDUCE_MAP[self.cfg.reduce]
        intprm = [data_bits, reduce_mode, self.cfg.num_slots]

        kwargs: dict[str, Any] = {
            "name": sensor_name,
            "type": mujoco.mjtSensor.mjSENS_CONTACT,
            "objtype": _MODE_TO_OBJTYPE[self.cfg.primary.mode],
            "objname": _prefix_name(primary_name, self.cfg.primary.entity),
            "intprm": intprm,
        }

        if secondary_name is not None:
            assert self.cfg.secondary is not None
            kwargs["reftype"] = _MODE_TO_OBJTYPE[self.cfg.secondary.mode]
            kwargs["refname"] = _prefix_name(secondary_name, self.cfg.secondary.entity)

        if self.cfg.debug:
            self._print_debug(sensor_name, field, intprm, kwargs)

        scene_spec.add_sensor(**kwargs)

    def _print_debug(
        self,
        sensor_name: str,
        field: str,
        intprm: list[int],
        kwargs: dict[str, Any],
    ) -> None:
        objtype_name = _objtype_name(kwargs["objtype"])
        reftype_val = kwargs.get("reftype")
        refname_val = kwargs.get("refname")
        if refname_val is None:
            ref_str = "<any>"
        else:
            ref_str = f"{_objtype_name(reftype_val)}:{refname_val}"
        print(
            "Adding contact sensor\n"
            f"  name    : {sensor_name}\n"
            f"  object  : {objtype_name}:{kwargs['objname']}\n"
            f"  ref     : {ref_str}\n"
            f"  field   : {field}  bits=0b{intprm[0]:b}\n"
            f"  reduce  : {self.cfg.reduce}  num_slots={self.cfg.num_slots}"
        )


def _prefix_name(name: str, entity: str | None) -> str:
    """当设置了实体作用域时，在 MuJoCo 名称前添加 ``entity/`` 前缀。"""
    if entity:
        return f"{entity}/{name}"
    return name


def _objtype_name(objtype: Any) -> str:
    """美化打印 MuJoCo 对象类型，去掉 ``mjOBJ_`` 前缀。"""
    return getattr(objtype, "name", str(objtype)).removeprefix("mjOBJ_")
