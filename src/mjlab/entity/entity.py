from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch

from mjlab import actuator
from mjlab.actuator import BuiltinActuatorGroup
from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.xml_actuator import XmlActuator
from mjlab.entity.data import EntityData
from mjlab.utils import spec_config as spec_cfg
from mjlab.utils.lab_api.string import resolve_matching_names
from mjlab.utils.mujoco import dof_width, qpos_width
from mjlab.utils.spec import auto_wrap_fixed_base_mocap
from mjlab.utils.string import resolve_expr
from mjlab.utils.xml import fix_spec_xml, strip_buffer_textures

if TYPE_CHECKING:
    from mjlab.entity.variants import VariantMetadata


@dataclass(frozen=False)
class EntityIndexing:
    """将实体元素映射到仿真中的全局索引和地址。"""

    # 元素。
    bodies: tuple[mujoco.MjsBody, ...]
    joints: tuple[mujoco.MjsJoint, ...]
    geoms: tuple[mujoco.MjsGeom, ...]
    sites: tuple[mujoco.MjsSite, ...]
    tendons: tuple[mujoco.MjsTendon, ...]
    cameras: tuple[mujoco.MjsCamera, ...]
    lights: tuple[mujoco.MjsLight, ...]
    materials: tuple[mujoco.MjsMaterial, ...]
    pairs: tuple[mujoco.MjsPair, ...]
    actuators: tuple[mujoco.MjsActuator, ...] | None

    # 索引。
    body_ids: torch.Tensor
    geom_ids: torch.Tensor
    site_ids: torch.Tensor
    tendon_ids: torch.Tensor
    cam_ids: torch.Tensor
    light_ids: torch.Tensor
    mat_ids: torch.Tensor
    pair_ids: torch.Tensor
    ctrl_ids: torch.Tensor
    joint_ids: torch.Tensor
    mocap_id: int | None

    # 地址。
    joint_q_adr: torch.Tensor
    joint_v_adr: torch.Tensor
    free_joint_q_adr: torch.Tensor
    free_joint_v_adr: torch.Tensor

    @property
    def root_body_id(self) -> int:
        return self.bodies[0].id


@dataclass
class EntityCfg:
    @dataclass
    class InitialStateCfg:
        # 根位置与朝向。
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        # 根线速度与角速度（仅用于浮动基座实体）。
        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        # 关节（仅用于关节实体）。
        # 设为 None 则使用模型的现有关键帧（若不存在则报错）。
        joint_pos: dict[str, float] | None = field(default_factory=lambda: {".*": 0.0})
        joint_vel: dict[str, float] = field(default_factory=lambda: {".*": 0.0})

    init_state: InitialStateCfg = field(default_factory=InitialStateCfg)
    spec_fn: Callable[[], mujoco.MjSpec] = field(
        default_factory=lambda: (lambda: mujoco.MjSpec())
    )
    articulation: EntityArticulationInfoCfg | None = None
    sort_actuators: bool = False
    """当为 True 时，重新排列驱动器使 ``model.ctrl`` 遵循关节/腱/站点的
    定义顺序，而非驱动器的配置出现顺序。XML 驱动器不受排序影响，
    始终保留其声明顺序。
    """

    # 编辑器。
    lights: tuple[spec_cfg.LightCfg, ...] = field(default_factory=tuple)
    cameras: tuple[spec_cfg.CameraCfg, ...] = field(default_factory=tuple)
    textures: tuple[spec_cfg.TextureCfg, ...] = field(default_factory=tuple)
    materials: tuple[spec_cfg.MaterialCfg, ...] = field(default_factory=tuple)
    collisions: tuple[spec_cfg.CollisionCfg, ...] = field(default_factory=tuple)

    def build(self) -> Entity:
        """从此配置构建实体实例。

        在子类中重写以返回自定义 Entity 类型。
        """
        return Entity(self)


@dataclass
class EntityArticulationInfoCfg:
    actuators: tuple[actuator.ActuatorCfg, ...] = field(default_factory=tuple)
    soft_joint_pos_limit_factor: float = 1.0


class Entity:
    """实体表示仿真中的一个物理对象。

    实体类型矩阵
    ============
    MuJoCo 实体可沿两个维度进行分类：

    1. 基座类型：
      - 固定基座：实体焊接到世界空间（无 freejoint）
      - 浮动基座：实体有 6 自由度运动（有 freejoint）

    2. 关节：
      - 非关节式：除 freejoint 外无其他关节
      - 关节式：运动学树中有关节（可能有驱动器也可能没有）

    固定非关节式实体可选项为 mocap 物体，其位置和朝向可在每个时间步
    直接设置，而非由物理决定。此特性对于创建可调整位置和朝向的道具
    非常有用。

    支持的组合：
    ----------------------
    | 类型                     | 示例              | is_fixed_base | is_articulated | is_actuated |
    |--------------------------|-------------------|---------------|----------------|-------------|
    | 固定非关节式             | 桌子、墙壁        | True          | False          | False       |
    | 固定关节式               | 机械臂、门        | True          | True           | True/False  |
    | 浮动非关节式             | 盒子、球、杯子    | False         | False          | False       |
    | 浮动关节式               | 人形机器人、四足  | False         | True           | True/False  |
    """

    def __init__(self, cfg: EntityCfg) -> None:
        self.cfg = cfg
        self._actuators: list[actuator.Actuator] = []
        self._variant_metadata: VariantMetadata | None = None
        self._build_spec()
        self._identify_joints()
        self._apply_spec_editors()
        self._add_actuators()
        self._add_initial_state_keyframe()

    def _build_spec(self) -> None:
        from mjlab.entity.variants import VariantEntityCfg, build_merged_variant_spec

        if isinstance(self.cfg, VariantEntityCfg):
            self._spec, self._variant_metadata = build_merged_variant_spec(self.cfg)
        else:
            self._spec = auto_wrap_fixed_base_mocap(self.cfg.spec_fn)()

    @property
    def variant_metadata(self) -> VariantMetadata | None:
        return self._variant_metadata

    def _identify_joints(self) -> None:
        self._all_joints = self._spec.joints
        self._free_joint = None
        self._non_free_joints = tuple(self._all_joints)
        if self._all_joints and self._all_joints[0].type == mujoco.mjtJoint.mjJNT_FREE:
            self._free_joint = self._all_joints[0]
            if not self._free_joint.name:
                self._free_joint.name = "floating_base_joint"
            self._non_free_joints = tuple(self._all_joints[1:])

    def _apply_spec_editors(self) -> None:
        for cfg_list in [
            self.cfg.lights,
            self.cfg.cameras,
            self.cfg.textures,
            self.cfg.materials,
            self.cfg.collisions,
        ]:
            for cfg in cfg_list:
                cfg.edit_spec(self._spec)

    def _add_actuators(self) -> None:
        if self.cfg.articulation is None:
            return

        # 收集驱动器实例及其目标。
        pending: list[tuple[actuator.ActuatorCfg, actuator.Actuator, list[str]]] = []
        for actuator_cfg in self.cfg.articulation.actuators:
            # 根据传输类型查找目标。resolve_matching_names 在正则无匹配时
            # 抛出 ValueError；我们捕获它以便在下方给出带有命名空间提示的
            # 更友好错误信息。
            target_ids: list[int] = []
            target_names: list[str] = []
            target_spec_names: list[str] = []
            try:
                if actuator_cfg.transmission_type == TransmissionType.JOINT:
                    target_ids, target_names = self.find_joints(
                        actuator_cfg.target_names_expr
                    )
                    target_spec_names = [
                        self._non_free_joints[i].name for i in target_ids
                    ]
                elif actuator_cfg.transmission_type == TransmissionType.TENDON:
                    target_ids, target_names = self.find_tendons(
                        actuator_cfg.target_names_expr
                    )
                    target_spec_names = [self._spec.tendons[i].name for i in target_ids]
                elif actuator_cfg.transmission_type == TransmissionType.SITE:
                    target_ids, target_names = self.find_sites(
                        actuator_cfg.target_names_expr
                    )
                    target_spec_names = [self.spec.sites[i].name for i in target_ids]
                else:
                    raise TypeError(
                        f"Invalid transmission_type: {actuator_cfg.transmission_type}. "
                        f"Must be TransmissionType.JOINT, TransmissionType.TENDON, "
                        f"or TransmissionType.SITE."
                    )
            except ValueError:
                pass  # target_names 保持为空，继续进入提示逻辑

            # 检查其他命名空间是否有匹配。如果什么都没找到，则产生
            # 有帮助的错误信息。如果确实找到了目标，则对未驱动的其他命名空间
            # 匹配发出警告。
            current = actuator_cfg.transmission_type
            other_matches: dict[TransmissionType, tuple[str, list[str]]] = {}
            other_namespaces = {
                TransmissionType.JOINT: ("joint", self.joint_names),
                TransmissionType.TENDON: ("tendon", self.tendon_names),
                TransmissionType.SITE: ("site", self.site_names),
            }
            for tt, (label, names) in other_namespaces.items():
                if tt == current or not names:
                    continue
                try:
                    _, matched = resolve_matching_names(
                        actuator_cfg.target_names_expr, names
                    )
                    other_matches[tt] = (label, matched)
                except ValueError:
                    pass

            if len(target_names) == 0:
                msg = f"No {current.value}s matched expressions: {actuator_cfg.target_names_expr}"
                if other_matches:
                    hints = [
                        f"{label}s ({', '.join(matched)})"
                        for label, matched in other_matches.values()
                    ]
                    msg += (
                        f". Matches were found in: {'; '.join(hints)}. "
                        f"Check that transmission_type is correct."
                    )
                raise ValueError(msg)

            for tt, (label, matched) in other_matches.items():
                warnings.warn(
                    f"Actuator config matched {len(target_names)} {current.value}(s) "
                    f"but the same expressions also match {len(matched)} {label}(s): "
                    f"{', '.join(matched)}. Add a separate config with "
                    f"transmission_type=TransmissionType.{tt.name} if those should "
                    f"be actuated too.",
                    stacklevel=2,
                )

            actuator_instance = actuator_cfg.build(self, target_ids, target_names)
            self._actuators.append(actuator_instance)
            pending.append((actuator_cfg, actuator_instance, target_spec_names))

        if not self.cfg.sort_actuators:
            for _, inst, names in pending:
                inst.edit_spec(self._spec, names)
            return

        # 排序驱动器使 ctrl 顺序与关节/腱/站点定义顺序匹配。
        # XmlActuators 最先添加（它们封装已有的 XML 驱动器），
        # 然后按传输类型和目标顺序排列其余驱动器。
        order_maps = {
            TransmissionType.JOINT: {
                name: i for i, name in enumerate(self.joint_names)
            },
            TransmissionType.TENDON: {
                name: i for i, name in enumerate(self.tendon_names)
            },
            TransmissionType.SITE: {name: i for i, name in enumerate(self.site_names)},
        }
        # 按传输类型分组（排序是惯例性的，非物理驱动）。
        # 每组内部，驱动器按其目标在 spec 中的定义顺序排序。
        type_priority = {
            TransmissionType.JOINT: 0,
            TransmissionType.TENDON: 1,
            TransmissionType.SITE: 2,
        }

        # XmlActuators 按声明顺序最先添加（它们引用 spec 中已有的驱动器）。
        for _, inst, names in pending:
            if isinstance(inst, XmlActuator):
                inst.edit_spec(self._spec, names)

        # 将其余驱动器展开为 (instance, single_target) 对并排序。
        flat: list[tuple[actuator.ActuatorCfg, actuator.Actuator, str]] = []
        for cfg, inst, names in pending:
            if not isinstance(inst, XmlActuator):
                for name in names:
                    flat.append((cfg, inst, name))

        flat.sort(
            key=lambda item: (
                type_priority[item[0].transmission_type],
                order_maps[item[0].transmission_type].get(item[2], float("inf")),
            )
        )
        for _, inst, name in flat:
            inst.edit_spec(self._spec, [name])

    def _add_initial_state_keyframe(self) -> None:
        # 若 joint_pos 为 None，则使用模型的现有关键帧。
        if self.cfg.init_state.joint_pos is None:
            if not self._spec.keys:
                raise ValueError(
                    "joint_pos=None requires the model to have a keyframe, but none exists."
                )
            # 保留现有关键帧，仅重命名。
            self._spec.keys[0].name = "init_state"
            if self.is_fixed_base:
                self.root_body.pos[:] = self.cfg.init_state.pos
                self.root_body.quat[:] = self.cfg.init_state.rot
            return

        qpos_components = []

        if self._free_joint is not None:
            qpos_components.extend([self.cfg.init_state.pos, self.cfg.init_state.rot])

        joint_pos = None
        if self._non_free_joints:
            joint_pos = resolve_expr(
                self.cfg.init_state.joint_pos, self.joint_names, 0.0
            )
            qpos_components.append(joint_pos)

        key_qpos = np.hstack(qpos_components) if qpos_components else np.array([])
        key = self._spec.add_key(name="init_state", qpos=key_qpos.tolist())

        if self.is_actuated and joint_pos is not None:
            name_to_pos = {
                name: joint_pos[i] for i, name in enumerate(self.joint_names)
            }
            ctrl = []
            for act in self._spec.actuators:
                joint_name = act.target
                ctrl.append(name_to_pos.get(joint_name, 0.0))
            key.ctrl = np.array(ctrl)

        if self.is_fixed_base:
            self.root_body.pos[:] = self.cfg.init_state.pos
            self.root_body.quat[:] = self.cfg.init_state.rot

    # 属性。

    @property
    def is_fixed_base(self) -> bool:
        """实体是否焊接到世界空间。"""
        return self._free_joint is None

    @property
    def is_articulated(self) -> bool:
        """实体是否有（固定或可驱动的）关节。"""
        return len(self._non_free_joints) > 0

    @property
    def is_actuated(self) -> bool:
        """实体是否有可驱动关节。"""
        return len(self._actuators) > 0

    @property
    def has_tendon_actuators(self) -> bool:
        """实体是否有使用腱传动的驱动器。"""
        if self.cfg.articulation is None:
            return False
        return any(
            act.transmission_type == TransmissionType.TENDON
            for act in self.cfg.articulation.actuators
        )

    @property
    def has_site_actuators(self) -> bool:
        """实体是否有使用站点传动的驱动器。"""
        if self.cfg.articulation is None:
            return False
        return any(
            act.transmission_type == TransmissionType.SITE
            for act in self.cfg.articulation.actuators
        )

    @property
    def is_mocap(self) -> bool:
        """实体根刚体是否是 mocap 物体（仅用于固定基座实体）。"""
        return bool(self.root_body.mocap) if self.is_fixed_base else False

    @property
    def spec(self) -> mujoco.MjSpec:
        return self._spec

    @property
    def data(self) -> EntityData:
        return self._data

    @property
    def actuators(self) -> list[actuator.Actuator]:
        return self._actuators

    # 名称。

    @property
    def body_names(self) -> tuple[str, ...]:
        return tuple(b.name.split("/")[-1] for b in self.spec.bodies[1:])

    @property
    def all_joint_names(self) -> tuple[str, ...]:
        return tuple(j.name.split("/")[-1] for j in self._all_joints)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(j.name.split("/")[-1] for j in self._non_free_joints)

    @property
    def geom_names(self) -> tuple[str, ...]:
        return tuple(g.name.split("/")[-1] for g in self.spec.geoms)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name.split("/")[-1] for s in self.spec.sites)

    @property
    def tendon_names(self) -> tuple[str, ...]:
        return tuple(t.name.split("/")[-1] for t in self._spec.tendons)

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(c.name.split("/")[-1] for c in self.spec.cameras)

    @property
    def light_names(self) -> tuple[str, ...]:
        return tuple(lt.name.split("/")[-1] for lt in self.spec.lights)

    @property
    def material_names(self) -> tuple[str, ...]:
        return tuple(m.name.split("/")[-1] for m in self.spec.materials)

    @property
    def pair_names(self) -> tuple[str, ...]:
        return tuple(p.name.split("/")[-1] for p in self.spec.pairs)

    @property
    def actuator_names(self) -> tuple[str, ...]:
        return tuple(a.name.split("/")[-1] for a in self.spec.actuators)

    # 计数。

    @property
    def num_bodies(self) -> int:
        return len(self.body_names)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def num_geoms(self) -> int:
        return len(self.geom_names)

    @property
    def num_sites(self) -> int:
        return len(self.site_names)

    @property
    def num_tendons(self) -> int:
        return len(self.tendon_names)

    @property
    def num_cameras(self) -> int:
        return len(self.camera_names)

    @property
    def num_lights(self) -> int:
        return len(self.light_names)

    @property
    def num_materials(self) -> int:
        return len(self.material_names)

    @property
    def num_pairs(self) -> int:
        return len(self.pair_names)

    @property
    def num_actuators(self) -> int:
        return len(self.actuator_names)

    @property
    def root_body(self) -> mujoco.MjsBody:
        return self.spec.bodies[1]

    # 查找方法。

    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        return resolve_matching_names(name_keys, self.body_names, preserve_order)

    def find_joints(
        self,
        name_keys: str | Sequence[str],
        joint_subset: Sequence[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        if joint_subset is None:
            joint_subset = self.joint_names
        return resolve_matching_names(name_keys, joint_subset, preserve_order)

    def find_joints_by_actuator_names(
        self,
        actuator_name_keys: str | Sequence[str],
    ) -> tuple[list[int], list[str]]:
        # 收集所有被驱动的关节名称。
        actuated_joint_names_set = set()
        for act in self._actuators:
            actuated_joint_names_set.update(act.target_names)

        # 过滤 self.joint_names 仅保留被驱动关节，保持自然顺序。
        actuated_in_natural_order = [
            name for name in self.joint_names if name in actuated_joint_names_set
        ]

        # 在被驱动关节中查找匹配模式的关节。
        _, matched_joint_names = self.find_joints(
            actuator_name_keys,
            joint_subset=actuated_in_natural_order,
            preserve_order=False,
        )

        # 将关节名称映射回实体本地索引（self.joint_names 中的索引）。
        name_to_entity_idx = {name: i for i, name in enumerate(self.joint_names)}
        joint_ids = [name_to_entity_idx[name] for name in matched_joint_names]
        return joint_ids, matched_joint_names

    def find_geoms(
        self,
        name_keys: str | Sequence[str],
        geom_subset: Sequence[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        if geom_subset is None:
            geom_subset = self.geom_names
        return resolve_matching_names(name_keys, geom_subset, preserve_order)

    def find_sites(
        self,
        name_keys: str | Sequence[str],
        site_subset: Sequence[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        if site_subset is None:
            site_subset = self.site_names
        return resolve_matching_names(name_keys, site_subset, preserve_order)

    def find_tendons(
        self,
        name_keys: str | Sequence[str],
        tendon_subset: Sequence[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        if tendon_subset is None:
            tendon_subset = self.tendon_names
        return resolve_matching_names(name_keys, tendon_subset, preserve_order)

    def find_cameras(
        self,
        name_keys: str | Sequence[str],
        camera_subset: Sequence[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        if camera_subset is None:
            camera_subset = self.camera_names
        return resolve_matching_names(name_keys, camera_subset, preserve_order)

    def find_lights(
        self,
        name_keys: str | Sequence[str],
        light_subset: Sequence[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        if light_subset is None:
            light_subset = self.light_names
        return resolve_matching_names(name_keys, light_subset, preserve_order)

    def find_materials(
        self,
        name_keys: str | Sequence[str],
        material_subset: Sequence[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        if material_subset is None:
            material_subset = self.material_names
        return resolve_matching_names(name_keys, material_subset, preserve_order)

    def find_pairs(
        self,
        name_keys: str | Sequence[str],
        pair_subset: Sequence[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        if pair_subset is None:
            pair_subset = self.pair_names
        return resolve_matching_names(name_keys, pair_subset, preserve_order)

    def find_actuators(
        self,
        name_keys: str | Sequence[str],
        actuator_subset: Sequence[str] | None = None,
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        if actuator_subset is None:
            actuator_subset = self.actuator_names
        return resolve_matching_names(name_keys, actuator_subset, preserve_order)

    def compile(self) -> mujoco.MjModel:
        """将底层的 MjSpec 编译为 MjModel。"""
        return self.spec.compile()

    def write_xml(self, xml_path: Path) -> None:
        """将 MjSpec 写入磁盘。

        在 spec 的副本上操作以避免修改原数据。
        """
        tmp = self.spec.copy()
        strip_buffer_textures(tmp)
        xml_path.write_text(fix_spec_xml(tmp.to_xml()))

    def to_zip(self, path: Path) -> None:
        """将 MjSpec 写入 zip 文件。"""
        with path.open("wb") as f:
            mujoco.MjSpec.to_zip(self.spec, f)

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        """在 spec 编译后为仿真准备实体。

        计算全局索引映射，初始化驱动器，并在 ``EntityData`` 中分配所有
        nworld 状态和目标张量。由场景在环境构造期间调用一次。
        """
        indexing = self._compute_indexing(mj_model, device)
        self.indexing = indexing
        nworld = data.nworld

        for act in self._actuators:
            act.initialize(mj_model, model, data, device)

        # 向量化内置驱动器；自定义驱动器将逐个循环处理。
        builtin_group, custom_actuators = BuiltinActuatorGroup.process(self._actuators)
        builtin_group.initialize(nworld, device)
        self._builtin_group = builtin_group
        self._custom_actuators = custom_actuators

        # 根状态。
        default_root_state = self._build_default_root_state(nworld, device)

        # 关节状态。
        if self.is_articulated:
            if self.cfg.init_state.joint_pos is None:
                # 使用关键帧中的关节位置。
                key_qpos = mj_model.key("init_state").qpos
                nq_root = 7 if not self.is_fixed_base else 0
                default_joint_pos = torch.tensor(key_qpos[nq_root:], device=device)[
                    None
                ].repeat(nworld, 1)
            else:
                default_joint_pos = torch.tensor(
                    resolve_expr(self.cfg.init_state.joint_pos, self.joint_names, 0.0),
                    device=device,
                )[None].repeat(nworld, 1)
            default_joint_vel = torch.tensor(
                resolve_expr(self.cfg.init_state.joint_vel, self.joint_names, 0.0),
                device=device,
            )[None].repeat(nworld, 1)

            # 关节限位。
            joint_ids_list = [j.id for j in self._non_free_joints]
            dof_limits = model.jnt_range[:, joint_ids_list]
            default_joint_pos_limits = dof_limits.clone()
            joint_pos_limits = default_joint_pos_limits.clone()

            joint_pos_mean = (joint_pos_limits[..., 0] + joint_pos_limits[..., 1]) / 2
            joint_pos_range = joint_pos_limits[..., 1] - joint_pos_limits[..., 0]

            # 软限位。
            soft_limit_factor = (
                self.cfg.articulation.soft_joint_pos_limit_factor
                if self.cfg.articulation
                else 1.0
            )
            soft_joint_pos_limits = torch.stack(
                [
                    joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor,
                    joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor,
                ],
                dim=-1,
            )

            # MuJoCo 中无限位的关节 jnt_range=[0,0]，使所有计算的限位
            # 为 [0,0]。覆盖为 [-inf, inf] 使下游裁剪成为空操作。
            # （不能在软限位计算之前做，因为 inf - inf = NaN。）
            unlimited = ~torch.tensor(
                mj_model.jnt_limited[joint_ids_list], device=device, dtype=torch.bool
            )
            for limits in (
                joint_pos_limits,
                default_joint_pos_limits,
                soft_joint_pos_limits,
            ):
                limits[:, unlimited, 0] = float("-inf")
                limits[:, unlimited, 1] = float("inf")
        else:
            empty_shape = (nworld, 0)
            default_joint_pos = torch.empty(
                *empty_shape, dtype=torch.float, device=device
            )
            default_joint_vel = torch.empty(
                *empty_shape, dtype=torch.float, device=device
            )
            default_joint_pos_limits = torch.empty(
                *empty_shape, 2, dtype=torch.float, device=device
            )
            joint_pos_limits = torch.empty(
                *empty_shape, 2, dtype=torch.float, device=device
            )
            soft_joint_pos_limits = torch.empty(
                *empty_shape, 2, dtype=torch.float, device=device
            )

        if self.is_actuated:
            joint_pos_target = torch.zeros(
                (nworld, self.num_joints), dtype=torch.float, device=device
            )
            joint_vel_target = torch.zeros(
                (nworld, self.num_joints), dtype=torch.float, device=device
            )
            joint_effort_target = torch.zeros(
                (nworld, self.num_joints), dtype=torch.float, device=device
            )
        else:
            joint_pos_target = torch.empty(nworld, 0, dtype=torch.float, device=device)
            joint_vel_target = torch.empty(nworld, 0, dtype=torch.float, device=device)
            joint_effort_target = torch.empty(
                nworld, 0, dtype=torch.float, device=device
            )

        # 仅在有使用腱传动的驱动器时才分配腱目标。
        if self.has_tendon_actuators:
            num_tendons = len(self.tendon_names)
            tendon_len_target = torch.zeros(
                (nworld, num_tendons), dtype=torch.float, device=device
            )
            tendon_vel_target = torch.zeros(
                (nworld, num_tendons), dtype=torch.float, device=device
            )
            tendon_effort_target = torch.zeros(
                (nworld, num_tendons), dtype=torch.float, device=device
            )
        else:
            tendon_len_target = torch.empty(nworld, 0, dtype=torch.float, device=device)
            tendon_vel_target = torch.empty(nworld, 0, dtype=torch.float, device=device)
            tendon_effort_target = torch.empty(
                nworld, 0, dtype=torch.float, device=device
            )

        # 仅在有使用站点传动的驱动器时才分配站点目标。
        if self.has_site_actuators:
            num_sites = len(self.site_names)
            site_effort_target = torch.zeros(
                (nworld, num_sites), dtype=torch.float, device=device
            )
        else:
            site_effort_target = torch.empty(
                nworld, 0, dtype=torch.float, device=device
            )

        # 编码器偏置，用于模拟编码器标定误差。
        # 形状：(num_envs, num_joints)。默认为零（无偏置）。
        if self.is_articulated:
            encoder_bias = torch.zeros(
                (nworld, self.num_joints), dtype=torch.float, device=device
            )
        else:
            encoder_bias = torch.empty(nworld, 0, dtype=torch.float, device=device)

        self._data = EntityData(
            indexing=indexing,
            data=data,
            model=model,
            device=device,
            default_root_state=default_root_state,
            default_joint_pos=default_joint_pos,
            default_joint_vel=default_joint_vel,
            default_joint_pos_limits=default_joint_pos_limits,
            joint_pos_limits=joint_pos_limits,
            soft_joint_pos_limits=soft_joint_pos_limits,
            gravity_vec_w=torch.tensor([0.0, 0.0, -1.0], device=device).repeat(
                nworld, 1
            ),
            forward_vec_b=torch.tensor([1.0, 0.0, 0.0], device=device).repeat(
                nworld, 1
            ),
            is_fixed_base=self.is_fixed_base,
            is_articulated=self.is_articulated,
            is_actuated=self.is_actuated,
            joint_pos_target=joint_pos_target,
            joint_vel_target=joint_vel_target,
            joint_effort_target=joint_effort_target,
            tendon_len_target=tendon_len_target,
            tendon_vel_target=tendon_vel_target,
            tendon_effort_target=tendon_effort_target,
            site_effort_target=site_effort_target,
            encoder_bias=encoder_bias,
        )

    def update(self, dt: float) -> None:
        """将驱动器内部状态推进一个物理子步。

        在 decimation 循环内的每次 ``sim.step()`` 之后调用。
        """
        for act in self._actuators:
            act.update(dt)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """清零驱动器目标并重置驱动器内部状态。

        当环境在 episode 边界处重置时，以及由在 episode 中途将机器人
        传送到新姿态的命令调用时，由场景调用。
        """
        self._data.clear_state(env_ids)

        for act in self._actuators:
            act.reset(env_ids)

    def clear_state(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """已弃用。请使用 ``reset`` 替代。"""
        warnings.warn(
            "Entity.clear_state() is deprecated. Use Entity.reset().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.reset(env_ids)

    def write_data_to_sim(self) -> None:
        """将驱动器目标转换为低级控制并写入仿真。

        在 decimation 循环内的每次 ``sim.step()`` 之前调用。内置驱动器
        通过单次批量操作施加；自定义驱动器逐个施加。
        """
        self._apply_actuator_controls()

    def write_ctrl_to_sim(
        self,
        ctrl: torch.Tensor,
        ctrl_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """将控制输入写入仿真。

        Args:
          ctrl: 控制输入张量。
          ctrl_ids: 控制索引张量。
          env_ids: 可选张量或切片，指定要设置的环境。若为 None，则设置所有环境。
        """
        self._data.write_ctrl(ctrl, ctrl_ids, env_ids)

    def write_root_state_to_sim(
        self, root_state: torch.Tensor, env_ids: torch.Tensor | slice | None = None
    ) -> None:
        """将根状态写入仿真。

        根状态包含：位置 (3)、朝向四元数 (w,x,y,z) (4)、
        线速度 (3) 和角速度 (3)，共 13 个值。所有量均在世界坐标系中。

        Args:
          root_state: 形状为 (N, 13) 的张量，N 为环境数量。
          env_ids: 可选张量或切片，指定要设置的环境。若为 None，则设置所有环境。
        """
        self._data.write_root_state(root_state, env_ids)

    def write_root_link_pose_to_sim(
        self,
        root_pose: torch.Tensor,
        env_ids: torch.Tensor | slice | None = None,
    ):
        """将根姿态写入仿真。类似 ``write_root_state_to_sim()``，
        但仅设置位置和朝向。

        Args:
          root_pose: 形状为 (N, 7) 的张量，N 为环境数量。
          env_ids: 可选张量或切片，指定要设置的环境。若为 None，则设置所有环境。
        """
        self._data.write_root_pose(root_pose, env_ids)

    def write_root_link_velocity_to_sim(
        self,
        root_velocity: torch.Tensor,
        env_ids: torch.Tensor | slice | None = None,
    ):
        """将根刚体（物体原点）速度写入仿真。类似 ``write_root_state_to_sim()``，
        但仅设置线速度和角速度。

        Args:
          root_velocity: 形状为 (N, 6) 的张量，N 为环境数量。
            包含物体原点处的线速度 (3) 和角速度 (3)，均在世界坐标系中。
          env_ids: 可选张量或切片，指定要设置的环境。若为 None，则设置所有环境。
        """
        self._data.write_root_velocity(root_velocity, env_ids)

    def write_root_com_velocity_to_sim(
        self,
        root_velocity: torch.Tensor,
        env_ids: torch.Tensor | slice | None = None,
    ):
        """将根质心速度写入仿真。

        Args:
          root_velocity: 形状为 (N, 6) 的张量，N 为环境数量。
            包含质心处的线速度 (3) 和角速度 (3)，均在世界坐标系中。
          env_ids: 可选张量或切片，指定要设置的环境。若为 None，则设置所有环境。
        """
        self._data.write_root_com_velocity(root_velocity, env_ids)

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ):
        """将关节状态写入仿真。

        关节状态包含关节位置和速度，不包括根状态。

        Args:
          position: 形状为 (N, num_joints) 的张量，N 为环境数量。
          velocity: 形状为 (N, num_joints) 的张量，N 为环境数量。
          joint_ids: 可选张量或切片，指定要设置的关节。若为 None，则设置所有关节。
          env_ids: 可选张量或切片，指定要设置的环境。若为 None，则设置所有环境。
        """
        self._data.write_joint_state(position, velocity, joint_ids, env_ids)

    def write_joint_position_to_sim(
        self,
        position: torch.Tensor,
        joint_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ):
        """将关节位置写入仿真。类似 ``write_joint_state_to_sim()``，
        但仅设置关节位置。

        Args:
          position: 形状为 (N, num_joints) 的张量，N 为环境数量。
          joint_ids: 可选张量或切片，指定要设置的关节。若为 None，则设置所有关节。
          env_ids: 可选张量或切片，指定要设置的环境。若为 None，则设置所有环境。
        """
        self._data.write_joint_position(position, joint_ids, env_ids)

    def write_joint_velocity_to_sim(
        self,
        velocity: torch.Tensor,
        joint_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ):
        """将关节速度写入仿真。类似 ``write_joint_state_to_sim()``，
        但仅设置关节速度。

        Args:
          velocity: 形状为 (N, num_joints) 的张量，N 为环境数量。
          joint_ids: 可选张量或切片，指定要设置的关节。若为 None，则设置所有关节。
          env_ids: 可选张量或切片，指定要设置的环境。若为 None，则设置所有环境。
        """
        self._data.write_joint_velocity(velocity, joint_ids, env_ids)

    def set_joint_position_target(
        self,
        position: torch.Tensor,
        joint_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """设置关节位置目标。

        Args:
          position: 目标关节位置，形状为 (N, num_joints)。
          joint_ids: 可选的要设置的关节索引。若为 None，则设置所有关节。
          env_ids: 可选的环境索引。若为 None，则设置所有环境。
        """
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        self._data.joint_pos_target[env_ids, joint_ids] = position

    def set_joint_velocity_target(
        self,
        velocity: torch.Tensor,
        joint_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """设置关节速度目标。

        Args:
          velocity: 目标关节速度，形状为 (N, num_joints)。
          joint_ids: 可选的要设置的关节索引。若为 None，则设置所有关节。
          env_ids: 可选的环境索引。若为 None，则设置所有环境。
        """
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        self._data.joint_vel_target[env_ids, joint_ids] = velocity

    def set_joint_effort_target(
        self,
        effort: torch.Tensor,
        joint_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """设置关节力矩目标。

        Args:
          effort: 目标关节力矩，形状为 (N, num_joints)。
          joint_ids: 可选的要设置的关节索引。若为 None，则设置所有关节。
          env_ids: 可选的环境索引。若为 None，则设置所有环境。
        """
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        self._data.joint_effort_target[env_ids, joint_ids] = effort

    def set_tendon_len_target(
        self,
        length: torch.Tensor,
        tendon_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """设置腱长度目标。

        Args:
          length: 目标腱长度，形状为 (N, num_tendons)。
          tendon_ids: 可选的要设置的腱索引。若为 None，则设置所有腱。
          env_ids: 可选的环境索引。若为 None，则设置所有环境。
        """
        if env_ids is None:
            env_ids = slice(None)
        if tendon_ids is None:
            tendon_ids = slice(None)
        self._data.tendon_len_target[env_ids, tendon_ids] = length

    def set_tendon_vel_target(
        self,
        velocity: torch.Tensor,
        tendon_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """设置腱速度目标。

        Args:
          velocity: 目标腱速度，形状为 (N, num_tendons)。
          tendon_ids: 可选的要设置的腱索引。若为 None，则设置所有腱。
          env_ids: 可选的环境索引。若为 None，则设置所有环境。
        """
        if env_ids is None:
            env_ids = slice(None)
        if tendon_ids is None:
            tendon_ids = slice(None)
        self._data.tendon_vel_target[env_ids, tendon_ids] = velocity

    def set_tendon_effort_target(
        self,
        effort: torch.Tensor,
        tendon_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """设置腱力矩目标。

        Args:
          effort: 目标腱力矩，形状为 (N, num_tendons)。
          tendon_ids: 可选的要设置的腱索引。若为 None，则设置所有腱。
          env_ids: 可选的环境索引。若为 None，则设置所有环境。
        """
        if env_ids is None:
            env_ids = slice(None)
        if tendon_ids is None:
            tendon_ids = slice(None)
        self._data.tendon_effort_target[env_ids, tendon_ids] = effort

    def set_site_effort_target(
        self,
        effort: torch.Tensor,
        site_ids: torch.Tensor | slice | None = None,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """设置站点力矩目标。

        Args:
          effort: 目标站点力矩，形状为 (N, num_sites)。
          site_ids: 可选的要设置的站点索引。若为 None，则设置所有站点。
          env_ids: 可选的环境索引。若为 None，则设置所有环境。
        """
        if env_ids is None:
            env_ids = slice(None)
        if site_ids is None:
            site_ids = slice(None)
        self._data.site_effort_target[env_ids, site_ids] = effort

    def write_external_wrench_to_sim(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        env_ids: torch.Tensor | slice | None = None,
        body_ids: Sequence[int] | slice | None = None,
    ) -> None:
        """对仿真中的物体施加外部力/力矩。

        底层通过设置 MuJoCo 数据结构中的 ``xfrc_applied`` 字段实现。
        力/力矩在世界坐标系中指定，持续有效直至下次调用此函数或仿真重置。

        Args:
          forces: 形状为 (N, num_bodies, 3) 的张量，N 为环境数量。
          torques: 形状为 (N, num_bodies, 3) 的张量，N 为环境数量。
          env_ids: 可选张量或切片，指定要设置的环境。若为 None，则设置所有环境。
          body_ids: 可选的物体索引列表或切片，指定要施加力/力矩的物体。
            若为 None，则施加到所有物体。
        """
        self._data.write_external_wrench(forces, torques, body_ids, env_ids)

    def write_mocap_pose_to_sim(
        self,
        mocap_pose: torch.Tensor,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """将 mocap 物体的姿态写入仿真。

        Args:
          mocap_pose: 形状为 (N, 7) 的张量，N 为环境数量。
            格式：[pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
          env_ids: 可选张量或切片，指定要设置的环境。若为 None，则设置所有环境。
        """
        self._data.write_mocap_pose(mocap_pose, env_ids)

    ##
    # 私有方法。
    ##

    def _build_default_root_state(self, nworld: int, device: str) -> torch.Tensor:
        """构建默认根状态张量，在所有世界空间中统一。"""
        base = self.cfg.init_state
        components: list[tuple[float, ...]] = [base.pos, base.rot]
        if not self.is_fixed_base:
            components.extend([base.lin_vel, base.ang_vel])
        return torch.tensor(
            sum((tuple(c) for c in components), ()),
            dtype=torch.float,
            device=device,
        ).repeat(nworld, 1)

    def _compute_indexing(self, model: mujoco.MjModel, device: str) -> EntityIndexing:
        bodies = tuple([b for b in self.spec.bodies[1:]])
        joints = self._non_free_joints
        geoms = tuple(self.spec.geoms)
        sites = tuple(self.spec.sites)
        tendons = tuple(self.spec.tendons)
        cameras = tuple(self.spec.cameras)
        lights = tuple(self.spec.lights)
        materials = tuple(self.spec.materials)
        pairs = tuple(self.spec.pairs)

        body_ids = torch.tensor([b.id for b in bodies], dtype=torch.int, device=device)
        geom_ids = torch.tensor([g.id for g in geoms], dtype=torch.int, device=device)
        site_ids = torch.tensor([s.id for s in sites], dtype=torch.int, device=device)
        tendon_ids = torch.tensor(
            [t.id for t in tendons], dtype=torch.int, device=device
        )
        cam_ids = torch.tensor([c.id for c in cameras], dtype=torch.int, device=device)
        light_ids = torch.tensor(
            [lt.id for lt in lights], dtype=torch.int, device=device
        )
        mat_ids = torch.tensor(
            [m.id for m in materials], dtype=torch.int, device=device
        )
        pair_ids = torch.tensor([p.id for p in pairs], dtype=torch.int, device=device)
        joint_ids = torch.tensor([j.id for j in joints], dtype=torch.int, device=device)

        if self.is_actuated:
            actuators = tuple(self.spec.actuators)
            ctrl_ids = torch.tensor(
                [a.id for a in actuators], dtype=torch.int, device=device
            )
        else:
            actuators = None
            ctrl_ids = torch.empty(0, dtype=torch.int, device=device)

        joint_q_adr = []
        joint_v_adr = []
        free_joint_q_adr = []
        free_joint_v_adr = []
        for joint in self.spec.joints:
            jnt = model.joint(joint.name)
            jnt_type = jnt.type[0]
            vadr = jnt.dofadr[0]
            qadr = jnt.qposadr[0]
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                free_joint_v_adr.extend(range(vadr, vadr + 6))
                free_joint_q_adr.extend(range(qadr, qadr + 7))
            else:
                joint_v_adr.extend(range(vadr, vadr + dof_width(jnt_type)))
                joint_q_adr.extend(range(qadr, qadr + qpos_width(jnt_type)))
        joint_q_adr = torch.tensor(joint_q_adr, dtype=torch.int, device=device)
        joint_v_adr = torch.tensor(joint_v_adr, dtype=torch.int, device=device)
        free_joint_v_adr = torch.tensor(
            free_joint_v_adr, dtype=torch.int, device=device
        )
        free_joint_q_adr = torch.tensor(
            free_joint_q_adr, dtype=torch.int, device=device
        )

        if self.is_fixed_base and self.is_mocap:
            mocap_id = int(model.body_mocapid[self.root_body.id])
        else:
            mocap_id = None

        return EntityIndexing(
            bodies=bodies,
            joints=joints,
            geoms=geoms,
            sites=sites,
            tendons=tendons,
            cameras=cameras,
            lights=lights,
            materials=materials,
            pairs=pairs,
            actuators=actuators,
            body_ids=body_ids,
            geom_ids=geom_ids,
            site_ids=site_ids,
            tendon_ids=tendon_ids,
            cam_ids=cam_ids,
            light_ids=light_ids,
            mat_ids=mat_ids,
            pair_ids=pair_ids,
            ctrl_ids=ctrl_ids,
            joint_ids=joint_ids,
            mocap_id=mocap_id,
            joint_q_adr=joint_q_adr,
            joint_v_adr=joint_v_adr,
            free_joint_q_adr=free_joint_q_adr,
            free_joint_v_adr=free_joint_v_adr,
        )

    def _apply_actuator_controls(self) -> None:
        self._builtin_group.apply_controls(self._data)
        for act in self._custom_actuators:
            command = act.get_command(self._data)
            command = act.apply_delay(command)
            self._data.write_ctrl(act.compute(command), act.ctrl_ids)
