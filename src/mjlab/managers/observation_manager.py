"""用于计算观测值的观测管理器。"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase, ManagerTermBaseCfg
from mjlab.utils.buffers import CircularBuffer, DelayBuffer
from mjlab.utils.noise import noise_cfg, noise_model
from mjlab.utils.noise.noise_cfg import NoiseCfg, NoiseModelCfg


@dataclass
class ObservationTermCfg(ManagerTermBaseCfg):
    """观测项的配置。

    处理流水线：计算 → 噪声 → 裁剪 → 缩放 → 延迟 → 历史记录。
    延迟模拟传感器延迟。历史记录提供时序上下文。两者均为可选项且可组合使用。
    """

    noise: NoiseCfg | NoiseModelCfg | None = None
    """应用于观测值的噪声模型。"""

    clip: tuple[float, float] | None = None
    """裁剪观测值的范围 (min, max)。"""

    scale: tuple[float, ...] | float | torch.Tensor | None = None
    """乘以观测值的缩放因子。"""

    delay_min_lag: int = 0
    """延迟观测的最小延迟步数。延迟从 [min_lag, max_lag] 均匀采样。
  转换为毫秒：lag * (1000 / control_hz)。"""

    delay_max_lag: int = 0
    """延迟观测的最大延迟步数。设置 min=max 可实现恒定延迟。"""

    delay_per_env: bool = True
    """如果为 True，每个环境独立采样自己的延迟。如果为 False，所有环境在每步
  共享相同的延迟。"""

    delay_hold_prob: float = 0.0
    """重用上一个延迟而不重新采样的概率。对于时间相关的延迟模式很有用。"""

    delay_update_period: int = 0
    """每 N 步重新采样延迟（模拟多速率传感器）。如果为 0，则每步更新。"""

    delay_per_env_phase: bool = True
    """如果为 True 且 update_period > 0，则在各环境之间错开更新时机，
  避免同步重采样。"""

    history_length: int = 0
    """保留在历史记录中的过去观测数量。0 = 无历史记录。"""

    flatten_history_dim: bool = True
    """是否将历史维度展平到观测中。

  当为 True 且 concatenate_terms=True 时，使用 term-major 排序：
  [A_t0, A_t1, ..., A_tH-1, B_t0, B_t1, ..., B_tH-1, ...]
  详见 docs/source/observation.rst。"""


@dataclass
class ObservationGroupCfg:
    """观测组的配置。

    观测组将多个观测项捆绑在一起。组通常用于为不同用途分离观测
    （例如 "actor" 用于策略网络，"critic" 用于价值函数）。
    """

    terms: dict[str, ObservationTermCfg]
    """将项名称映射到其配置的字典。"""

    concatenate_terms: bool = True
    """是否将所有项连接为单个张量。如果为 False，返回将项名称
  映射到各自张量的字典。"""

    concatenate_dim: int = -1
    """连接项的维度。默认 -1（最后一维）。"""

    enable_corruption: bool = False
    """是否向观测值施加噪声干扰。训练时设为 True 用于域随机化，
  评估时设为 False。"""

    history_length: int | None = None
    """组级别的历史长度覆盖。如果设置，应用于该组中的所有项。
  如果为 None，每个项使用自己的 ``history_length`` 设置。"""

    flatten_history_dim: bool = True
    """是否将历史展平到观测维度。如果为 True，观测形状为
  ``(num_envs, obs_dim * history_length)``。如果为 False，
  形状为 ``(num_envs, history_length, obs_dim)``。"""

    nan_policy: Literal["disabled", "warn", "sanitize", "error"] = "disabled"
    """该组观测值的 NaN/Inf 处理策略。

  - 'disabled': 不检查（默认，最快）
  - 'warn': 记录警告，包含项名称和环境 ID，然后进行清理（调试用）
  - 'sanitize': 静默清理为 0.0，类似奖励管理器（生产环境安全）
  - 'error': 遇到 NaN/Inf 抛出 ValueError（严格开发模式）
  """

    nan_check_per_term: bool = True
    """如果为 True，逐个检查每个观测项以确定 NaN 来源。
  如果为 False，仅检查最终拼接后的输出（更快但信息较少）。
  仅在 nan_policy != 'disabled' 时生效。"""


class ObservationManager(ManagerBase):
    """管理环境的观测计算。

    观测管理器从组织到组中的多个项计算观测值。每个项可以应用
    噪声、裁剪、缩放、延迟和历史记录。组可以选择将其项连接为单个张量。
    """

    def __init__(self, cfg: dict[str, ObservationGroupCfg], env):
        self.cfg = deepcopy(cfg)
        super().__init__(env=env)

        self._group_obs_dim: dict[str, tuple[int, ...] | list[tuple[int, ...]]] = dict()

        for group_name, group_term_dims in self._group_obs_term_dim.items():
            if self._group_obs_concatenate[group_name]:
                term_dims = torch.stack(
                    [torch.tensor(dims, device="cpu") for dims in group_term_dims],
                    dim=0,
                )
                if len(term_dims.shape) > 1:
                    if self._group_obs_concatenate_dim[group_name] >= 0:
                        dim = self._group_obs_concatenate_dim[group_name] - 1
                    else:
                        dim = self._group_obs_concatenate_dim[group_name]
                    dim_sum = torch.sum(term_dims[:, dim], dim=0)
                    term_dims[0, dim] = dim_sum
                    term_dims = term_dims[0]
                else:
                    term_dims = torch.sum(term_dims, dim=0)
                self._group_obs_dim[group_name] = tuple(term_dims.tolist())
            else:
                self._group_obs_dim[group_name] = group_term_dims

        self._obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None = (
            None
        )

    def __str__(self) -> str:
        msg = (
            f"<ObservationManager> contains {len(self._group_obs_term_names)} groups.\n"
        )
        for group_name, group_dim in self._group_obs_dim.items():
            table = PrettyTable()
            table.title = f"Active Observation Terms in Group: '{group_name}'"
            if self._group_obs_concatenate[group_name]:
                table.title += f" (shape: {group_dim})"  # type: ignore
            table.field_names = ["Index", "Name", "Shape"]
            table.align["Name"] = "l"
            obs_terms = zip(
                self._group_obs_term_names[group_name],
                self._group_obs_term_dim[group_name],
                self._group_obs_term_cfgs[group_name],
                strict=False,
            )
            for index, (name, dims, term_cfg) in enumerate(obs_terms):
                if term_cfg.history_length > 0 and term_cfg.flatten_history_dim:
                    # 展平的历史记录：显示为 (9,) ← 3×(3,)
                    original_size = int(np.prod(dims)) // term_cfg.history_length
                    original_shape = (original_size,) if len(dims) == 1 else dims[1:]
                    shape_str = f"{dims}  ← {term_cfg.history_length}×{original_shape}"
                else:
                    shape_str = str(tuple(dims))
                table.add_row([index, name, shape_str])
            msg += table.get_string()
            msg += "\n"
        return msg

    def get_active_iterable_terms(
        self, env_idx: int
    ) -> Sequence[tuple[str, Sequence[float]]]:
        terms = []

        if self._obs_buffer is None:
            self.compute()
        assert self._obs_buffer is not None
        obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = self._obs_buffer

        for group_name, _ in self.group_obs_dim.items():
            if not self.group_obs_concatenate[group_name]:
                buffers = obs_buffer[group_name]
                assert isinstance(buffers, dict)
                for name, term in buffers.items():
                    terms.append(
                        (group_name + "-" + name, term[env_idx].cpu().tolist())
                    )  # type: ignore[unsupported-operator]
                continue

            idx = 0
            data = obs_buffer[group_name]
            assert isinstance(data, torch.Tensor)
            for name, shape in zip(
                self._group_obs_term_names[group_name],
                self._group_obs_term_dim[group_name],
                strict=False,
            ):
                data_length = np.prod(shape)
                term = data[env_idx, idx : idx + data_length]
                terms.append((group_name + "-" + name, term.cpu().tolist()))
                idx += data_length

        return terms

    # Properties.

    @property
    def active_terms(self) -> dict[str, list[str]]:
        return self._group_obs_term_names

    @property
    def group_obs_dim(self) -> dict[str, tuple[int, ...] | list[tuple[int, ...]]]:
        return self._group_obs_dim

    @property
    def group_obs_term_dim(self) -> dict[str, list[tuple[int, ...]]]:
        return self._group_obs_term_dim

    @property
    def group_obs_concatenate(self) -> dict[str, bool]:
        return self._group_obs_concatenate

    # Methods.

    def get_term_cfg(self, group_name: str, term_name: str) -> ObservationTermCfg:
        if group_name not in self._group_obs_term_names:
            raise ValueError(f"Group '{group_name}' not found in active groups.")
        if term_name not in self._group_obs_term_names[group_name]:
            raise ValueError(f"Term '{term_name}' not found in group '{group_name}'.")
        index = self._group_obs_term_names[group_name].index(term_name)
        return self._group_obs_term_cfgs[group_name][index]

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
        # Invalidate cache since reset envs will have different observations.
        self._obs_buffer = None

        for group_name, group_cfg in self._group_obs_class_term_cfgs.items():
            for term_cfg in group_cfg:
                term_cfg.func.reset(env_ids=env_ids)
            for term_name in self._group_obs_term_names[group_name]:
                batch_ids = None if isinstance(env_ids, slice) else env_ids
                if term_name in self._group_obs_term_delay_buffer[group_name]:
                    self._group_obs_term_delay_buffer[group_name][term_name].reset(
                        batch_ids=batch_ids
                    )
                if term_name in self._group_obs_term_history_buffer[group_name]:
                    self._group_obs_term_history_buffer[group_name][term_name].reset(
                        batch_ids=batch_ids
                    )
        for group_mods in self._group_obs_class_instances.values():
            for mod in group_mods.values():
                mod.reset(env_ids=env_ids)
        return {}

    def _check_and_handle_nans(
        self, tensor: torch.Tensor, context: str, policy: str
    ) -> torch.Tensor:
        """检查 NaN/Inf 并根据策略处理。

        Args:
          tensor: 待检查的观测张量。
          context: 用于错误/警告消息的上下文字符串（例如 "actor/base_lin_vel"）。
          policy: NaN 处理策略（"disabled"、"warn"、"sanitize"、"error"）。

        Returns:
          根据策略可能已清理的张量。

        Raises:
          ValueError: 如果策略为 "error" 且检测到 NaN/Inf。
        """
        if policy == "disabled":
            return tensor

        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()

        if not (has_nan or has_inf):
            return tensor

        if policy == "error":
            nan_mask = torch.isnan(tensor).any(dim=-1) | torch.isinf(tensor).any(dim=-1)
            nan_env_ids = torch.where(nan_mask)[0].cpu().tolist()
            raise ValueError(
                f"NaN/Inf detected in observation '{context}' "
                f"for environments: {nan_env_ids[:10]}"
            )

        if policy == "warn":
            nan_mask = torch.isnan(tensor).any(dim=-1) | torch.isinf(tensor).any(dim=-1)
            nan_env_ids = torch.where(nan_mask)[0].cpu().tolist()
            print(
                f"[ObservationManager] NaN/Inf in '{context}' "
                f"(envs: {nan_env_ids[:5]}). Sanitizing to 0."
            )

        # Sanitize (applies to both "warn" and "sanitize" policies).
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def compute(
        self, update_history: bool = False
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        # 如果不更新且缓存存在，则返回缓存观测值。
        # 这可以防止 compute() 在每个控制步中被多次调用时
        # 向延迟缓冲区重复推送数据（例如，在 step() 之后调用 get_observations()）。
        if not update_history and self._obs_buffer is not None:
            return self._obs_buffer

        obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = dict()
        for group_name in self._group_obs_term_names:
            obs_buffer[group_name] = self.compute_group(group_name, update_history)
        self._obs_buffer = obs_buffer
        return obs_buffer

    def compute_group(
        self, group_name: str, update_history: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        group_cfg = self.cfg[group_name]
        group_term_names = self._group_obs_term_names[group_name]
        group_obs: dict[str, torch.Tensor] = {}
        obs_terms = zip(
            group_term_names, self._group_obs_term_cfgs[group_name], strict=False
        )
        for term_name, term_cfg in obs_terms:
            obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()
            if isinstance(term_cfg.noise, noise_cfg.NoiseCfg):
                obs = term_cfg.noise.apply(obs)
            elif isinstance(term_cfg.noise, noise_cfg.NoiseModelCfg):
                obs = self._group_obs_class_instances[group_name][term_name](obs)
            if term_cfg.clip:
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
            if term_cfg.scale is not None:
                scale = term_cfg.scale
                assert isinstance(scale, torch.Tensor)
                obs = obs.mul_(scale)

            # 在延迟/历史缓冲区之前检查 NaN/Inf（逐项检查）。
            if group_cfg.nan_check_per_term and group_cfg.nan_policy != "disabled":
                obs = self._check_and_handle_nans(
                    obs,
                    context=f"{group_name}/{term_name}",
                    policy=group_cfg.nan_policy,
                )

            if term_cfg.delay_max_lag > 0:
                delay_buffer = self._group_obs_term_delay_buffer[group_name][term_name]
                delay_buffer.append(obs)
                obs = delay_buffer.compute()
            if term_cfg.history_length > 0:
                circular_buffer = self._group_obs_term_history_buffer[group_name][
                    term_name
                ]
                if update_history or not circular_buffer.is_initialized:
                    circular_buffer.append(obs)

                if term_cfg.flatten_history_dim:
                    group_obs[term_name] = circular_buffer.buffer.reshape(
                        self._env.num_envs, -1
                    )
                else:
                    group_obs[term_name] = circular_buffer.buffer
            else:
                group_obs[term_name] = obs

        # 非逐项检查模式下的最终 NaN 检查。
        if not group_cfg.nan_check_per_term and group_cfg.nan_policy != "disabled":
            if self._group_obs_concatenate[group_name]:
                # Will check after concatenation below.
                pass
            else:
                for term_name in group_obs:
                    group_obs[term_name] = self._check_and_handle_nans(
                        group_obs[term_name],
                        context=f"{group_name}/{term_name}",
                        policy=group_cfg.nan_policy,
                    )

        if self._group_obs_concatenate[group_name]:
            result = torch.cat(
                list(group_obs.values()),
                dim=self._group_obs_concatenate_dim[group_name],
            )
            # 对拼接结果进行最终检查（非逐项检查模式）。
            if not group_cfg.nan_check_per_term and group_cfg.nan_policy != "disabled":
                result = self._check_and_handle_nans(
                    result, context=group_name, policy=group_cfg.nan_policy
                )
            return result
        return group_obs

    def _prepare_terms(self) -> None:
        self._group_obs_term_names: dict[str, list[str]] = dict()
        self._group_obs_term_dim: dict[str, list[tuple[int, ...]]] = dict()
        self._group_obs_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_class_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
        self._group_obs_concatenate: dict[str, bool] = dict()
        self._group_obs_concatenate_dim: dict[str, int] = dict()
        self._group_obs_class_instances: dict[
            str, dict[str, noise_model.NoiseModel]
        ] = {}
        self._group_obs_term_delay_buffer: dict[str, dict[str, DelayBuffer]] = dict()
        self._group_obs_term_history_buffer: dict[str, dict[str, CircularBuffer]] = (
            dict()
        )

        for group_name, group_cfg in self.cfg.items():
            group_cfg: ObservationGroupCfg | None
            if group_cfg is None:
                print(f"group: {group_name} set to None, skipping...")
                continue

            if not any(t is not None for t in group_cfg.terms.values()):
                print(f"group: {group_name} has no active terms, skipping...")
                continue

            self._group_obs_term_names[group_name] = list()
            self._group_obs_term_dim[group_name] = list()
            self._group_obs_term_cfgs[group_name] = list()
            self._group_obs_class_term_cfgs[group_name] = list()
            self._group_obs_class_instances[group_name] = {}
            group_entry_delay_buffer: dict[str, DelayBuffer] = dict()
            group_entry_history_buffer: dict[str, CircularBuffer] = dict()

            self._group_obs_concatenate[group_name] = group_cfg.concatenate_terms
            self._group_obs_concatenate_dim[group_name] = (
                group_cfg.concatenate_dim + 1
                if group_cfg.concatenate_dim >= 0
                else group_cfg.concatenate_dim
            )

            for term_name, term_cfg in group_cfg.terms.items():
                term_cfg: ObservationTermCfg | None
                if term_cfg is None:
                    print(f"term: {term_name} set to None, skipping...")
                    continue

                # 注意：这个深拷贝很重要，可避免项配置的跨组污染。
                term_cfg = deepcopy(term_cfg)
                self._resolve_common_term_cfg(term_name, term_cfg)

                if not group_cfg.enable_corruption:
                    term_cfg.noise = None
                if group_cfg.history_length is not None:
                    term_cfg.history_length = group_cfg.history_length
                    term_cfg.flatten_history_dim = group_cfg.flatten_history_dim
                self._group_obs_term_names[group_name].append(term_name)
                self._group_obs_term_cfgs[group_name].append(term_cfg)
                if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
                    self._group_obs_class_term_cfgs[group_name].append(term_cfg)

                obs_dims = tuple(term_cfg.func(self._env, **term_cfg.params).shape)

                if term_cfg.scale is not None:
                    term_cfg.scale = torch.tensor(
                        term_cfg.scale, dtype=torch.float, device=self._env.device
                    )

                if term_cfg.noise is not None and isinstance(
                    term_cfg.noise, noise_cfg.NoiseModelCfg
                ):
                    noise_model_cls = term_cfg.noise.class_type
                    assert issubclass(noise_model_cls, noise_model.NoiseModel), (
                        f"Class type for observation term '{term_name}' NoiseModelCfg"
                        f" is not a subclass of 'NoiseModel'. Received: '{type(noise_model_cls)}'."
                    )
                    self._group_obs_class_instances[group_name][term_name] = (
                        noise_model_cls(
                            term_cfg.noise,
                            num_envs=self._env.num_envs,
                            device=self._env.device,
                        )
                    )

                if term_cfg.delay_max_lag > 0:
                    group_entry_delay_buffer[term_name] = DelayBuffer(
                        min_lag=term_cfg.delay_min_lag,
                        max_lag=term_cfg.delay_max_lag,
                        batch_size=self._env.num_envs,
                        device=self._env.device,
                        per_env=term_cfg.delay_per_env,
                        hold_prob=term_cfg.delay_hold_prob,
                        update_period=term_cfg.delay_update_period,
                        per_env_phase=term_cfg.delay_per_env_phase,
                    )

                if term_cfg.history_length > 0:
                    group_entry_history_buffer[term_name] = CircularBuffer(
                        max_len=term_cfg.history_length,
                        batch_size=self._env.num_envs,
                        device=self._env.device,
                    )
                    old_dims = list(obs_dims)
                    old_dims.insert(1, term_cfg.history_length)
                    obs_dims = tuple(old_dims)
                    if term_cfg.flatten_history_dim:
                        obs_dims = (obs_dims[0], int(np.prod(obs_dims[1:])))

                self._group_obs_term_dim[group_name].append(obs_dims[1:])

            self._group_obs_term_delay_buffer[group_name] = group_entry_delay_buffer
            self._group_obs_term_history_buffer[group_name] = group_entry_history_buffer
