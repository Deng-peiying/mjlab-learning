"""用于在训练期间记录自定义逐步指标的指标管理器。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase, ManagerTermBaseCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MetricsTermCfg(ManagerTermBaseCfg):
    """指标项（term）配置。

    Attributes:
      per_substep: 如果为 True，则在 decimation 循环内每个物理子步评估一次此项，
      并报告逐步均值。在循环中，只有积分状态（qpos、qvel、act）是最新的；
      所有派生量（xpos、xquat、site_xpos、actuator_force、contacts 等）是滞后的。
      reduce: 如何将逐步值聚合为 episode 指标。
      ``"mean"``（默认）报告 ``sum / step_count``。``"last"`` 报告
      episode 最后一步的值，适用于不应在时间步上进行平均的二值成功指标。
    """

    per_substep: bool = False
    reduce: Literal["mean", "last"] = "mean"


class MetricsManager(ManagerBase):
    """累加逐步指标值，报告 episode 平均值。

    与奖励不同，指标没有权重、没有 dt 缩放，也不按 episode 长度归一化。
    Episode 值是真正的逐步平均值（sum / step_count），因此一个 [0,1] 范围内
    的指标在日志记录器中仍保持在 [0,1] 范围内。
    """

    _env: ManagerBasedRlEnv

    def __init__(self, cfg: dict[str, MetricsTermCfg], env: ManagerBasedRlEnv):
        self._term_names: list[str] = list()
        self._term_cfgs: list[MetricsTermCfg] = list()
        self._class_term_cfgs: list[MetricsTermCfg] = list()
        self._step_term_indices: list[int] = list()
        self._substep_term_indices: list[int] = list()

        self.cfg = deepcopy(cfg)
        super().__init__(env=env)

        self._episode_sums: dict[str, torch.Tensor] = {}
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )
        # 为子步项预解析张量引用，避免在热循环中进行字典查找。
        self._substep_accum: list[torch.Tensor] = []
        self._substep_episode_sums: list[torch.Tensor] = []
        for idx in self._substep_term_indices:
            name = self._term_names[idx]
            buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._substep_accum.append(buf)
            self._substep_episode_sums.append(self._episode_sums[name])
        self._substep_count: int = 0
        self._step_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._step_values = torch.zeros(
            (self.num_envs, len(self._term_names)),
            dtype=torch.float,
            device=self.device,
        )

    def __str__(self) -> str:
        msg = f"<MetricsManager> contains {len(self._term_names)} active terms.\n"
        table = PrettyTable()
        table.title = "Active Metrics Terms"
        table.field_names = ["Index", "Name"]
        table.align["Name"] = "l"
        for index, name in enumerate(self._term_names):
            table.add_row([index, name])
        msg += table.get_string()
        msg += "\n"
        return msg

    # 属性。

    @property
    def active_terms(self) -> list[str]:
        return self._term_names

    # 方法。

    def reset(
        self, env_ids: torch.Tensor | slice | None = None
    ) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        extras = {}
        counts = self._step_count[env_ids].float()
        # 避免对尚未步进的环境除以零。
        safe_counts = torch.clamp(counts, min=1.0)
        for idx, key in enumerate(self._episode_sums):
            if self._term_cfgs[idx].reduce == "last":
                extras["Episode_Metrics/" + key] = torch.mean(
                    self._step_values[env_ids, idx]
                )
            else:
                extras["Episode_Metrics/" + key] = torch.mean(
                    self._episode_sums[key][env_ids] / safe_counts
                )
            self._episode_sums[key][env_ids] = 0.0
        self._step_count[env_ids] = 0
        for buf in self._substep_accum:
            buf[env_ids] = 0.0
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        return extras

    def compute_substep(self) -> None:
        """在 decimation 循环内累加逐子步的指标值。

        如果没有配置 ``per_substep`` 项，则为空操作。
        """
        if not self._substep_term_indices:
            return
        for i, idx in enumerate(self._substep_term_indices):
            value = self._compute_term(idx)
            self._substep_accum[i] += value
        self._substep_count += 1

    def compute(self) -> None:
        self._step_count += 1
        if self._substep_term_indices and self._substep_count > 0:
            for i, idx in enumerate(self._substep_term_indices):
                avg = self._substep_accum[i] / self._substep_count
                self._substep_episode_sums[i] += avg
                self._step_values[:, idx] = avg
                self._substep_accum[i].zero_()
            self._substep_count = 0
        for idx in self._step_term_indices:
            name = self._term_names[idx]
            value = self._compute_term(idx)
            self._episode_sums[name] += value
            self._step_values[:, idx] = value

    def get_active_iterable_terms(
        self, env_idx: int
    ) -> Sequence[tuple[str, Sequence[float]]]:
        terms = []
        for idx, name in enumerate(self._term_names):
            terms.append((name, [self._step_values[env_idx, idx].cpu().item()]))
        return terms

    def _prepare_terms(self):
        for term_name, term_cfg in self.cfg.items():
            term_cfg: MetricsTermCfg | None
            if term_cfg is None:
                print(f"term: {term_name} set to None, skipping...")
                continue
            self._resolve_common_term_cfg(term_name, term_cfg)
            idx = len(self._term_names)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            if term_cfg.per_substep:
                self._substep_term_indices.append(idx)
            else:
                self._step_term_indices.append(idx)
            if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
                self._class_term_cfgs.append(term_cfg)

    def _compute_term(self, idx: int) -> torch.Tensor:
        name = self._term_names[idx]
        term_cfg = self._term_cfgs[idx]
        value = term_cfg.func(self._env, **term_cfg.params)
        self._check_term_shape(name, value)
        return value


class NullMetricsManager:
    """缺失指标管理器时的占位符，安全地将所有操作作为空操作处理。"""

    def __init__(self):
        self.active_terms: list[str] = []
        self.cfg = None

    def __str__(self) -> str:
        return "<NullMetricsManager> (inactive)"

    def __repr__(self) -> str:
        return "NullMetricsManager()"

    def get_active_iterable_terms(
        self, env_idx: int
    ) -> Sequence[tuple[str, Sequence[float]]]:
        return []

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
        return {}

    def compute_substep(self) -> None:
        pass

    def compute(self) -> None:
        pass
