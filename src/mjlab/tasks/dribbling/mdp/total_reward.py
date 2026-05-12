"""Multiplicative total reward: r_t = r_pos * exp(r_neg).

Aligned with DribbleBot paper: "The total reward at timestep t is represented as
rt = r_pos_t * exp(r_neg_t), where r_pos_t and r_neg_t represent positive task
reward and negative penalizing reward respectively."  (Appendix, Table III)

Also provides joint_vel_l1 which the paper uses for the |q̇| penalty term.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.reward_manager import RewardTermCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# ---------------------------------------------------------------------------
# Joint Velocity L1 (paper uses |q̇|, not |q̇|²)
# ---------------------------------------------------------------------------

def joint_vel_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint velocities using L1 norm.  Paper Table III: |q̇|, weight -0.0001."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint accelerations using L1 norm.  Paper Table III: |q̈|, weight -2.5e-7."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_pos_limits_binary(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint limit violations using binary indicator per joint.

    Paper Table III: 1_{qi>qmax||qi<qmin}, weight -10.0.
    Returns the COUNT of joints that exceed soft limits.
    """
    asset: Entity = env.scene[asset_cfg.name]
    soft_limits = asset.data.soft_joint_pos_limits
    assert soft_limits is not None
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lower = soft_limits[:, asset_cfg.joint_ids, 0]
    upper = soft_limits[:, asset_cfg.joint_ids, 1]
    violation = ((joint_pos < lower) | (joint_pos > upper)).float()
    return torch.sum(violation, dim=1)


# ---------------------------------------------------------------------------
# Multiplicative reward combiner
# ---------------------------------------------------------------------------

# Registry: name -> function.  Populated by make_dribbling_env_cfg() before
# DribblingTotalReward is instantiated.
_FUNC_REGISTRY: dict[str, object] = {}


def register_reward_func(name: str, func: object) -> None:
    """Register a reward/penalty function by name for the multiplicative combiner."""
    _FUNC_REGISTRY[name] = func


class DribblingTotalReward:
    """Combines pos/neg rewards multiplicatively: r_pos * exp(r_neg).

    Instantiated by RewardManager (detected as a class because it has ``reset``).
    Sub-reward configs live in ``cfg.params["pos_terms"]`` and
    ``cfg.params["neg_terms"]`` as lists of ``{"func": str, "weight": float,
    "params": dict}``.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._env = env

        pos_cfgs: list[dict] = cfg.params.get("pos_terms", [])
        neg_cfgs: list[dict] = cfg.params.get("neg_terms", [])

        self._pos: list[tuple[object, float, dict]] = []
        self._neg: list[tuple[object, float, dict]] = []

        for c in pos_cfgs:
            func = _FUNC_REGISTRY[c["func"]]
            params = c.get("params", {})
            self._resolve_scene_cfgs(params, env)
            self._pos.append((c["name"], func, c["weight"], params))

        for c in neg_cfgs:
            func = _FUNC_REGISTRY[c["func"]]
            params = c.get("params", {})
            self._resolve_scene_cfgs(params, env)
            self._neg.append((c["name"], func, c["weight"], params))

    @staticmethod
    def _resolve_scene_cfgs(params: dict, env: ManagerBasedRlEnv) -> None:
        for v in params.values():
            if isinstance(v, SceneEntityCfg):
                v.resolve(env.scene)

    def __call__(self, env: ManagerBasedRlEnv, **__) -> torch.Tensor:
        r_pos = torch.zeros(env.num_envs, device=env.device)
        r_neg = torch.zeros(env.num_envs, device=env.device)

        for name, func, weight, params in self._pos:
            value = func(env, **params)
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            r_pos += value * weight
            env.extras["log"][f"Reward/{name}"] = torch.mean(value * weight)

        for name, func, weight, params in self._neg:
            value = func(env, **params)
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            r_neg += value * weight  # weight is negative, r_neg accumulates ≤ 0
            env.extras["log"][f"Reward/{name}"] = torch.mean(value * weight)

        total = r_pos * torch.exp(r_neg)  # paper eq. rt = r_pos * exp(r_neg)

        env.extras["log"]["Reward/r_pos_sum"] = torch.mean(r_pos)
        env.extras["log"]["Reward/r_neg_sum"] = torch.mean(r_neg)

        return total

    def reset(self, env_ids: torch.Tensor) -> None:
        del env_ids  # stateless
