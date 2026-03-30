"""CartPole 自定义奖励函数（mjlab 向量化接口）。

cartpole.xml 关节顺序：
  0 - slider（cart 位置，单位 m）
  1 - hinge （pole 角度，单位 rad，0 = 竖直向上）

所有函数签名须符合 mjlab 规范：
  func(env: ManagerBasedRlEnv, ...) -> torch.Tensor  shape (num_envs,)
"""
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("cartpole")


def pole_upright(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for keeping the pole upright.

    Uses Gaussian decay: 1.0 when perfectly vertical, ~0 when |angle| > 0.5 rad.

    Returns:
        shape (num_envs,), values in (0, 1].
    """
    asset: Entity = env.scene[asset_cfg.name]
    pole_angle = asset.data.joint_pos[:, 1]  # hinge joint, rad
    return torch.exp(-torch.square(pole_angle) / (0.25**2))


def cart_position_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize cart moving far from center.

    Returns:
        shape (num_envs,), values >= 0 (used with negative weight in cfg).
    """
    asset: Entity = env.scene[asset_cfg.name]
    cart_pos = asset.data.joint_pos[:, 0]  # slider joint, m
    return torch.square(cart_pos)
