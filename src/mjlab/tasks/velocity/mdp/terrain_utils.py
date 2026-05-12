"""从射线命中点估计地形法线。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import RayCastSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def fit_terrain_normal(
    points: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """通过协方差特征分解从 3D 点中拟合平面法线。

    Args:
      points: [B, N, 3] 世界坐标系位置。
      valid_mask: [B, N] 布尔值（True = 有效）。

    Returns:
      方向朝上的 [B, 3] 单位法线。当有效点少于 3 个时回退到 [0, 0, 1]。
    """
    B = points.shape[0]
    device = points.device

    count = valid_mask.sum(dim=1)
    enough = count >= 3

    mask_f = valid_mask.float().unsqueeze(-1)
    masked_points = points * mask_f
    count_clamped = count.clamp(min=1).float().unsqueeze(-1)
    centroid = masked_points.sum(dim=1) / count_clamped
    centered = (points - centroid.unsqueeze(1)) * mask_f

    cov = torch.einsum("bni,bnj->bij", centered, centered)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    normal = eigenvectors[:, :, 0]  # 最小特征值对应平面法线。
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # 方向朝上。
    normal = torch.where((normal[:, 2] < 0).unsqueeze(-1), -normal, normal)

    # 当拟合退化时（共线或近重复点）回退。
    # 有效平面有一个小特征值和两个明显更大的特征值；
    # 线状或点状的点云不符合此特征。
    eps = torch.finfo(eigenvalues.dtype).eps
    plane_like = (eigenvalues[:, 0] / eigenvalues[:, 1].clamp(min=eps)) < 0.1
    has_spread = eigenvalues[:, 1] > eigenvalues[:, 2].clamp(min=eps) * 1e-6
    reliable = enough & plane_like & has_spread

    up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(B, 3)
    return torch.where(reliable.unsqueeze(-1), normal, up)


# 缓存子采样索引，避免每步重新分配。
_subsample_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def _subsample_indices(
    total: int, max_points: int, device: torch.device
) -> torch.Tensor:
    key = (total, max_points, device)
    if key not in _subsample_cache:
        _subsample_cache[key] = torch.linspace(
            0, total - 1, max_points, device=device
        ).long()
    return _subsample_cache[key]


def terrain_normal_from_sensors(
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...],
    max_points: int = 32,
) -> torch.Tensor:
    """从一个或多个射线传感器估计地形法线。

    收集命中位置，每个传感器子采样到 *max_points* 个点，然后通过
    :func:`fit_terrain_normal` 拟合平面法线。

    Returns:
      [B, 3] 世界坐标系下的单位地形法线。
    """
    all_points: list[torch.Tensor] = []
    all_valid: list[torch.Tensor] = []

    for name in sensor_names:
        sensor = env.scene[name]
        if not isinstance(sensor, RayCastSensor):
            raise TypeError(
                f"Sensor '{name}' is {type(sensor).__name__}, expected RayCastSensor."
            )

        hit_pos = sensor.data.hit_pos_w
        valid = sensor.data.distances >= 0

        N = hit_pos.shape[1]
        if N > max_points:
            idx = _subsample_indices(N, max_points, hit_pos.device)
            hit_pos = hit_pos[:, idx]
            valid = valid[:, idx]

        all_points.append(hit_pos)
        all_valid.append(valid)

    points = torch.cat(all_points, dim=1)
    valid_mask = torch.cat(all_valid, dim=1)
    return fit_terrain_normal(points, valid_mask)
