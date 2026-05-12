"""Trot gait clock: generates desired contact states for Swing/Stance Phase rewards.

Simplified from Walk These Ways (Margolis & Agrawal, CoRL 2022) and DribbleBot.
Uses a fixed 2 Hz trot gait: FL+RR in phase, FR+RL opposite.
"""

import torch


def update_gait_clock(env, env_ids) -> None:
    """每步更新 gait clock，生成期望接触状态并存到 env.gait_state。

    作为 mode="step" event 被 EventManager 调用。
    对角线 trot：FL+RR 同相（0°），FR+RL 同相（180°）。
    """
    del env_ids  # step 模式全局更新

    freq = 2.0       # 步态频率 [Hz]
    duration = 0.5   # 支撑相占比（0~0.5=支撑, 0.5~1=摆动）
    kappa = 0.07     # von Mises 平滑系数

    device = env.device
    num_envs = env.num_envs

    # 每条腿的基础相位 [FL, FR, RL, RR]  (trot: 对角配对)
    base_phase = torch.tensor([0.0, 0.5, 0.5, 0.0], device=device)

    # 时间 → 全局相位
    time = env.episode_length_buf.float() * env.step_dt  # [B]
    global_phase = (time * freq) % 1.0  # [B]

    # 每条腿的独立相位
    phase = (global_phase.unsqueeze(1) + base_phase.unsqueeze(0)) % 1.0  # [B, 4]

    # Von Mises 平滑：用正态 CDF 做软切换，避免 sharp 的 0/1 切换
    from torch.distributions.normal import Normal as NormalDist
    cdf = NormalDist(0, kappa).cdf

    def smooth(x):
        return cdf(x) * (1.0 - cdf(x - 0.5)) + cdf(x - 1.0) * (1.0 - cdf(x - 0.5 - 1.0))

    desired_contact = smooth(phase)  # [B, 4], 0=摆动 ~ 1=支撑

    # 首次调用时在 env 上创建 gait_state 属性（reward 函数可能在 step event 前被调用）
    if not hasattr(env, "gait_state"):
        env.gait_state = {}

    env.gait_state["desired_contact"] = desired_contact
    env.gait_state["gait_phase"] = phase

    # Timing reference θ_cmd: sin/cos of each trot pair phase (Walk These Ways, CoRL 2022).
    # Pair 0 (FL+RR, phase=0.0) and Pair 1 (FR+RL, phase=0.5).
    pair0_phase = phase[:, 0]  # FL
    pair1_phase = phase[:, 1]  # FR
    env.gait_state["timing_ref"] = torch.stack([
        torch.sin(2 * torch.pi * pair0_phase),
        torch.cos(2 * torch.pi * pair0_phase),
        torch.sin(2 * torch.pi * pair1_phase),
        torch.cos(2 * torch.pi * pair1_phase),
    ], dim=-1)  # [B, 4]
    # Also store yaw (heading) for observation
    env.gait_state["global_phase"] = global_phase  # [B]
