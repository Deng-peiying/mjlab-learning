"""Unitree Go1 运球任务的 PPO 训练配置。

对齐 DribbleBot 论文 Table V 的超参数设置。

训练规模：
  - 并行环境数：4096
  - 每 rollout 步数：21（每个环境）
  - 每 iteration 样本：4096 * 21 = 86,016
  - 总 iterations：81,000
  - 总 timesteps：~7B（论文值）

网络架构：
  - Actor：[512, 256, 128] ELU，Gaussian 分布输出
  - Critic：[512, 256, 128] ELU
  - 观测归一化：关闭（obs_normalization=False）
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_go1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """创建对齐论文 Table V 的 PPO runner 配置。"""
    return RslRlOnPolicyRunnerCfg(
        # ---- Actor 网络 ----
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,          # 无条件归一化
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,               # 初始探索噪声
                "std_type": "scalar",          # 所有动作共享一个 std
            },
        ),
        # ---- Critic 网络 ----
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
        ),
        # ---- PPO 算法参数（Table V） ----
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,               # 价值损失权重
            use_clipped_value_loss=True,       # 价值函数裁切
            clip_param=0.2,                    # PPO clip 范围
            entropy_coef=0.01,                 # 熵 bonus α2
            num_learning_epochs=5,             # 每 rollout epoch 数
            num_mini_batches=4,                # minibatch 数
            learning_rate=1.0e-3,              # 学习率
            schedule="adaptive",               # 自适应学习率调度
            gamma=0.99,                        # 折扣因子
            lam=0.95,                          # GAE λ
            desired_kl=0.01,                   # 目标 KL 散度
            max_grad_norm=1.0,                 # 梯度裁剪
        ),
        # ---- 运行配置 ----
        experiment_name="go1_dribbling",
        save_interval=50,                       # 每 50 iterations 保存一次
        num_steps_per_env=21,                   # 每环境 rollout 步数
        max_iterations=81_000,                  # 总训练 iterations（~7B steps）
    )
