from mjlab.rl.config import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def cartpole_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    actor_cfg = RslRlModelCfg(
        hidden_dims=(64, 64),
        activation="relu",
        obs_normalization=False,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )
    critic_cfg = RslRlModelCfg(
        hidden_dims=(64, 64),
        activation="relu",
        obs_normalization=False,
    )
    algorithm_cfg = RslRlPpoAlgorithmCfg(
        clip_param=0.1,
        entropy_coef=0.01,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=5.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
    )
    return RslRlOnPolicyRunnerCfg(
        actor=actor_cfg,
        critic=critic_cfg,
        algorithm=algorithm_cfg,
        num_steps_per_env=200,
        max_iterations=5000,
        save_interval=100,
        experiment_name="cartpole_ppo",
        run_name="",
        resume=False,
        logger="tensorboard",
        upload_model=False,
    )
