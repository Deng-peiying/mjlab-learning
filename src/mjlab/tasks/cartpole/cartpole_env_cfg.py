from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import observations as mdp_obs
from mjlab.envs.mdp import rewards as mdp_rewards
from mjlab.envs.mdp import terminations as mdp_term
from mjlab.envs.mdp import events as mdp_events
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab.tasks.cartpole import mdp as cartpole_mdp


def make_cartpole_env_cfg() -> ManagerBasedRlEnvCfg:
    """CartPole 倒立摆任务基础配置。"""

    # ---------- 观测 ----------
    # cartpole.xml 有两个关节：slider（cart 位置）和 hinge（pole 角度）
    # joint_pos_rel 返回 shape (num_envs, 2)，joint_vel_rel 同理
    # 合计观测维度：4（与 CartPole_PPO 的 state 一致）
    asset_cfg = SceneEntityCfg("cartpole")

    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp_obs.joint_pos_rel,
            params={"asset_cfg": asset_cfg},
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp_obs.joint_vel_rel,
            params={"asset_cfg": asset_cfg},
        ),
    }
    critic_terms = {**actor_terms}

    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=False),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    # ---------- 动作 ----------
    # cartpole.xml 的 actuator 名：cart_force（控制 slider）
    actions: dict[str, ActionTermCfg] = {
        "cart_force": JointPositionActionCfg(
            entity_name="cartpole",
            actuator_names=("cart_force",),
            scale=1.0,
            use_default_offset=True,
        )
    }

    # ---------- 事件（重置） ----------
    events = {
        "reset_cartpole": EventTermCfg(
            func=mdp_events.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.05, 0.05),
                "velocity_range": (-0.05, 0.05),
                "asset_cfg": asset_cfg,
            },
        ),
    }

    # ---------- 奖励 ----------
    rewards = {
        # 存活奖励：每步 +1（框架会乘以 dt=0.02，等效为时间积分）
        "alive": RewardTermCfg(
            func=mdp_rewards.is_alive,
            weight=1.0,
        ),
        # 自定义：pole 越竖直奖励越大（Gaussian 形式）
        "upright": RewardTermCfg(
            func=cartpole_mdp.pole_upright,
            weight=2.0,
            params={"asset_cfg": asset_cfg},
        ),
        # 惩罚：cart 跑太远
        "cart_pos_penalty": RewardTermCfg(
            func=cartpole_mdp.cart_position_penalty,
            weight=-0.1,
            params={"asset_cfg": asset_cfg},
        ),
        # 平滑动作
        "action_rate": RewardTermCfg(
            func=mdp_rewards.action_rate_l2,
            weight=-0.01,
        ),
    }

    # ---------- 终止条件 ----------
    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp_term.time_out,
            time_out=True,
        ),
        # pole 倾斜超过 ~20°（比 CartPole-v0 的 12° 宽松一些，方便学习）
        "pole_fallen": TerminationTermCfg(
            func=mdp_term.bad_orientation,
            params={
                "limit_angle": 0.35,  # ~20 度
                "asset_cfg": asset_cfg,
            },
        ),
    }

    # ---------- 场景 ----------
    scene = SceneCfg(
        num_envs=512,
        env_spacing=2.0,
    )

    # ---------- 仿真参数 ----------
    # SimulationCfg 不含 dt/decimation，这两个字段在 ManagerBasedRlEnvCfg 里
    sim = SimulationCfg(
        mujoco=MujocoCfg(
            timestep=0.01,    # cartpole 动力学较快，10ms 物理步长
            integrator="euler",
        ),
    )

    return ManagerBasedRlEnvCfg(
        scene=scene,
        sim=sim,
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        episode_length_s=10.0,
        decimation=2,         # 控制频率 = 10ms * 2 = 20ms，与 CartPole_PPO 一致
        viewer=ViewerConfig(),
    )
