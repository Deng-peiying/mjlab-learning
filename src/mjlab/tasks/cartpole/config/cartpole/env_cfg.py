import mujoco
from pathlib import Path

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.cartpole.cartpole_env_cfg import make_cartpole_env_cfg

# MuJoCo 自带 cartpole.xml 的路径
_CARTPOLE_XML = Path(mujoco.__file__).parent / "model" / "cartpole.xml"


def _get_cartpole_spec() -> mujoco.MjSpec:
    """加载 cartpole MjSpec，与 get_yam_spec() 写法一致。"""
    return mujoco.MjSpec.from_file(str(_CARTPOLE_XML))


def get_cartpole_entity_cfg() -> EntityCfg:
    """返回 EntityCfg，与 get_yam_robot_cfg() 写法一致。"""
    return EntityCfg(spec_fn=_get_cartpole_spec)


def cartpole_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_cartpole_env_cfg()

    # 绑定 MuJoCo XML 模型（与 YAM manipulation 完全一样的模式）
    cfg.scene.entities = {
        "cartpole": get_cartpole_entity_cfg(),
    }

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False

    return cfg
