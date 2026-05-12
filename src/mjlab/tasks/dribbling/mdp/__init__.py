"""dribbling 任务的 MDP 命名空间。

将以下模块的函数/clas导入，使 `from mjlab.tasks.dribbling import mdp`
可以直接访问所有 MDP 组件：

导入来源：
  mjlab.envs.mdp          — 框架通用函数（sensor、reward、termination、domain rand）
  .ball_command           — BallVelocityCommand / BallVelocityCommandCfg
  .ball_reward            — 5 个球相关奖励函数
  .ball_events            — 球传送 + drag 事件函数
  .total_reward           — 乘法奖励组合器 + joint_vel_l1
  .curriculums            — 地形课程学习
  .observations           — ball_pos_b / ball_vel_w / gait_timing_ref / body_yaw
  .rewards                — upright / self_collision_cost / 动作/能耗惩罚
  .terminations           — illegal_contact / out_of_terrain_bounds
  .gait                   — update_gait_clock
  .gait_reward            — swing_phase / stance_phase
"""

from mjlab.envs.mdp import *  # noqa: F401, F403

from .ball_command import *  # noqa: F403
from .ball_reward import *  # noqa: F403
from .ball_events import *  # noqa: F403
from .total_reward import *  # noqa: F403
from .curriculums import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
from .terminations import *  # noqa: F403
from .gait import *  # noqa: F403
from .gait_reward import *  # noqa: F403
