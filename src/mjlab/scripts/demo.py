"""使用预训练策略运行动作跟踪演示。

此演示从云存储下载预训练检查点和运动文件，
并启动交互式查看器，展示人形机器人执行侧手翻动作。
"""

import tyro

import mjlab
from mjlab.scripts.gcs import ensure_default_checkpoint, ensure_default_motion
from mjlab.scripts.play import PlayConfig, run_play


def main() -> None:
    """使用预训练跟踪策略运行演示。"""
    print("🎮 正在设置 mjlab 演示，加载预训练跟踪策略...")

    try:
        checkpoint_path = ensure_default_checkpoint()
        motion_path = ensure_default_motion()
    except RuntimeError as e:
        print(f"❌ 下载演示资源失败: {e}")
        print("请检查网络连接后重试。")
        return

    args = tyro.cli(
        PlayConfig,
        default=PlayConfig(
            checkpoint_file=checkpoint_path,
            motion_file=motion_path,
            num_envs=8,
            viewer="viser",
            _demo_mode=True,
        ),
        config=mjlab.TYRO_FLAGS,
    )
    run_play("Mjlab-Tracking-Flat-Unitree-G1", args)


if __name__ == "__main__":
    main()
