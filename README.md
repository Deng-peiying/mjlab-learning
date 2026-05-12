![项目横幅](https://raw.githubusercontent.com/mujocolab/mjlab/main/docs/source/_static/mjlab-banner.jpg)

# mjlab

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/mujocolab/mjlab/ci.yml?branch=main)](https://github.com/mujocolab/mjlab/actions/workflows/ci.yml?query=branch%3Amain)
[![文档](https://github.com/mujocolab/mjlab/actions/workflows/docs.yml/badge.svg)](https://mujocolab.github.io/mjlab/)
[![许可证](https://img.shields.io/github/license/mujocolab/mjlab)](https://github.com/mujocolab/mjlab/blob/main/LICENSE)
[![夜间基准测试](https://img.shields.io/badge/夜间-基准测试-blue)](https://mujocolab.github.io/mjlab/nightly/)
[![PyPI](https://img.shields.io/pypi/v/mjlab)](https://pypi.org/project/mjlab/)
[![PyPI 下载量](https://img.shields.io/pypi/dm/mjlab?color=blue)](https://pypistats.org/packages/mjlab)

mjlab 将 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 的基于管理器的 API 与 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) 相结合，后者是 [MuJoCo](https://github.com/google-deepmind/mujoco) 的 GPU 加速版本。
该框架提供了可组合的构建模块用于环境设计，
依赖项极少，并可直接访问原生 MuJoCo 数据结构。

## 快速开始

mjlab 训练需要 NVIDIA GPU。macOS 仅支持评估模式。

**立即体验：**

运行演示（无需安装）：

```bash
uvx --from mjlab --refresh demo
```

或在 [Google Colab](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb) 中试用（无需本地配置）。

**从源码安装：**

```bash
git clone https://github.com/mujocolab/mjlab.git && cd mjlab
uv run demo
```

其他安装方式（PyPI、Docker）请参阅[安装指南](https://mujocolab.github.io/mjlab/main/source/installation.html)。

## 训练示例

### 1. 速度跟踪

训练 Unitree G1 人形机器人在平坦地形上跟随速度指令：

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

**多 GPU 训练：** 使用 `--gpu-ids` 扩展到多个 GPU：

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 4096
```

详情请参阅[分布式训练指南](https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html)。

训练过程中评估策略（从 Weights & Biases 获取最新检查点）：

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 2. 动作模仿

训练人形机器人模仿参考动作。预处理设置请参阅[动作模仿指南](https://mujocolab.github.io/mjlab/main/source/training/motion_imitation.html)。

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name your-org/motions/motion-name --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 3. 使用虚拟代理进行完整性检查

使用内置代理在训练前对 MDP 进行完整性检查：

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # 发送零动作
uv run play Mjlab-Your-Task-Id --agent random  # 发送均匀随机动作
```

运行动作跟踪任务时，请在命令中添加 `--registry-name your-org/motions/motion-name`。


## 文档

完整文档请访问 **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**。

## 开发

```bash
make test          # 运行所有测试
make test-fast     # 跳过慢速测试
make format        # 格式化和代码检查
make docs          # 本地构建文档
```

开发环境设置：`uvx pre-commit install`

## 引用

mjlab 已被用于已发表的研究和开源机器人项目。请参阅[研究](https://mujocolab.github.io/mjlab/main/source/research.html)页面查看出版物和项目，或在[展示与分享](https://github.com/mujocolab/mjlab/discussions/categories/show-and-tell)中分享您的成果。

如果您在研究中使用 mjlab，请考虑引用：

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## 许可证

mjlab 基于 [Apache License, Version 2.0](LICENSE) 许可。

### 第三方代码

mjlab 的部分代码派生自外部项目：

- **`src/mjlab/utils/lab_api/`** — 派生自 [NVIDIA Isaac
  Lab](https://github.com/isaac-sim/IsaacLab) 的工具代码（BSD-3-Clause 许可证，详见文件头）

派生组件保留其原始许可证。详情请参阅文件头。

## 致谢

mjlab 的诞生离不开 Isaac Lab 团队的出色工作，mjlab 在其 API 设计和抽象基础之上构建。

感谢 MuJoCo Warp 团队——特别是 Erik Frey 和 Taylor Howell——无数次回答我们的问题、提供有益的反馈并根据我们的需求实现功能。
