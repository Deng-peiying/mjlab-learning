"""运球任务专用 RL Runner。

继承 MjlabOnPolicyRunner，在保存 checkpoint 时额外导出 ONNX 模型和 wandb 日志。
"""

import wandb
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner


class DribblingOnPolicyRunner(MjlabOnPolicyRunner):
    """运球任务 Runner：支持 ONNX 导出和 wandb 模型上传。

    继承 MjlabOnPolicyRunner 的标准训练循环（PPO rollout + 更新），
    在 save() 时额外执行：
      1. 导出策略网络为 ONNX 格式
      2. 附加 metadata（任务名、wandb run 信息）
      3. 上传 ONNX 模型到 wandb
    """

    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None):
        """保存 checkpoint + 导出 ONNX + 上传 wandb。"""
        super().save(path, infos)
        policy_dir, filename, onnx_path = self._get_export_paths(path)
        try:
            self.export_policy_to_onnx(str(policy_dir), filename)
            run_name: str = (
                wandb.run.name
                if self.logger.logger_type == "wandb" and wandb.run
                else "local"
            )
            metadata = get_base_metadata(self.env.unwrapped, run_name)
            attach_metadata_to_onnx(str(onnx_path), metadata)
            if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
                wandb.save(str(onnx_path), base_path=str(policy_dir))
        except Exception as e:
            print(f"[WARN] ONNX export failed (training continues): {e}")
