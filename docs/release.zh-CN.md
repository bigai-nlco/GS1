# 模型发布

[English](release.md) · [简体中文](release.zh-CN.md)

最终每个目录都是独立完整的 Hub 仓库：`GS1-0.8B`、`GS1-2B`、`GS1-4B`，目标为 [GS1 集合](https://huggingface.co/collections/bigai/gs1)下的 `bigai/GS1-<size>`。9B 在检查点提供后按相同规范补齐。

发布目录包含：声明 `model_type: gs1`、`architectures: [GS1Model]`、AutoConfig／AutoModel `auto_map`、结构 token ID 和 `inference_limits` 的配置；原样 safetensors 和分片索引；tokenizer／chat-template／processor；`modular_gs1.py` 及其同目录相对依赖；`requirements.txt`；中英 README；许可与必要小型图片。

HF 默认推理使用 `AutoModel.from_pretrained(..., trust_remote_code=True)` 与 `model.decide(...)`，无需安装 GS1 仓库。优化路径安装 GS1，对相同原生权重执行本地运行时资格校验。各模型 README 均给出两种调用。

发布前确认：

- 权重文件 SHA-256 与源检查点相同，参数名、形状、dtype 保持不变。
- 元数据仅包含公开模型／推理信息；架构、结构 token ID 和推理上限统一放在 `config.json`，无需独立 artifact manifest。原始权重哈希和来源信息保留在上传目录外的发布验证记录中。
- 在无法导入 `gs1` 的进程中 AutoClass 仍可用，三类决策均返回合法分布。
- 原生加载、保存重载与 API 决策符合声明的数值策略。
- 本地 GPU profile、凭据、训练状态、评测输入和内部路径不进入上传。

将单个最终目录内容作为对应 HF 仓库根目录上传。原 ZIP 保留为恢复来源。上传和可见性调整属于单独发布动作；整理后的本地目录在上传前即可运行。主仓库仅包含代码与公开汇总结果，不包含模型权重。
