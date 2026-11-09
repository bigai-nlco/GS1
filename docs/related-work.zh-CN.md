# 相关项目与发布设计

[English](related-work.md) · [简体中文](related-work.zh-CN.md)

## Jev 与类型化决策

Jev 的目标是让软件快速获取可执行的判断：状态与运行时定义的问题直接映射为类型化结果和概率。主要优势是结果能直接接入程序、候选范围明确，决策任务省去文本解码和解析。官方也强调校准和并行评估；其他实现仍需独立测量这些性质。[TypeSafe 官方介绍](https://docs.typesafe.ai/introduction)、[发布博客](https://typesafe.ai/blog/introducing-system-one-models-and-jev)。

GS1 开放 Choice、Noul、Score、原生候选条件 logits 和完整检查点文件。置信度公式明确，[Brier/ECE 与 AA 组件结果](results.zh-CN.md)展示实际表现。当前 Jev 文档还列出 Box；GS1 的公开接口覆盖已实现的三种原语。概率输出和有限候选范围本身不能证明正确性或校准程度。

开源生态提供了不同的实现路线。[SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev)从既有模型读取选项概率；[AnyJev](https://github.com/nokia-applied-research/AnyJev)提供通用类型化决策适配。GS1 同时发布原生决策头参数、模型定义和推理引擎。这些路线的实现选择不同，接口兼容不代表架构相同或分数等价。

## 架构与推理项目的发布经验

以下设计取舍是我们对所链接项目公开材料的归纳，不代表某种展示方式能保证传播效果。

| 一手项目 | 有价值的展示方式 | GS1 中的对应设计 |
|---|---|---|
| [Mamba](https://github.com/state-spaces/mamba) | 模型说明、最小可运行接口、预训练模型链接 | 保留 canonical modular 模型，精简架构说明，独立 HF 入口 |
| [BitNet](https://github.com/microsoft/BitNet) | 推理实现配合明确的速度与显存测量 | 默认与优化路径分别说明，标注计时边界、分位数和内存 |
| [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | 统一模型家族、权重入口、技术证据与明确许可 | 单张型号表、分型号模型卡、精确检查点身份和引用 |
| [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) | 小而可读的推理核心及直接使用示例 | `model/`、`inference/`、`api/`，有界图缓存和可执行快速开始 |
| [JevBench](https://github.com/fstandhartinger/jevbench) | 固定评测契约下的准确率、校准与运行指标 | 锁定公开任务、保留完整分母和 Brier/ECE，公开任务与密封成绩分开 |

## 发布顺序

首页先呈现清晰能力、已验证的核心分数、简短可运行示例和架构图。完整成绩一跳可达，较弱组件也保留展示。每份指南和模型卡提供英文与简体中文。

先发布完整分型号权重目录，验证本地与 Hub 两种加载，再宣布模型可用。合集、GitHub、模型卡和 playground 使用一致模型 ID。真实模型演示与可复现的结果文件可直接支撑发布内容。按评测维护者的当前流程提交结果；发布公告和榜单提交是独立的对外操作。

后续测量发生变化时，同时更新结果 JSON 并重绘图表。保留 benchmark revision、精度、覆盖率和实际执行模式，使不同版本能公平比较。
