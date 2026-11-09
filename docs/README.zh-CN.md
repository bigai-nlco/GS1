# GS1 文档

[English](README.md) · [简体中文](README.zh-CN.md)

从[仓库快速开始](../README.zh-CN.md)进入，再选择对应指南。

| 指南 | 内容 |
|---|---|
| [模型](model.zh-CN.md) | 架构、参数结构、输出语义与检查点精度 |
| [推理](inference.zh-CN.md) | 独立 HF 决策、原生 Engine、资格校验与 CUDA Graph |
| [API](api.zh-CN.md) | 请求、概率、置信度及错误 |
| [结果（英文）](../benchmark/results.md) | 完整公开成绩、常规 benchmark、校准与延迟 |
| [推理时延（英文）](../benchmark/latency.md) | 当前发布代码的 HF／eager／CUDA Graph 时延、原始样本与复现 |
| [评测](../benchmark/README.zh-CN.md) | 复现保留的核心公开评测 |
| [官方 Decision Index](../benchmark/di/README.zh-CN.md) | 使用上游脚本评测 GS1 HTTP 服务 |
| [Playground](../playground/README.zh-CN.md) | Query、Go、Maze、Snake |

`gs1` wheel 只包含模型推理（`model/`）、优化执行（`inference/`）和本地服务（`api/`）。评测与资格生成位于 `benchmark/`，多卡启动器位于 `scripts/serve_replicas.py`，与 Node playground 一样从源码目录运行。权重和评测输入位于 Git 外。
