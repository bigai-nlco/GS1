# AA v1 公共组件评测

[English](README.md) · [简体中文](README.zh-CN.md)

这里保留历史 AA v1 的四个冻结公共组件。[当前官方方法](https://artificialanalysis.ai/methodology/decision-index)已经演进，本地结果不作为当前官方指数成绩。

| 组件 | 冻结样本 | 指标 |
|---|---:|---|
| AA-Omniscience public | 600 | 准确率；`abstained/(abstained+wrong)` NHR |
| HLE 多选 | 513 | 答案准确率 |
| When2Call | 3,423 | 回复类别准确率 |
| WildGuard | 856 | prompt-harm、response-harm、refusal 三项 F1 均值 |

输入位于外部 `aadi-v1` 目录，布局为 `<suite>/test.jsonl`。[`inputs.py`](inputs.py)固定来源、revision、最终样本数与 SHA-256。独立数据包不进入 Python wheel 或 Git。按[评测指南](../README.zh-CN.md)指定四个 suite 和 `--data-dir /path/to/aadi-v1 --offline`。

Omniscience 已包含三个冻结干扰项、正确答案和 `I don't know`，评测不再调用模型生成或审核；使用 600 个公共问题，与官方 500 个私有问题不同。WildGuard 本地冻结集从 880 条一致标签中排除 24 条，和官方 835 条样本不同；来源 ID 与 provenance 仍固定。HLE 和 When2Call 同样保留历史冻结选择。

本地 v1 汇总为 `100*(0.125*AA_accuracy + 0.125*AA_NHR + 0.25*HLE_accuracy + 0.25*When2Call_accuracy + 0.25*WildGuard_mean_F1)`，`official_comparable` 保持 false。缺失或修改文件在加载模型前报错，不静默生成替代数据。显式覆盖输入作为独立本地来源记录。
