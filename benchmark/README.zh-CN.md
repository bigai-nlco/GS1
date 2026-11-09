# GS1 评测

[English](README.md) · [简体中文](README.zh-CN.md)

在源码仓库中运行 `python -m benchmark`。本地推理读取完整 GS1 artifact；HTTP 模式调用兼容 TypeSafe 的服务。评测器不含训练接入或数据合成。

```bash
uv sync --locked --extra eval --extra cuda
uv run python -m benchmark --model /path/to/GS1-0.8B \
  --device cuda --precision bf16 --suites jevbench --out runs/evaluation/0.8B
```

## 保留的 benchmark

| 参数 | 输入与指标 |
|---|---|
| `jevbench` | 231 个固定公开 Easy／Original／Hard 任务；准确率、期望 Brier、十区间 ECE |
| `agnews` | 原生 7,600 个测试样本；四分类准确率 |
| `toolace` | 640 个固定对话哈希有效样本；首次工具／不调用准确率 |
| `aa_omniscience` | 600 个冻结公开五选项问题；准确率与 NHR |
| `hle_multichoice` | 513 个冻结纯文本多选样本；准确率 |
| `when2call` | 3,423 个冻结样本；回复类别准确率 |
| `wildguard_test` | 856 个冻结样本；三个二分类 F1 均值 |

默认评测 JevBench，从固定上游 revision 下载并校验 SHA-256。AG News、ToolACE 从固定官方源读取，仅建立确定性评测缓存。AA 最终 JSONL 在 Git 外提供，生成／审核／过滤代码均已移除。[AA 口径](aadi/README.zh-CN.md)

```bash
# 不加载模型，仅准备公开评测缓存。
uv run python -m benchmark --download-only --suites jevbench agnews toolace

# 完整常规测试集。
uv run python -m benchmark --model /path/to/GS1-4B --device cuda --precision bf16 \
  --suites jevbench agnews toolace --out runs/evaluation/4B-regular

# 读取外部冻结 AA 输入，不下载、不生成。
uv run python -m benchmark --model /path/to/GS1-2B --device cuda --precision bf16 \
  --suites aa_omniscience hle_multichoice when2call wildguard_test \
  --data-dir /path/to/aadi-v1 --offline --out runs/evaluation/2B-aa

# 评测已启动的服务。
uv run python -m benchmark --api-url http://127.0.0.1:8000 --model GS1 \
  --suites jevbench --out runs/evaluation/api
```

`--dataset SUITE=PATH` 指定 JSONL，`--data-dir` 调整输入／缓存根目录，`--cache-dir` 调整 JevBench 缓存。`--offline` 禁止缺失源下载；`--limit N` 产生明确标记的局部接入测试，不能视为完整成绩。API 凭据通过 `--api-key-env` 与可选 `.env` 读取，本地 GS1 无需凭据。

## 报告与评分

`--out` 输出 `evaluation.json`、`accuracy_manifest.json`，拒绝覆盖已有结果。报告保留数据和选择哈希、模型身份、所有尝试决策、原始请求／响应、计时、错误、准确率分母、Brier 和可靠性区间。失败的准确率记零，概率指标只使用有效且可计分输出并报告数量，不虚构概率或拟合校准。

Choice／Score 使用 argmax，按固定 benchmark 约定处理并列；只有软标签时计算期望正确率。Brier 为 `sum((p-target)^2) + 1 - sum(target^2)`，Noul 使用 `[1-p,p]`。ECE 使用最大类别概率的十个等宽区间，标签不送入模型。本地默认 CPU／FP32／reference，因此 CUDA 测量需要显式指定设备和精度。

退出码：0 为全部有效完成；1 为预测失败；2 为配置、输入或初始化错误。[已发布完整结果](../docs/results.zh-CN.md)包含来源哈希与口径。Decision Index 使用[官方脚本](di/README.zh-CN.md)，仓库不含其 builder 或 scorer。

## 推理时延

当前代码的时延评测比较相同权重、输入和精度下的 HF 默认 `model.decide`、GS1 eager 与经过数值校验的 CUDA Graph：

```bash
uv sync --locked --extra serve --extra eval --extra cuda
CUDA_VISIBLE_DEVICES=0 uv run python -m benchmark.latency \
  --model /path/to/GS1-2B --out runs/latency/GS1-2B.json \
  --contexts 256 1024 --questions 1 8 --warmup 5 --repeats 30
```

计时覆盖进程内完整决策，排除模型加载、图捕获和网络传输。JSON 保留每次调用、P50／P95／P99、实际 token 长度、显存、软件与检查点身份，并确认测量期间发生常驻图回放。[测量方法与完整结果](../docs/latency.zh-CN.md)。
