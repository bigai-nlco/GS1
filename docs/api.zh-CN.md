# GS1 API

[English](api.md) · [简体中文](api.zh-CN.md)

Python 服务提供 `POST /v1/systemone`、`POST /v1/admission`、`GET /v1/runtime`、`GET /v1/models` 与 FastAPI `/docs`。请求／响应兼容 TypeSafe 协议中的 Choice、Noul、Score 三类。`model.decide` 以 Python 字典返回同一答案定义；`Engine.evaluate` 返回 Pydantic `DecisionResponse`。

## 单卡服务默认配置

```bash
# 将 0 替换为所选 GPU；每个服务进程使用一张卡。
CUDA_VISIBLE_DEVICES=0 uv run gs1 serve \
  --model /data/shared/GS1/GS1-9B --model-name GS1-9B
```

默认监听 `0.0.0.0:9000`，局域网 SDK 的 base URL 为 `http://<服务器IP>:9000`。默认配置不固定 GPU 编号或多卡拓扑，由用户选择显卡。多卡可通过 `scripts/serve_replicas.py --gpus ...` 显式选择副本。

| 设置 | 默认值 |
|---|---|
| 引擎／精度 | `reference`／`bf16` |
| context／batch token 上限 | 16384／16384，同时受模型上限约束 |
| 问题数／请求逻辑 token 数／分支数 | 512／131072／5120，同时受模型上限约束 |
| GPU 显存预留 | 1024 MiB |
| 前缀缓存／可选 BF16 权重副本缓存 | 0 MiB／0 MiB |
| 队列／超时／CPU 线程 | 32／120 秒／2 |
| 图执行槽数／图缓存／条目数／捕获策略 | 1／2048 MiB／16／`prewarm` |

每次推理前释放分配器中可回收的空闲块，注意力及临时张量预算取**驱动报告的实际空闲显存减去预留额度**，再受 `--max-attention-bytes` 和模型上限约束。模型权重和常驻图池已从实际空闲显存中扣除。配置上限默认为 512 GiB，它是限制值，不代表预分配或保证可用。用户显式选择多个并发图执行槽时，剩余临时显存预算在槽之间均分。CPU 服务使用配置与模型的上限。模型 buffer 变化后权重副本缓存仍遵守额度，原始权重精度保持不变。

CUDA Graphs 需提供本机资格与预热文件：`--runtime-profile ... --experimental-cuda-graphs --graph-warmup-requests ...`。资格文件绑定源码和运行环境，更新后需要重新生成。非零 `--compute-weight-cache-mb` 必须与资格文件中验证过的 `deployment.compute_weight_cache_mb` 一致。

## 请求

```json
{
  "model": "GS1",
  "state": "A customer was charged twice and requests a refund.",
  "questions": {
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "technical": "Software"}},
    "refund": {"type": "noul", "instructions": "Is a refund requested?"},
    "urgency": {"type": "score", "instructions": "Rate urgency.", "criteria": ["Routine", "Urgent", "Emergency"]}
  }
}
```

状态和 instructions 可为字符串、对象或数组。Choice 描述可为字符串、对象、数组或 null，包含 1–256 个不同候选 ID；Noul 可提供 true／false 描述；Score 为包含 2–256 级的有序数组。问题映射不能为空，问题和候选 ID 均予保留，不支持的类型或非法数值明确拒绝。

```bash
curl http://127.0.0.1:9000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request-with-model.json
```

也可在[示例请求](../examples/request.json)中补入 `"model": "GS1"`；服务别名由 `--model-name` 指定。

## TypeSafe SDK

启动本地服务后，可通过[官方 Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)直接调用：

```bash
python -m pip install typesafe-sdk==0.7.0
```

```python
from typesafe_sdk import Choice, TypeSafeClient

with TypeSafeClient(base_url="http://127.0.0.1:9000", api_key="local", model="GS1") as client:
    result = client.system_one(
        state="A customer was charged twice and requests a refund.",
        questions={
            "team": Choice(
                instructions="Which team should handle this ticket?",
                criteria={"billing": "Payments and refunds", "technical": "Software problems"},
            )
        },
    )
print(result.choices["team"].choice)
print(result.choices["team"].probabilities)
```

`base_url` 为 API 根地址，不附加 `/v1/systemone`；`model` 与 `--model-name` 一致。`api_key="local"` 仅满足 SDK 的必填项，本地服务不要求 TypeSafe 云端密钥。三类结果分别通过 `result.choices`、`result.nouls`、`result.scores` 读取；完整分布、置信度与 token 使用量予以保留。[完整可运行示例](../examples/typesafe_client.py)。

## 答案

| 类型 | 返回字段 |
|---|---|
| Choice | `choice`、完整 `probabilities`、归一化 `confidence` |
| Noul | `noul = p(true)` |
| Score | 从零开始的等级期望 `score`、完整等级概率、`legend`、`confidence` |

响应包含 `model`、`answers`、`usage`（`input_tokens`、`output_tokens`）。本地 GS1 的生成输出 token 数为零。概率必须有限、范围合法且完整，不为缺失选项虚构概率。类型正确不等于事实判断正确。

## 置信度

设 K 个候选的概率为 p：

- Choice：`(max(p) - 1/K) / (1 - 1/K)`；单候选置信度为 1。
- Noul 公开值为 `p(true)`，可选二元确定性指标为 `abs(2*p(true)-1)`。
- Score：`max(0, 1 - spread/even_spread)`；`spread = sum(p[i]*abs(i-m))`，m 为首个最可能等级；`even_spread = sum(abs(i-(K-1)/2))/K`。

Score 分布 `[0.25, 0.5, 0.25]` 的分数为 1、置信度为 0.25。公开置信度与 ECE 使用的最大类别概率不同。阈值应在目标工作负载上评估，评测器不在测试集上拟合校准参数。

## 错误与客户端行为

格式错误和超预算输入明确报错。`POST /v1/admission` 只编译并估算请求，不执行推理。容量不足返回 422，`error.details` 包含实际分支 token 数、估算显存、当前上限与请求哈希；准入诊断不包含提示词内容。`GET /v1/runtime` 返回就绪状态、显存预算、已验证形状数和常驻图统计。成功响应包含 `X-GS1-Graph-Hits`、`X-GS1-Graph-Misses`、`X-GS1-Profile-Fallback`。

队列满时返回 529 与 `Retry-After`，超时返回 504。调用者取消后不再投递响应，GPU 资源仍安全回收。评测默认零重试并记录失败；应用重试由客户端决定。

Python 服务执行本地 GS1 推理；Node playground 可通过 `TYPESAFE_API_KEY` 调用 TypeSafe 官方 Jev API，凭据仅保留在 Node 服务端，原始供应商响应与经校验的响应分别保存。[Playground 配置](../playground/README.zh-CN.md#通过-typesafe-官方-api-使用-jev)。
