# GS1 API

[English](api.md) · [简体中文](api.zh-CN.md)

Python 服务提供 `POST /v1/systemone`、`GET /v1/models` 与 FastAPI `/docs`。请求／响应兼容 TypeSafe 协议中的 Choice、Noul、Score 三类。`model.decide` 以 Python 字典返回同一答案定义；`Engine.evaluate` 返回 Pydantic `DecisionResponse`。

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

状态和 instructions 可为字符串、对象或数组。Choice 描述可为字符串、对象、数组或 null，包含 1–255 个不同候选 ID；Noul 可提供 true／false 描述；Score 为包含 2–10 级的有序数组。问题映射不能为空，问题和候选 ID 均予保留，不支持的类型或非法数值明确拒绝。

```bash
curl http://127.0.0.1:8000/v1/systemone \
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

with TypeSafeClient(base_url="http://127.0.0.1:8000", api_key="local", model="GS1") as client:
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

格式错误和超预算输入明确报错。队列满时返回 529 与 `Retry-After`，超时返回 504。调用者取消后不再投递响应，GPU 资源仍安全回收。评测默认零重试并记录失败；应用重试由客户端决定。

Python 服务执行本地 GS1 推理；Node playground 可额外连接 Jev／OpenRouter，凭据仅保留在 Node 服务端，原始供应商响应与用于控制器执行的校验响应分别保存。
