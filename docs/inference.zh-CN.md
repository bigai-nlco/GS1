# GS1 推理

[English](inference.md) · [简体中文](inference.zh-CN.md)

## 模型默认推理

每个发布目录包含原生模型、tokenizer、processor 与相对 Python 依赖，HF AutoClass 可独立于本仓库加载。`model.decide(state, questions, precision="bf16")` 返回可序列化的响应。在兼容 CPU 环境中使用 `precision="fp32"`。CUDA 算子选择沿用安装的 Qwen 上游运行时；测量环境使用 FLA 0.5.2 和 causal-conv1d 1.7.0。

默认路径为每个问题编译一条完整分支，独立执行；同一问题的候选项一起评分。保留序列化的决策头计算精度，最终概率归约使用 FP32，不生成文本。

## 原生 Engine

```python
from gs1 import Engine

engine = Engine.from_pretrained("bigai/GS1-0.8B", device="cuda", precision="bf16")
try:
    response = engine.evaluate(
        {
            "state": "A refund is requested.",
            "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
        }
    )
    print(response.model_dump())
finally:
    engine.close()
```

加载同时接受 Hub ID 与本地标准 HF 目录，支持 `revision`、`local_files_only`、`cache_dir`、`token`。运行时直接加载 `config.json`、safetensors 和 tokenizer／processor，不再要求 `artifact.json`；准入上限读取 `config.json.inference_limits`。

默认引擎为 `reference`。资格校验允许启用匹配的批处理与算术策略。`shared` 复用不可变状态、全注意力 K/V、GDN 卷积／递归边界，以及各个决策交叉注意力层独立的状态 K/V 投影；需要显式选择并提供匹配资格。共享是否更快取决于请求形状。

## CUDA Graph 推理

`gs1` wheel 只包含推理与本地 API。基准和资格校验在源码目录中通过 `python -m benchmark.inference` 执行，统一使用 `gs1.runtime-profile.v2` 格式；迁移后请重新生成旧 profile。绑定身份现为根据实际模型和输入编码文件计算的 `checkpoint_id`；旧 artifact／predictor 身份的 profile 也需重新生成。

```bash
# 源码仓库中的 benchmark 工具不包含在 gs1 wheel 中。
uv sync --locked --extra serve --extra cuda --extra eval

# Qualify the installed implementation, precision and request geometries.
uv run python -m benchmark.inference --model /path/to/GS1-2B \
  --out runs/gs1-2b-runtime.json \
  --runtime-profile-out runs/gs1-2b-profile.json \
  --graph-cache-mb 16384 --graph-cache-entries 128

# The benchmark also writes runs/gs1-2b-profile-warmup.json.
uv run gs1 serve --model /path/to/GS1-2B --model-name GS1 \
  --runtime-profile runs/gs1-2b-profile.json \
  --experimental-cuda-graphs --graph-cache-mb 4096 --graph-cache-entries 128 \
  --graph-warmup-requests runs/gs1-2b-profile-warmup.json
```

Profile 绑定检查点、源码、硬件、精度和测试过的形状。输出的 warmup 文件可直接用于 `--graph-warmup-requests`。默认 `prewarm` 策略在服务启动时捕获图，再重放已知形状。图缓存未命中时采用原生 eager，未被校验的请求形状采用单分支 reference。每个服务槽拥有独立 stream、缓冲区和图缓存。

Python 中可向 `Engine.from_pretrained` 传入 `runtime_profile="runs/gs1-2b-profile.json"`、`cuda_graphs=True`、`graph_cache_mb=4096`、`graph_cache_entries=128`。首次匹配调用可能捕获图，热延迟测量应排除这部分；部署优先使用服务预热入口。

Profile 是本地执行校验记录，修改源码、检查点、精度或运行环境后需重建；不作为模型权重上传 HF。FP32 校验采用 logits 绝对／相对误差 `1e-5 / 1e-4`、概率 TV `1e-4`；有效 BF16 路径采用 `2e-3 / 1e-2`、TV `5e-3`。图重放必须与对应 eager 计算逐位相同。

## 预算与测量

编译器在执行前检查完整分支上下文和逻辑 token 总量，并约束问题、分支、估算注意力工作、logits、batch、缓存、队列和图显存。超预算明确报错，不静默截断状态或删减候选项。

计时应区分加载／捕获、编译、引擎执行、排队和 HTTP 格式化。[已发布延迟结果](results.zh-CN.md#延迟与显存)写明各自边界。短输入通常受益于减少 kernel launch 开销，长形状可能超出图常驻预算并回退到 eager。
