# GS1 部署

[English](deployment.md) · [简体中文](deployment.zh-CN.md)

## 单 GPU

```bash
uv run gs1 serve --model /path/to/GS1-0.8B --model-name GS1 \
  --device cuda --precision bf16 --host 127.0.0.1 --port 8000
```

模型参数支持本地发布目录或已上传 Hub ID。服务启动前校验 manifest 身份、文件、tokenizer 和结构 token，无需初始化时的原始模型目录，保留检查点存储精度。

## 图推理与并发

按[图资格校验流程](inference.zh-CN.md#cuda-graph-推理)生成 profile 和预热列表，并提供足够的图缓存显存与条目。`--graph-slots 1`、`2`、`4` 创建独立执行槽，多槽要求 prewarm；每个槽拥有独立 stream、可变状态、图缓冲区和有界缓存份额。图显存与队列并发是不同参数。

`--max-queue` 默认 32，`--deadline-seconds` 默认 30，覆盖包含排队的整个请求生命周期。取消请求后释放逻辑所有权，但不会提前复用仍在被 GPU 使用的缓冲区。

## 副本监督器

`scripts/serve_replicas.py` 为选定的 GPU 各启动一个服务，检查就绪状态并在预算内重启；每个子进程拥有独立队列和截止时间：

```bash
uv run python scripts/serve_replicas.py --model /path/to/GS1-0.8B --model-name GS1 --gpus 0,1 --base-port 18080
```

上述服务分别监听 `127.0.0.1:18080` 和 `127.0.0.1:18081`。监督器输出就绪端点，并在 `supervisor.json` 记录状态；可由现有入口代理分发请求。使用当前进程可见的 GPU 编号。[副本扩展实测](results.zh-CN.md#延迟与显存)采用独立副本，而非张量并行。

## 大规模公共评测请求

完整公共评测使用 32k 上下文和较大的注意力准入上限。通过官方 HTTP 评测器复现时可使用：

```bash
uv run gs1 serve --model /path/to/GS1-2B --model-name GS1 \
  --max-context 32768 --batch-tokens 32768 \
  --max-attention-bytes 549755813888 --deadline-seconds 120
```

注意力参数是准入工作量估计，并不预分配等量显存；实际峰值取决于输入形状与 SDPA 实现。通常服务默认更严格。未通过形状校验的长请求采用 reference 回退，所有失败均保留可见。

开发默认监听本地地址；公开部署使用现有入口代理控制。源码仓库中的 benchmark 将热态执行与启动、图捕获分别测量。
