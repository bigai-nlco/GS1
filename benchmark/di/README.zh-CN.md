# 使用官方 Decision Index 评测

[English](README.md) · [简体中文](README.zh-CN.md)

使用[上游 Decision Index 复现工具](https://github.com/apolinario/decision-index)。此目录仅保留文档；数据构建、关联样本计分、随机校正与汇总由官方项目维护。

1. 按完整公开评测使用的 32k 服务配置启动 GS1：

   ```bash
   uv run gs1 serve --model /path/to/GS1-2B --model-name GS1 \
     --max-context 32768 --batch-tokens 32768 \
     --max-attention-bytes 549755813888 --deadline-seconds 120
   ```

   attention 上限是准入工作量估算；实际 GPU 分配取决于输入形状与 attention 实现。

2. 单独安装上游工具并准备公开数据，按上游要求取得受限来源访问。
3. 用其 HTTP 引擎评测 GS1，保留环境、逐 benchmark 和汇总文件。

```bash
# Install the official public-suite reproduction kit in its own environment.
pip install 'git+https://github.com/apolinario/decision-index.git@9eb2dbe2a358004c8782c66e40a83ac07b953fec'

python -m decision_index pipeline --engine http \
  --option base_url=http://127.0.0.1:8000 --option model=GS1 \
  --edition 0.3 --out runs/gs1-public-index
```

固定 revision 对应已发布本地成绩记录的历史公共套件谱系。选择更新 revision 或 edition 前查阅上游，并随结果记录。榜单 0.4／0.4.1 与公共套件 0.3 是不同版本标识，不能直接替换为未经上游支持的 `--edition 0.4.1`。

公共分数与私有 Full 榜单分数不同。[完整结果（英文）](../results.md)给出本地完整面板、五领域图、37 项明细和来源哈希。抽样或局部面板不能复现完整分数；超限、失败和重试均需报告，不丢弃样本。
