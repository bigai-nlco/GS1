# 参与 GS1 开发

[English](CONTRIBUTING.md) · [简体中文](CONTRIBUTING.zh-CN.md)

修改聚焦模型推理、服务、playground 和可复现评测。`modular_gs1.py` 与独立 HF 发布共享；除非明确进行模型版本变更，应保留参数名、精度策略与输出语义。

```bash
uv sync --locked --extra serve --extra eval --extra dev
uv run pytest -q
uv run ruff check gs1 benchmark tests scripts examples
uv run ruff format --check gs1 benchmark tests scripts examples
cd playground
npm ci
npm run build
npm run test:server
npm run test:e2e
```

GPU 修改还需要相对单分支 reference 的当前资格校验，以及图／eager 逐位一致性检查。记录硬件、源码／权重身份、精度、形状、计时边界和显存，并如实标注回退。

完整权重、数据、密钥、缓存和构建产物不进入 Git。面向用户的 Markdown 同时维护英文与简体中文，命令和定量表格保持一致。保留第三方许可。Issue／PR 提供最小复现请求与版本信息。
