# GS1 Playground

[English](README.md) · [简体中文](README.zh-CN.md)

The React client and Node server provide a **Query** workspace for typed decisions. GS1 is selected by default. Jev is available through the official TypeSafe API for comparison; no cloud key is needed for local GS1.

![GS1 Playground](../assets/playground.png)

## Local GS1

```bash
# Repository root, terminal 1:
uv run gs1 serve --model /path/to/GS1-0.8B --model-name GS1

# Terminal 2:
cd playground
npm ci
npm run build
npm start
```

Open [localhost:3000](http://127.0.0.1:3000). `backends.yaml` points `gs1_local` at port 8000 and model alias `GS1`. Change its endpoint/model to match a deployed service. The service must expose `GET /v1/models` and `POST /v1/systemone`. The actual model identity comes from the configured endpoint, not a hard-coded size label.

## Optional Jev via TypeSafe

Get an API key from the [TypeSafe console](https://console.typesafe.ai). After installing dependencies and building as above, run from `playground/`:

```bash
export TYPESAFE_API_KEY="your-typesafe-api-key"
npm start -- --backends-config backends_jev.yaml --default-backend jev
```

Open [localhost:3000](http://127.0.0.1:3000) to use Jev without starting a local model server. The configured model is `jev-latest`; Node sends requests directly to `https://api.typesafe.ai/v1/systemone` with Bearer authentication, following the [official quick start](https://docs.typesafe.ai/introduction/quickstart). The key stays on the Node server.

To offer both GS1 and Jev, export the same variable and run `npm start` with the default catalog, then choose **Jev · TypeSafe** in the model selector.

## Development

```bash
# Separate terminals in playground/:
npm run dev:server
npm run dev
```

Vite on port 5173 proxies the Node API on port 3000. `npm start -- --host 127.0.0.1 --port 3000 --backends-config backends.yaml --default-backend gs1_local` sets the server options. Rebuild after frontend changes.

## Query

Edit state and mixed Choice/Noul/Score questions in split editors, or supply a full request. Text and structured JSON round-trip intact. Examples cover multiple choice, ordinal scoring and resume screening.

Use **Run query** or Cmd/Ctrl+Enter to evaluate the request; **Cancel** aborts a pending call. The model selector switches between local GS1 and the official TypeSafe Jev API. Overview shows each primitive's probabilities and confidence; JSON preserves the raw response.

Latency measures the Node server's upstream round trip plus validation, excluding browser rendering. Raw Jev probabilities and their audited rounding normalization remain separate from the validated response.

## Validation

```bash
npm run typecheck
npm run test:server
npm run build
npm run test:e2e
```

Browser tests use Query fixtures. Optional live Jev checks require a configured `TYPESAFE_API_KEY` and `GS1_LIVE=1`. Source stays in this folder, generated `dist/`, logs and browser artifacts stay ignored.
