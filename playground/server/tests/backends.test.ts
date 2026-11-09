import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { RemoteBackend } from '../backends.js';
import { SnakeGame } from '../snake.js';
import { normalizeResponse, candidateIds, makeAnswer, HttpError } from '../protocol.js';
import { readCatalog, readKey } from '../config.js';
import type { DecisionRequest } from '../../src/types.js';
const request: DecisionRequest = { state: { battery: 18 }, model: 'fixture', questions: {
  'battery:R1': { type: 'score', instructions: 'Rate battery.', criteria: ['Low', 'Medium', 'High'] },
  'action:R3': { type: 'choice', instructions: 'Choose an action.', criteria: { wait: 'Wait', 'job:J4': 'Job 4', 'job:J2': 'Job 2', 'job:J1': 'Job 1', 'job:J3': 'Job 3', recharge: 'Recharge' } },
} };
function response() { return { model: 'jev-fixture', answers: Object.fromEntries(Object.entries(request.questions).map(([qid, q]) => [qid, makeAnswer(q, candidateIds(q).map(() => 1 / candidateIds(q).length))])), usage: { input_tokens: 100, output_tokens: 20, cost: 0.01 } }; }

test('hosted Jev score derivation preserves probabilities and rejects malformed distributions', () => {
  const raw = response(), answer = raw.answers['battery:R1'];
  answer.probabilities = { '0': 0.08, '1': 0.66, '2': 0.26 }; answer.score = 1.17;
  const normalized = normalizeResponse(raw, request, true);
  assert.ok(Math.abs(normalized.response.answers['battery:R1'].score! - 1.18) < 1e-8);
  assert.equal(raw.answers['battery:R1'].score, 1.17); assert.equal(normalized.derived.length, 1);
  assert.throws(() => normalizeResponse(raw, request, false));
  answer.probabilities['0'] = 0.2; assert.throws(() => normalizeResponse(raw, request, true));
});
test('Jev two-decimal rounding is normalized with an audit and preserves the selected action', () => {
  for (const total of [0.99, 1.01]) {
    const raw = response(); raw.model = 'jev-1.13.0';
    const answer = raw.answers['action:R3'];
    answer.probabilities = { wait: 0.04, 'job:J4': 0.01, 'job:J2': 0.28, 'job:J1': 0.27, 'job:J3': 0.16, recharge: total === 0.99 ? 0.23 : 0.25 };
    answer.choice = 'job:J2';
    const original = structuredClone(raw);
    const normalized = normalizeResponse(raw, request, true), served = normalized.response.answers['action:R3'];
    assert.ok(Math.abs(Object.values(served.probabilities!).reduce((a, b) => a + b, 0) - 1) < 1e-12);
    assert.equal(served.choice, 'job:J2'); assert.equal(served.confidence, answer.confidence);
    assert.ok(Math.abs(served.probabilities!['job:J2'] - 0.28 / total) < 1e-12);
    assert.deepEqual(raw, original);
    assert.equal(normalized.normalizedProbabilities.length, 1);
    assert.equal(normalized.normalizedProbabilities[0].question_id, 'action:R3');
    assert.deepEqual(normalized.normalizedProbabilities[0].reported_probabilities, answer.probabilities);
    assert.deepEqual(normalized.normalizedProbabilities[0].served_probabilities, served.probabilities);
    assert.throws(() => normalizeResponse(raw, request, false), /sum to one/);
  }
});
test('rounded Jev scores use the normalized distribution while malformed responses still fail', () => {
  const raw = response(); raw.model = 'jev-1.13.0';
  const answer = raw.answers['battery:R1'];
  answer.probabilities = { '0': 0.56, '1': 0.41, '2': 0.02 }; answer.score = 0.46;
  const normalized = normalizeResponse(raw, request, true);
  assert.ok(Math.abs(normalized.response.answers['battery:R1'].score! - 0.45 / 0.99) < 1e-12);
  assert.equal(normalized.derived[0].reported_score, 0.46);
  assert.equal(normalized.normalizedProbabilities.length, 1);
  for (const probabilities of [{ '0': 0.334, '1': 0.334, '2': 0.333 }, { '0': 0.5, '1': 0.4, '2': 0.2 }, { '0': 0, '1': 0, '2': 0 }]) {
    answer.probabilities = probabilities;
    assert.throws(() => normalizeResponse(raw, request, true), /sum to one/);
  }
  answer.probabilities = { '0': 0.56, '1': 0.41, '2': 0.02 }; raw.model = 'another-provider/model';
  assert.throws(() => normalizeResponse(raw, request, true), /sum to one/);
});
test('TypeSafe preserves a selected argmax tie across floating-point serialization', () => {
  const raw = response(); raw.model = 'jev-1.13.0';
  const answer = raw.answers['action:R3'];
  answer.probabilities = { 'job:J2': 0.27999999999999997, recharge: 0.25, 'job:J3': 0.11, 'job:J4': 0.02, 'job:J1': 0.28, wait: 0.06 };
  answer.choice = 'job:J2';
  const normalized = normalizeResponse(raw, request, true);
  assert.equal(normalized.response.answers['action:R3'].choice, 'job:J2');
  assert.deepEqual(normalized.response.answers['action:R3'].probabilities, answer.probabilities);
  assert.throws(() => normalizeResponse(raw, request, false), /argmax/);
  answer.probabilities['job:J2'] = 0.279; answer.probabilities.wait = 0.061;
  assert.throws(() => normalizeResponse(raw, request, true), /argmax/);
});
test('TypeSafe telemetry retains the raw rounded Jev response and its normalization', async () => {
  const raw = response(); raw.model = 'jev-1.13.0';
  raw.answers['battery:R1'].probabilities = { '0': 0.56, '1': 0.41, '2': 0.02 };
  const text = JSON.stringify(raw);
  const backend = new RemoteBackend({ base_url: 'https://api.typesafe.ai', model: 'jev-latest' }, 'fixture-key', 30, (async () => new Response(text)) as typeof fetch);
  const result = await backend.evaluate(request, new AbortController().signal);
  assert.equal(result.telemetry.raw_response_text, text);
  assert.equal((result.telemetry.probability_normalizations as unknown[]).length, 1);
});
test('TypeSafe uses its key once; local serving uses a placeholder', async () => {
  for (const [url, hosted] of [
    ['https://api.typesafe.ai', true],
    ['https://api.typesafe.ai/v1/systemone', true],
    ['http://127.0.0.1:8000', false],
    ['http://127.0.0.1:8000/v1/systemone', false],
    ['https://model.example.com', false],
  ] as const) {
    let calls = 0;
    const fetcher = (async (input, init) => {
      calls++; assert.equal(String(input), url.endsWith('/v1/systemone') ? url : `${url}/v1/systemone`);
      assert.equal(init?.method, 'POST'); assert.equal(init?.redirect, 'error');
      assert.equal(new Headers(init?.headers).get('Authorization'), hosted ? 'Bearer fixture-key' : 'Bearer local');
      assert.equal(JSON.parse(String(init?.body)).model, 'configured-model');
      return new Response(JSON.stringify(response()));
    }) as typeof fetch;
    const backend = new RemoteBackend({ base_url: url, model: 'configured-model' }, 'fixture-key', 30, fetcher);
    const result = await backend.evaluate(request, new AbortController().signal);
    assert.equal(calls, 1); assert.equal(result.response.usage.output_tokens, 20);
    assert.equal(JSON.stringify(result).includes('fixture-key'), false);
  }
});
test('failed responses retain status/raw body without retry or fallback', async () => {
  let count = 0;
  const backend = new RemoteBackend({ base_url: 'https://api.typesafe.ai', model: 'fixture' }, 'key', 30, (async () => { count++; return new Response('rate limited', { status: 429 }); }) as typeof fetch);
  await assert.rejects(backend.evaluate(request, new AbortController().signal), (error: unknown) => error instanceof HttpError && error.status === 429 && error.telemetry.raw_response_text === 'rate limited');
  assert.equal(count, 1);
});
test('Snake’s batched questions work through both Jev and the deployed model adapter', async () => {
  for (const hosted of [true, false]) {
    const endpoint = hosted ? 'https://api.typesafe.ai' : 'http://127.0.0.1:8000';
    const model = hosted ? 'jev-latest' : 'GS1';
    const game = new SnakeGame(6, 7), input = game.request(model);
    const best = game.moves().filter(move => move.safe).sort((a, b) => b.advance - a.advance)[0].direction;
    const raw = { model, usage: { input_tokens: 120, output_tokens: 0 }, answers: Object.fromEntries(Object.entries(input.questions).map(([id, q]) => [id, makeAnswer(q, candidateIds(q).map((key, i) => q.type === 'choice' ? Number(key === best) : Number(i === 0)))])) };
    let calls = 0;
    const backend = new RemoteBackend({ base_url: endpoint, model }, 'fixture-key', 30, (async (url, init) => {
      calls++;
      assert.equal(url, `${endpoint}/v1/systemone`);
      assert.equal(new Headers(init?.headers).get('Authorization'), hosted ? 'Bearer fixture-key' : 'Bearer local');
      assert.deepEqual(JSON.parse(String(init?.body)), input);
      return new Response(JSON.stringify(raw));
    }) as typeof fetch);
    const result = await backend.evaluate(input, new AbortController().signal);
    assert.equal(calls, 1); assert.equal(result.telemetry.raw_response_text, JSON.stringify(raw));
    assert.deepEqual(Object.keys(result.response.answers), ['move', 'risk', 'food']);
    assert.equal(game.apply(result.response.answers.move.choice!, result.response.answers.move.probabilities), best);
    assert.equal(game.state.steps, 1); assert.equal(game.state.interventions, 0);
  }
});
test('catalog rejects serving variables and embedded credentials', t => {
  const dir = mkdtempSync(join(tmpdir(), 'gs1-catalog-')); t.after(() => rmSync(dir, { recursive: true, force: true }));
  const path = join(dir, 'backends.yaml');
  writeFileSync(path, 'backends:\n  local:\n    base_url: http://localhost:8000\n    model: fixture\n    device: cuda\n'); assert.throws(() => readCatalog(path));
  writeFileSync(path, 'backends:\n  local:\n    base_url: http://secret:password@localhost:8000\n    model: fixture\n'); assert.throws(() => readCatalog(path));
});

test('Jev requires a TypeSafe key before making any network request', async () => {
  let calls = 0;
  const backend = new RemoteBackend({ base_url: 'https://api.typesafe.ai', model: 'jev-latest' }, '', 30, (async () => { calls++; throw new Error('Unexpected network request'); }) as typeof fetch);
  await backend.probe();
  assert.equal(backend.available, false); assert.match(backend.reason, /TYPESAFE_API_KEY/);
  await assert.rejects(backend.evaluate(request, new AbortController().signal), (error: unknown) => error instanceof HttpError && error.status === 401 && error.message.includes('TYPESAFE_API_KEY'));
  assert.equal(calls, 0);
});
test('TypeSafe credentials come from the environment before an optional env file', t => {
  const dir = mkdtempSync(join(tmpdir(), 'gs1-typesafe-key-'));
  const previous = process.env.TYPESAFE_API_KEY;
  t.after(() => {
    if (previous === undefined) delete process.env.TYPESAFE_API_KEY;
    else process.env.TYPESAFE_API_KEY = previous;
    rmSync(dir, { recursive: true, force: true });
  });
  const envFile = join(dir, '.env');
  writeFileSync(envFile, 'TYPESAFE_API_KEY=fixture-file-key\n');
  process.env.TYPESAFE_API_KEY = 'fixture-environment-key';
  assert.equal(readKey(envFile), 'fixture-environment-key');
  delete process.env.TYPESAFE_API_KEY;
  assert.equal(readKey(envFile), 'fixture-file-key');
  assert.equal(readKey(join(dir, 'missing.env')), '');
});
