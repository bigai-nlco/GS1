import test from 'node:test';
import assert from 'node:assert/strict';
import { createPlaygroundServer } from '../app.js';
import { defaults } from '../config.js';
import { candidateIds, makeAnswer } from '../protocol.js';
import type { DecisionRequest } from '../../src/types.js';

async function fixture(t: test.TestContext) {
  let calls = 0;
  const registry = {
    profile: () => ({ modelName: 'fixture', async evaluate(request: DecisionRequest) {
      calls++;
      return { response: { model: 'fixture', usage: { input_tokens: 10, output_tokens: 0 }, answers: Object.fromEntries(Object.entries(request.questions).map(([qid, q]) => [qid, makeAnswer(q, candidateIds(q).map(() => 1 / candidateIds(q).length))])) }, telemetry: {} };
    } }),
    profiles: async () => [{ id: 'fixture', label: 'Fixture', model: 'fixture', available: true, reason: '' }],
  };
  const app = createPlaygroundServer({ ...defaults, defaultBackend: 'fixture' }, registry);
  await new Promise<void>(resolve => app.server.listen(0, '127.0.0.1', resolve));
  const address = app.server.address(); if (!address || typeof address === 'string') throw new Error('Missing test address');
  const url = `http://127.0.0.1:${address.port}`;
  t.after(() => app.close());
  return { url, calls: () => calls, post: async (path: string, body: unknown) => fetch(url + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) };
}

test('Query HTTP routes expose typed answers, raw responses, latency and backend discovery', async t => {
  const f = await fixture(t);
  const result = await f.post('/playground/query', { state: ['hello'], questions: { q: { type: 'noul', instructions: ['Is this a greeting?'] } } });
  assert.equal(result.status, 200);
  const value = await result.json(); assert.equal(value.raw_response.model, 'fixture'); assert.equal(typeof value.latency_ms, 'number');
  assert.deepEqual(value.request.state, ['hello']); assert.equal(value.response.answers.q.noul, 0.5);
  const bad = await f.post('/playground/query', { state: 'hello', questions: {} }); assert.equal(bad.status, 422);
  const backends = await (await fetch(`${f.url}/playground/backends`)).json();
  assert.equal(backends.default, 'fixture'); assert.equal(backends.backends[0].model, 'fixture');
  const docs = await (await fetch(`${f.url}/docs`)).json();
  assert.deepEqual(Object.keys(docs.playground), ['GET /playground/backends', 'POST /playground/query']);
  assert.equal(f.calls(), 1);
});

test('removed session endpoints return 404 without calling a model', async t => {
  const f = await fixture(t), id = '0'.repeat(32);
  assert.equal((await f.post('/playground/games', {})).status, 404);
  assert.equal((await fetch(`${f.url}/playground/games/${id}`)).status, 404);
  assert.equal((await f.post(`/playground/games/${id}/step`, {})).status, 404);
  assert.equal(f.calls(), 0);
});
