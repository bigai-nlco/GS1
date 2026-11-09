import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createPlaygroundServer } from '../app.js';
import { defaults } from '../config.js';
import { candidateIds, makeAnswer } from '../protocol.js';
import type { DecisionRequest } from '../../src/types.js';

async function fixture(t: test.TestContext) {
  const dir = mkdtempSync(join(tmpdir(), 'gs1-http-'));
  const registry = {
    profile: () => ({ modelName: 'fixture', async evaluate(request: DecisionRequest) { return { response: { model: 'fixture', usage: { input_tokens: 10, output_tokens: 0 }, answers: Object.fromEntries(Object.entries(request.questions).map(([qid, q]) => [qid, makeAnswer(q, candidateIds(q).map(() => 1 / candidateIds(q).length))])) }, telemetry: {} }; } }),
    profiles: async () => [{ id: 'fixture', label: 'Fixture', model: 'fixture', available: true, reason: '' }],
  };
  const settings = { ...defaults, runsDir: dir, defaultBackend: 'fixture' };
  const app = createPlaygroundServer(settings, registry);
  await new Promise<void>(resolve => app.server.listen(0, '127.0.0.1', resolve));
  const address = app.server.address(); if (!address || typeof address === 'string') throw new Error('Missing test address');
  const url = `http://127.0.0.1:${address.port}`;
  t.after(async () => { await app.close(); rmSync(dir, { recursive: true, force: true }); });
  return { app, url, settings, registry, post: async (path: string, body: unknown) => fetch(url + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) };
}

test('query and game HTTP routes expose raw responses, latency, and versioned moves', async t => {
  const f = await fixture(t);
  const result = await f.post('/playground/query', { state: ['hello'], questions: { q: { type: 'noul', instructions: ['Is this a greeting?'] } } });
  assert.equal(result.status, 200);
  const value = await result.json(); assert.equal(value.raw_response.model, 'fixture'); assert.equal(typeof value.latency_ms, 'number');
  assert.deepEqual(value.request.state, ['hello']);
  const bad = await f.post('/playground/query', { state: 'hello', questions: {} }); assert.equal(bad.status, 422);
  const game = await (await f.post('/playground/games', { kind: 'maze', size: 5, seed: 42 })).json();
  const step = await (await f.post(`/playground/games/${game.id}/step`, { version: 0 })).json();
  assert.equal(step.version, 1); assert.equal(step.state.steps, 1);
  assert.equal((await f.post(`/playground/games/${game.id}/step`, { version: 0 })).status, 409);
  assert.equal((await fetch(`${f.url}/playground/games/${game.id}`)).status, 200);
});
