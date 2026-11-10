import test from 'node:test';
import assert from 'node:assert/strict';
import { runQuery } from '../query.js';
import { parseDecisionRequest } from '../../src/request.js';
import { candidateIds, makeAnswer, HttpError } from '../protocol.js';
import type { DecisionRequest } from '../../src/types.js';

const questions = {
  choice: { type: 'choice', instructions: { question: 'Pick one.' }, criteria: { first: { index: 0 }, second: ['index 1'], other: null } },
  noul: { type: 'noul', instructions: ['Is this true?'], criteria: { true: { meaning: 'yes' }, false: ['no'] } },
  score: { type: 'score', instructions: 'Rate it.', criteria: ['Low', { level: 'Medium' }, ['High']] },
};
function fixture(delay = 0) {
  const requests: DecisionRequest[] = [];
  const models: Record<string, string> = { jev: 'jev-latest', gs1_local: 'GS1' };
  const registry = { profile: (id: string) => {
    if (!models[id]) throw new HttpError('Unknown backend.');
    return { modelName: models[id], async evaluate(request: DecisionRequest) {
      requests.push(structuredClone(request));
      if (delay) await new Promise(resolve => setTimeout(resolve, delay));
      const response = { model: request.model, usage: { input_tokens: 10, output_tokens: 0 }, answers: Object.fromEntries(Object.entries(request.questions).map(([id, q]) => [id, makeAnswer(q, candidateIds(q).map((_, i) => Number(i === 0)))])) };
      return { response, telemetry: { raw_response_text: JSON.stringify({ ...response, provider: 'TypeSafe', id: 'raw-provider-id', usage: { ...response.usage, cost: .0001 } }), client_ms: 12.5 } };
    } };
  } };
  return { query: (value: Record<string, unknown>, signal = new AbortController().signal) => runQuery(registry, 'jev', value, signal), requests };
}
test('all documented primitive shapes preserve nested state, instructions, criteria and provider JSON', async () => {
  const f = fixture();
  for (const state of ['plain text', { nested: [1, false, null] }, [{ name: 'first' }]]) {
    const result = await f.query({ state, questions, model: 'jev-latest' });
    assert.deepEqual(result.request.state, state); assert.deepEqual(result.request.questions, questions);
    assert.equal(result.raw_response.provider, 'TypeSafe'); assert.equal(result.raw_response.id, 'raw-provider-id');
    assert.equal((result.raw_response.usage as unknown as { cost: number }).cost, .0001);
    assert.equal(result.latency_ms, 12.5); assert.equal(result.response.usage.input_tokens, 10);
  }
});
test('malformed requests fail before consuming an API call', async () => {
  const f = fixture();
  for (const input of [{ state: null, questions }, { state: 'ok', questions: {} }, { state: [], questions: { q: { type: 'score', instructions: 'Rate', criteria: ['only one'] } } }, { state: 'ok', questions: { q: { type: 'choice', instructions: 'Pick', criteria: [] } } }, { state: 'ok', questions, model: 'other-model' }, { state: 'ok', questions, backend: 1 }, { state: 'ok', questions, backend: 'unknown' }]) {
    await assert.rejects(f.query(input));
  }
  assert.equal(f.requests.length, 0);
  const boundary = Object.fromEntries(Array.from({ length: 255 }, (_, i) => [String(i), null]));
  parseDecisionRequest({ state: [], questions: { q: { type: 'choice', instructions: 'Pick', criteria: boundary } } }, 'jev');
  assert.throws(() => parseDecisionRequest({ state: [], questions: { q: { type: 'choice', instructions: 'Pick', criteria: { ...boundary, extra: null } } } }, 'jev'));
});
test('Query dispatches to the selected local or TypeSafe model', async () => {
  const f = fixture();
  const local = await f.query({ state: 'hello', questions, backend: 'gs1_local' });
  const hosted = await f.query({ state: 'hello', questions, backend: 'jev' });
  assert.equal(local.request.model, 'GS1'); assert.equal(hosted.request.model, 'jev-latest');
  assert.deepEqual(f.requests.map(request => request.model), ['GS1', 'jev-latest']);
});
test('cancelled Query calls do not return a completed response', async () => {
  const f = fixture(15), controller = new AbortController();
  const pending = f.query({ state: 'hello', questions }, controller.signal); controller.abort();
  await assert.rejects(pending, (error: unknown) => error instanceof HttpError && error.status === 499);
});
