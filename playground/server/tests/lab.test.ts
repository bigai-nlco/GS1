import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Lab } from '../lab.js';
import { goSettlement } from '../games.js';
import { parseDecisionRequest } from '../../src/request.js';
import { candidateIds, makeAnswer } from '../protocol.js';
import type { DecisionRequest } from '../../src/types.js';

const questions = {
  choice: { type: 'choice', instructions: { question: 'Pick one.' }, criteria: { first: { index: 0 }, second: ['index 1'], other: null } },
  noul: { type: 'noul', instructions: ['Is this true?'], criteria: { true: { meaning: 'yes' }, false: ['no'] } },
  score: { type: 'score', instructions: 'Rate it.', criteria: ['Low', { level: 'Medium' }, ['High']] },
};
function fixture(t: test.TestContext, delay = 0, choice = 'C3') {
  const runs = mkdtempSync(join(tmpdir(), 'gs1-lab-')), requests: DecisionRequest[] = [];
  t.after(() => rmSync(runs, { recursive: true, force: true }));
  const registry = { profile: () => ({ modelName: 'jev-latest', async evaluate(request: DecisionRequest) {
    requests.push(structuredClone(request));
    if (delay) await new Promise(resolve => setTimeout(resolve, delay));
    const response = { model: 'jev-fixture', usage: { input_tokens: 10, output_tokens: 0 }, answers: Object.fromEntries(Object.entries(request.questions).map(([id, q]) => {
      const keys = candidateIds(q), p = keys.map((_, i) => keys.includes(choice) ? Number(keys[i] === choice) : Number(i === 0));
      return [id, makeAnswer(q, p)];
    })) };
    return { response, telemetry: { raw_response_text: JSON.stringify({ ...response, provider: 'TypeSafe', id: 'raw-provider-id', usage: { ...response.usage, cost: .0001 } }), client_ms: 12.5 } };
  } }) };
  return { lab: new Lab(registry, 'jev', runs), requests };
}
test('all documented primitive shapes preserve nested state, instructions, and criteria', async t => {
  const f = fixture(t);
  for (const state of ['plain text', { nested: [1, false, null] }, [{ name: 'first' }]]) {
    const result = await f.lab.query({ state, questions, model: 'jev-latest' }, new AbortController().signal);
    assert.deepEqual(result.request.state, state); assert.deepEqual(result.request.questions, questions);
    assert.equal(result.raw_response.provider, 'TypeSafe'); assert.equal(result.raw_response.id, 'raw-provider-id');
    assert.equal((result.raw_response.usage as unknown as { cost: number }).cost, .0001);
    assert.equal(result.latency_ms, 12.5);
    assert.equal(result.response.usage.input_tokens, 10);
  }
});
test('malformed requests fail before consuming an API call', async t => {
  const f = fixture(t);
  for (const input of [{ state: null, questions }, { state: 'ok', questions: {} }, { state: [], questions: { q: { type: 'score', instructions: 'Rate', criteria: ['only one'] } } }, { state: 'ok', questions: { q: { type: 'choice', instructions: 'Pick', criteria: [] } } }, { state: 'ok', questions, model: 'local-trained-model' }]) {
    await assert.rejects(f.lab.query(input, new AbortController().signal));
  }
  assert.equal(f.requests.length, 0);
  const boundary = Object.fromEntries(Array.from({ length: 255 }, (_, i) => [String(i), null]));
  parseDecisionRequest({ state: [], questions: { q: { type: 'choice', instructions: 'Pick', criteria: boundary } } }, 'jev');
  assert.throws(() => parseDecisionRequest({ state: [], questions: { q: { type: 'choice', instructions: 'Pick', criteria: { ...boundary, extra: null } } } }, 'jev'));
});
test('game steps reject overlaps and stale versions while alternating legal placements', async t => {
  const f = fixture(t, 15), first = f.lab.create({ kind: 'go' });
  const pending = f.lab.step(first.id, { version: 0 }, new AbortController().signal);
  await assert.rejects(f.lab.step(first.id, { version: 0 }, new AbortController().signal), /in progress/);
  const next = await pending;
  assert.equal(next.version, 1); assert.equal(next.moves[0].player, 'Black');
  await assert.rejects(f.lab.step(first.id, { version: 0 }, new AbortController().signal), /changed/);
  const nextMove = await f.lab.step(first.id, { version: 1 }, new AbortController().signal);
  assert.equal(nextMove.state.finished, false); assert.equal(nextMove.moves[1].player, 'White');
  const continued = await f.lab.step(first.id, { version: 2 }, new AbortController().signal);
  assert.equal(continued.version, 3); assert.equal(continued.state.finished, false);
  assert.deepEqual(continued.moves.map(move => move.player), ['Black', 'White', 'Black']);
  assert.ok(continued.moves.every(move => move.choice !== 'pass'));
  assert.ok(f.requests.every(request => !Object.keys(request.questions.move.criteria!).includes('pass')));
  assert.equal(f.requests.length, 3);
});
test('no legal Go placements pauses an unresolved session before consuming an API call', async t => {
  const f = fixture(t), first = f.lab.create({ kind: 'go' });
  const session = f.lab.get(first.id), state = session.game.state;
  if (state.kind !== 'go') throw new Error('Expected Go');
  // Black has one remaining eye: filling it would be suicide.
  state.board.fill(1); state.board[12] = 0;
  state.settlement = goSettlement(state.board, 5);
  const before = session.snapshot();
  await assert.rejects(f.lab.step(first.id, { version: 0 }, new AbortController().signal), /No legal Go placements/);
  assert.equal(f.requests.length, 0); assert.equal(session.busy, false);
  assert.deepEqual(session.snapshot(), before);
  assert.equal(state.finished, false); assert.equal(state.winner, null);
});
test('a model move that settles owned and neutral points ends the session and prevents further API calls', async t => {
  const f = fixture(t, 0, 'B3'), first = f.lab.create({ kind: 'go' });
  const state = f.lab.get(first.id).game.state;
  assert.equal(state.kind, 'go');
  if (state.kind !== 'go') throw new Error('Expected Go');
  state.board = ['BB.WW', '.B.W.', 'B..WW', '.B.W.', 'BB.WW'].join('').split('').map(s => s === 'B' ? 1 : s === 'W' ? 2 : 0);
  state.settlement = goSettlement(state.board, 5);
  const end = await f.lab.step(first.id, { version: 0 }, new AbortController().signal);
  assert.equal(end.state.finished, true); assert.equal(end.moves[0].choice, 'B3');
  assert.equal(state.settlement.neutral, 5); assert.equal(state.settlement.unsettled, 0);
  assert.deepEqual(state.score, { black: 10, white: 10.5 });
  await assert.rejects(f.lab.step(first.id, { version: 1 }, new AbortController().signal), /ended/);
  assert.equal(f.requests.length, 1);
});
test('cancelled model calls never apply a move', async t => {
  const f = fixture(t, 15), game = f.lab.create({ kind: 'maze', size: 5, seed: 42 }), controller = new AbortController();
  const pending = f.lab.step(game.id, { version: 0 }, controller.signal); controller.abort();
  await assert.rejects(pending, /cancelled/);
  assert.equal(f.lab.get(game.id).version, 0); assert.equal(f.lab.get(game.id).busy, false);
  assert.deepEqual(f.lab.get(game.id).snapshot().state, game.state);
});

test('Snake sessions preserve all three model answers, executed moves, and terminal state', async t => {
  const f = fixture(t, 0, 'UP'), first = f.lab.create({ kind: 'snake', size: 6, seed: 7, guarded: false });
  const session = f.lab.get(first.id), state = session.game.state;
  if (state.kind !== 'snake') throw new Error('Expected Snake');
  state.body = [[0, 0], [0, 1], [0, 2]]; state.food = [5, 5];
  const end = await f.lab.step(first.id, { version: 0 }, new AbortController().signal);
  assert.equal(end.state.finished, true); assert.equal(state.death_reason, 'wall');
  assert.equal(end.moves[0].choice, 'UP'); assert.equal(end.moves[0].proposed, 'UP');
  assert.equal(end.moves[0].player, 'Snake'); assert.equal(end.version, 1);
  assert.deepEqual(Object.keys(end.result!.response.answers), ['move', 'risk', 'food']);
  assert.equal(end.result!.raw_response.provider, 'TypeSafe'); assert.equal(end.result!.latency_ms, 12.5);
  await assert.rejects(f.lab.step(first.id, { version: 1 }, new AbortController().signal), /ended/);
  assert.equal(f.requests.length, 1);
  for (const input of [{ kind: 'snake', size: 7 }, { kind: 'snake', guarded: 'true' }, { kind: 'go', guarded: true }]) assert.throws(() => f.lab.create(input));
});

test('cancelled Snake decisions leave food, body, and version untouched', async t => {
  const f = fixture(t, 15), first = f.lab.create({ kind: 'snake', seed: 7 }), abort = new AbortController();
  const pending = f.lab.step(first.id, { version: 0 }, abort.signal); abort.abort();
  await assert.rejects(pending, /cancelled/);
  assert.deepEqual(f.lab.get(first.id).snapshot(), first);
  assert.equal(f.lab.get(first.id).busy, false);
});
