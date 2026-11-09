import test from 'node:test';
import assert from 'node:assert/strict';
import { SnakeGame } from '../snake.js';
import { parseDecisionRequest } from '../../src/request.js';

test('Snake creates deterministic boards and food outside its initial body', () => {
  for (const size of [6, 8, 10, 12]) {
    const game = new SnakeGame(size, 7);
    assert.deepEqual(game.state, new SnakeGame(size, 7).state);
    assert.equal(game.state.body.length, 4); assert.equal(game.state.guarded, true);
    assert.ok(!game.state.body.some(cell => String(cell) === String(game.state.food)));
    assert.ok(game.moves().some(move => move.safe));
  }
  assert.notDeepEqual(new SnakeGame(10, 1).state.food, new SnakeGame(10, 2).state.food);
  for (const size of [0, 5, 7, 100]) assert.throws(() => new SnakeGame(size), /Snake requires/);
  assert.throws(() => new SnakeGame(10, NaN), /integer seed/);
});

test('Snake sends raw board observations with neutral direction criteria', () => {
  const game = new SnakeGame(), before = structuredClone(game.state);
  const request = game.request('GS1');
  const state = request.state as Record<string, unknown>;
  assert.deepEqual(parseDecisionRequest(request, 'GS1'), request);
  assert.deepEqual(Object.keys(request.questions), ['move', 'risk', 'food']);
  assert.equal(request.questions.move.type, 'choice');
  if (request.questions.move.type !== 'choice') throw new Error('Expected a choice question');
  assert.deepEqual(Object.keys(request.questions.move.criteria!), ['UP', 'DOWN', 'LEFT', 'RIGHT']);
  assert.equal(request.questions.risk.type, 'noul'); assert.equal(request.questions.food.type, 'noul');
  assert.deepEqual(Object.keys(state).sort(), ['game', 'size', 'coordinates', 'body_order', 'body', 'food', 'direction', 'score', 'steps', 'rules'].sort());
  assert.deepEqual(state.body, before.body); assert.deepEqual(state.food, before.food);
  assert.equal(state.size, before.size); assert.equal(state.direction, before.direction);
  assert.equal(state.body_order, 'head_to_tail'); assert.equal(state.score, before.score); assert.equal(state.steps, before.steps);
  assert.doesNotMatch(JSON.stringify(request.questions.move.criteria), /safe|best|collision|blocked|trap|progress|food|legal|cycle/i);
  assert.doesNotMatch(JSON.stringify(state), /reachable|safe|best|blocked|trap|progress|legal|cycle|guard|seed/i);
  assert.match(String(request.questions.risk.instructions), /move now without a collision/);
  assert.match(String(request.questions.food.instructions), /including the current tail/);
  assert.deepEqual(game.state, before);
  game.state.body[0][0]++; game.state.food![0]++;
  assert.deepEqual(state.body, before.body); assert.deepEqual(state.food, before.food);
});

test('Snake questions never change with food location, move legality, or planner metadata', () => {
  const game = new SnakeGame(6, 7), original = game.request('jev');
  const questions = original.questions;
  // Change the preferred move and then supply an enclosed head with no path to
  // food. Every direction remains a candidate with exactly the same wording.
  game.state.food = [3, 4];
  assert.deepEqual(game.request('jev').questions, questions);
  game.state.body = [[2, 2], [2, 1], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1], [4, 1]];
  game.state.food = [5, 5]; game.state.direction = 'DOWN';
  assert.ok(game.moves().every(move => !move.legal));
  const trapped = game.request('jev');
  assert.deepEqual(trapped.questions, questions);
  assert.deepEqual((trapped.state as Record<string, unknown>).body, game.state.body);
  game.state.guarded = false; game.state.seed = 99; game.state.interventions = 17;
  game.state.last = { proposed: 'UP', executed: 'DOWN', intervened: true };
  game.moves = () => { throw new Error('Request generation must not evaluate or filter moves'); };
  assert.deepEqual(game.request('jev'), trapped);
  assert.deepEqual((original.state as Record<string, unknown>).body, new SnakeGame(6, 7).state.body, 'saved requests own their board snapshot');
});

test('food grows the snake by one and unknown directions do not advance it', () => {
  const game = new SnakeGame(6, 1, false);
  game.state.body = [[2, 2], [1, 2], [0, 2]]; game.state.food = [3, 2];
  const before = structuredClone(game.state);
  assert.throws(() => game.apply('pass'), /unknown/); assert.deepEqual(game.state, before);
  game.apply('RIGHT');
  assert.equal(game.state.score, 1); assert.equal(game.state.steps, 1);
  assert.deepEqual(game.state.body, [[3, 2], [2, 2], [1, 2], [0, 2]]);
  assert.ok(!game.state.body.some(cell => String(cell) === String(game.state.food)));
  assert.equal(game.state.finished, false);
});

test('unguarded wall, reverse, and body collisions end the game without corrupting the board', () => {
  for (const reason of ['wall', 'reverse', 'body'] as const) {
    const game = new SnakeGame(6, 1, false);
    game.state.body = reason === 'wall' ? [[0, 0], [0, 1], [0, 2]] : [[2, 2], [2, 1], [3, 1], [3, 2], [3, 3], [2, 3]];
    game.state.food = [5, 5];
    const body = structuredClone(game.state.body);
    game.apply(reason === 'body' ? 'RIGHT' : 'UP');
    assert.equal(game.state.death_reason, reason); assert.equal(game.state.outcome, 'lost');
    assert.equal(game.state.finished, true); assert.equal(game.state.steps, 1);
    assert.deepEqual(game.state.body, body);
    assert.throws(() => game.apply('DOWN'), /ended/);
  }
});

test('the tail vacates on a non-growing move', () => {
  const game = new SnakeGame(6, 1, false);
  game.state.body = [[2, 2], [2, 1], [3, 1], [3, 2], [3, 3], [2, 3]];
  game.state.food = [5, 5];
  assert.equal(game.moves().find(move => move.direction === 'DOWN')!.legal, true);
  game.apply('DOWN'); assert.equal(game.state.body.length, 6); assert.deepEqual(game.state.body[0], [2, 3]);
});

test('cycle guard chooses the highest model probability among certified moves and records the override', () => {
  const game = new SnakeGame(6, 7); game.state.food = [4, 0];
  const probabilities = { UP: .2, DOWN: .01, LEFT: .7, RIGHT: .09 };
  assert.ok(game.moves().filter(move => move.safe).length > 1);
  const selected = game.apply('LEFT', probabilities);
  assert.equal(selected, 'UP'); assert.equal(game.state.interventions, 1);
  assert.deepEqual(game.state.last, { proposed: 'LEFT', executed: 'UP', intervened: true });
  assert.equal(game.state.finished, false);
  assert.deepEqual(probabilities, { UP: .2, DOWN: .01, LEFT: .7, RIGHT: .09 });
});

test('guarded seeded games preserve the route, grow, and stop at a full board', () => {
  for (const seed of [1, 7, 42]) {
    const game = new SnakeGame(6, seed);
    // A deliberately poor model always proposes UP; the guard must still
    // preserve order and make eventual food progress, including tail moves.
    const probabilities = { UP: .7, DOWN: .1, LEFT: .1, RIGHT: .1 };
    for (let i = 0; i < 36 * 36 && !game.state.finished; i++) {
      const safe = game.moves().filter(move => move.safe).map(move => move.direction);
      assert.ok(safe.length);
      const selected = game.apply('UP', probabilities);
      assert.ok(safe.includes(selected));
      assert.equal(new Set(game.state.body.map(String)).size, game.state.body.length);
      assert.equal(game.state.body.length, 4 + game.state.score);
    }
    assert.equal(game.state.outcome, 'won'); assert.equal(game.state.score, 32);
    assert.equal(game.state.body.length, 36); assert.equal(game.state.food, null);
    assert.throws(() => game.apply('UP', probabilities), /ended/);
    assert.throws(() => game.request('jev'), /ended/);
  }
});
