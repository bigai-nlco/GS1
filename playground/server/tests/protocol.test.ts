import test from 'node:test';
import assert from 'node:assert/strict';
import { choiceConfidence, makeAnswer, noulConfidence } from '../protocol.js';

const close = (actual: number | undefined, expected: number) => assert.ok(
  actual !== undefined && Math.abs(actual - expected) < 1e-12,
  `Expected ${expected}, received ${actual}`,
);

test('Choice confidence matches the TypeSafe examples', () => {
  const question = { type: 'choice' as const, instructions: 'Choose', criteria: { a: null, b: null, c: null } };
  for (const p of [[0.6, 0.3, 0.1], [0.6, 0.2, 0.2]]) close(makeAnswer(question, p).confidence, 0.4);
  close(makeAnswer({ ...question, criteria: { a: null } }, [1]).confidence, 1);
});

test('Score confidence accounts for distance between ordered levels', () => {
  const cases: [number[], number][] = [
    [[0, 0.5, 0.5], 0.25],
    [[0.5, 0, 0.5], 0],
    [[0, 0.57, 0.43], 0.355],
    [[0.6, 0.2, 0.2], 0.1],
    [[0.5, 0.5, 0, 0], 0.5],
    [[0, 0, 0.2, 0.6, 0.2], 2 / 3],
  ];
  for (const [p, expected] of cases) {
    const question = { type: 'score' as const, instructions: 'Score', criteria: p.map((_, i) => String(i)) };
    const answer = makeAnswer(question, p);
    close(answer.confidence, expected);
    close(answer.score, p.reduce((total, value, i) => total + i * value, 0));
  }
});

test('Uniform distributions have zero confidence and certain outcomes have one', () => {
  for (let count = 2; count <= 10; count++) {
    const criteria = Array.from({ length: count }, (_, i) => String(i));
    const questions = [
      { type: 'choice' as const, instructions: 'Choose', criteria: Object.fromEntries(criteria.map(k => [k, null])) },
      { type: 'score' as const, instructions: 'Score', criteria },
    ];
    for (const question of questions) {
      close(makeAnswer(question, criteria.map(() => 1 / count)).confidence, 0);
      for (let level = 0; level < count; level++) close(makeAnswer(question, criteria.map((_, i) => Number(i === level))).confidence, 1);
    }
  }
});

test('Noul exposes p(true) and its optional confidence matches a binary Choice', () => {
  for (const [p, expected] of [[0, 1], [0.2, 0.6], [0.5, 0], [0.8, 0.6], [1, 1]]) {
    const answer = makeAnswer({ type: 'noul', instructions: 'Is this true?' }, [1 - p, p]);
    assert.deepEqual(answer, { type: 'noul', noul: p });
    close(noulConfidence(p), expected);
    close(choiceConfidence([1 - p, p]), expected);
  }
});
