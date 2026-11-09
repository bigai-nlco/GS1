import { expect, test, type Page } from '@playwright/test';
import { SnakeGame } from '../server/snake';
import type { GameSnapshot } from '../src/lab-types';

async function fixture(page: Page, setup: (game: SnakeGame) => void = () => {}) {
  let game: SnakeGame, snapshot: GameSnapshot;
  const requests: { version: number; backend: string }[] = [];
  let hold: Promise<void> | undefined;
  await page.route('**/playground/backends', route => route.fulfill({ json: { default: 'gs1_local', backends: [
    { id: 'jev', label: 'Jev · OpenRouter', model: '~typesafe/jev-latest', available: true, reason: '' },
    { id: 'gs1_local', label: 'GS1 · GS1', model: 'GS1', available: true, reason: '' },
  ] } }));
  await page.route('**/playground/games', route => {
    const input = route.request().postDataJSON();
    game = new SnakeGame(input.size, 7, input.guarded); setup(game);
    snapshot = { id: 'snake-fixture', version: 0, state: game.state, result: null, moves: [] };
    return route.fulfill({ json: snapshot });
  });
  await page.route('**/playground/games/snake-fixture/step', async route => {
    const body = route.request().postDataJSON(); requests.push(body);
    if (hold) await hold;
    const model = body.backend === 'gs1_local' ? 'GS1' : '~typesafe/jev-latest';
    const request = game.request(model), proposed = game.state.guarded ? 'LEFT' : 'UP';
    const probabilities = { UP: proposed === 'UP' ? .7 : .2, DOWN: .01, LEFT: proposed === 'LEFT' ? .7 : .2, RIGHT: .09 };
    const response = { model, usage: { input_tokens: 100, output_tokens: 0 }, answers: { move: { type: 'choice' as const, choice: proposed, probabilities, confidence: .7 }, risk: { type: 'noul' as const, noul: .9 }, food: { type: 'noul' as const, noul: .8 } } };
    const choice = game.apply(proposed, probabilities); snapshot.version++;
    snapshot.result = { request, response, raw_response: { ...response, provider: 'fixture-provider' }, latency_ms: 17 };
    snapshot.moves.push({ number: snapshot.version, player: 'Snake', proposed, choice, latency_ms: 17 });
    await route.fulfill({ json: snapshot });
  });
  return { requests, holdNext: (promise: Promise<void>) => { hold = promise; } };
}

test('Snake creates, resizes, and regenerates real seeded boards', async ({ page }) => {
  await page.goto('/'); await page.getByRole('button', { name: /^Snake/ }).click();
  await expect(page.getByRole('img', { name: /10 by 10 Snake board/ })).toBeVisible();
  await expect(page.getByRole('checkbox', { name: 'Cycle guard', exact: true })).toBeChecked();
  const seed = await page.locator('.snake-legend code').textContent();
  await page.getByRole('button', { name: 'New game', exact: true }).click();
  await expect(page.locator('.snake-legend code')).not.toHaveText(seed!);
  await page.getByLabel('Snake board size').selectOption('6');
  await expect(page.getByRole('img', { name: /6 by 6 Snake board/ })).toBeVisible();
  await page.getByRole('checkbox', { name: 'Cycle guard', exact: true }).uncheck();
  await expect(page.getByRole('checkbox', { name: 'Cycle guard', exact: true })).not.toBeChecked();
  await expect(page.getByText(/chosen direction is executed directly/)).toBeVisible();
  await page.getByRole('button', { name: 'Maze 03', exact: true }).click();
  await expect(page.getByRole('img', { name: /Maze. Explorer/ })).toBeVisible();
  await page.getByRole('button', { name: /^Snake/ }).click();
  await expect(page.getByRole('img', { name: /6 by 6 Snake board/ })).toBeVisible();
  await expect(page.getByRole('checkbox', { name: 'Cycle guard', exact: true })).not.toBeChecked();
});

test('Snake preserves model probabilities and guard interventions, selects deployed models, and pauses after the pending move', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  const f = await fixture(page, game => { game.state.food = [4, 0]; });
  let release!: () => void; f.holdNext(new Promise<void>(resolve => { release = resolve; }));
  try {
    await page.goto('/'); await page.getByRole('button', { name: /^Snake/ }).click();
    await page.getByLabel('Snake board size').selectOption('6');
    await expect(page.getByRole('img', { name: /6 by 6 Snake board/ })).toBeVisible();
    await page.getByLabel('Decision model').selectOption('gs1_local');
    await expect(page.locator('.connection')).toContainText('Connected to model');
    await page.getByRole('button', { name: 'Run snake', exact: true }).click();
    await expect.poll(() => f.requests.length).toBe(1);
    await page.getByRole('button', { name: 'Pause', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Pausing…', exact: true })).toBeDisabled();
    release();
    await expect(page.getByRole('button', { name: 'Run snake', exact: true })).toBeEnabled();
    await expect(page.locator('.game-status')).toContainText('Step 1');
    await expect(page.locator('.snake-decision')).toContainText('Model LEFT');
    await expect(page.locator('.snake-decision')).toContainText('Played UP');
    await expect(page.locator('.snake-guard')).toContainText('1 intervention');
    await expect(page.locator('.answer-card')).toHaveCount(3);
    await expect(page.locator('.answer-card').first().locator('.answer-value')).toHaveText('LEFT');
    await expect(page.locator('.latency')).toContainText('17 ms');
    await page.getByRole('tab', { name: 'JSON', exact: true }).click();
    await expect(page.getByLabel('Raw JSON response', { exact: true })).toContainText('"choice": "LEFT"');
    await expect(page.getByLabel('Raw JSON response', { exact: true })).toContainText('fixture-provider');
    await expect(page.locator('.latency')).toContainText('17 ms');
    await page.getByRole('tab', { name: 'Overview', exact: true }).click();
    expect(f.requests).toEqual([{ version: 0, backend: 'gs1_local' }]);
    await page.screenshot({ path: '../.local/output/playwright/snake-desktop.png', fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await expect(page.getByRole('button', { name: /^Snake/ })).toBeVisible();
    await page.screenshot({ path: '../.local/output/playwright/snake-mobile.png', fullPage: true });
    await page.getByRole('button', { name: 'New game', exact: true }).click();
    await expect(page.locator('.game-status')).toContainText('Step 0');
    await expect(page.locator('.snake-decision')).toHaveCount(0);
    await expect(page.locator('.answer-card')).toHaveCount(0);
    expect(errors).toEqual([]);
  } finally { release(); }
});

test('Snake stops on a model-selected collision and can start again', async ({ page }) => {
  const f = await fixture(page, game => { game.state.body = [[0, 0], [0, 1], [0, 2]]; game.state.food = [5, 5]; });
  await page.goto('/'); await page.getByRole('button', { name: /^Snake/ }).click();
  await page.getByRole('checkbox', { name: 'Cycle guard', exact: true }).uncheck();
  await page.getByRole('button', { name: 'Run snake', exact: true }).click();
  await expect(page.locator('.completion')).toContainText('Game over: hit the wall. Score 0.');
  await expect(page.getByRole('button', { name: 'Run snake', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Step', exact: true })).toBeDisabled();
  expect(f.requests).toHaveLength(1);
  await page.getByRole('button', { name: 'New game', exact: true }).click();
  await expect(page.locator('.completion')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Run snake', exact: true })).toBeEnabled();
});

test('Snake stops automatically when the final food fills the board', async ({ page }) => {
  const f = await fixture(page, game => {
    if (game.state.size !== 6) return;
    for (let i = 0; i < 36 * 36; i++) {
      const safe = game.moves().filter(move => move.safe).sort((a, b) => b.advance - a.advance);
      if (game.state.body.length === 35 && safe[0].eats) return;
      game.apply(safe[0].direction);
    }
    throw new Error('Fixture did not reach the last food');
  });
  await page.goto('/'); await page.getByRole('button', { name: /^Snake/ }).click();
  await page.getByLabel('Snake board size').selectOption('6');
  await expect(page.getByRole('img', { name: /6 by 6 Snake board/ })).toBeVisible();
  await page.getByRole('button', { name: 'Run snake', exact: true }).click();
  await expect(page.locator('.completion')).toContainText('Board filled. You win with 32 food eaten.');
  await expect(page.getByRole('button', { name: 'Run snake', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Step', exact: true })).toBeDisabled();
  expect(f.requests).toHaveLength(1);
});

test('live Jev answers all three Snake questions and executes one move per response', async ({ page }) => {
  test.skip(process.env.GS1_LIVE !== '1', 'Set GS1_LIVE=1 to use the configured server-side key.');
  test.setTimeout(120_000);
  await page.goto('/'); await page.getByLabel('Decision model').selectOption('jev'); await page.getByRole('button', { name: /^Snake/ }).click();
  await expect(page.getByRole('img', { name: /10 by 10 Snake board/ })).toBeVisible();
  const created = page.waitForResponse(response => response.url().endsWith('/playground/games') && response.request().method() === 'POST');
  await page.getByLabel('Snake board size').selectOption('6');
  let previous = await (await created).json() as GameSnapshot;
  await expect(page.getByRole('img', { name: /6 by 6 Snake board/ })).toBeVisible();
  let questions: unknown;
  for (let i = 0; i < 8; i++) {
    const pending = page.waitForResponse(response => response.url().endsWith('/step') && response.request().method() === 'POST');
    await page.getByRole('button', { name: 'Step', exact: true }).click();
    const response = await pending; expect(response.ok()).toBe(true);
    const snapshot = await response.json() as GameSnapshot;
    if (snapshot.state.kind !== 'snake' || previous.state.kind !== 'snake') throw new Error('Expected Snake');
    const input = snapshot.result!.request.state as Record<string, unknown>;
    expect(Object.keys(input).sort()).toEqual(['game', 'size', 'coordinates', 'body_order', 'body', 'food', 'direction', 'score', 'steps', 'rules'].sort());
    expect(input.body).toEqual(previous.state.body); expect(input.food).toEqual(previous.state.food);
    expect(input.direction).toBe(previous.state.direction); expect(input.steps).toBe(previous.state.steps);
    expect(Object.keys(snapshot.result!.request.questions.move.criteria!)).toEqual(['UP', 'DOWN', 'LEFT', 'RIGHT']);
    expect(JSON.stringify(snapshot.result!.request.questions.move.criteria)).not.toMatch(/safe|best|collision|blocked|trap|progress|food|legal|cycle/i);
    if (questions) expect(snapshot.result!.request.questions).toEqual(questions);
    questions = snapshot.result!.request.questions;
    expect(snapshot.version).toBe(i + 1); expect(snapshot.state.steps).toBe(i + 1);
    expect(Object.keys(snapshot.result!.response.answers)).toEqual(['move', 'risk', 'food']);
    expect(snapshot.moves.at(-1)?.proposed).toBe(snapshot.result!.response.answers.move.choice);
    expect(snapshot.moves.at(-1)?.choice).toBe(snapshot.state.last?.executed);
    expect(snapshot.state.outcome).not.toBe('lost');
    expect(snapshot.state.body.length).toBe(4 + snapshot.state.score);
    previous = snapshot;
    await expect(page.locator('.latency')).not.toContainText('—');
  }
  await expect(page.getByRole('alert')).toHaveCount(0);
  await expect(page.locator('.answer-card')).toHaveCount(3);
  await expect(page.locator('.answer-id')).toContainText(['move', 'can_move', 'food_reachable']);
  await expect(page.locator('.response-footnote')).toContainText('typesafe/jev');
  await page.screenshot({ path: '../.local/output/playwright/snake-live.png', fullPage: true });
});
