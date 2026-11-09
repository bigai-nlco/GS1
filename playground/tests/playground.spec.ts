import { expect, test } from '@playwright/test';
import { GoGame, goSettlement } from '../server/games';
import type { GameSnapshot } from '../src/lab-types';

test.beforeEach(async ({ page }, testInfo) => {
  if (testInfo.titlePath.some(title => title === 'live TypeSafe / Jev')) return;
  await page.route('**/playground/backends', route => route.fulfill({ json: {
    default: 'gs1_local',
    backends: [{ id: 'gs1_local', label: 'GS1 · GS1', model: 'GS1', available: true, reason: '' }],
  } }));
});

test('query editors preserve structured inputs, render each primitive, and retain untouched JSON with latency', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  const requests: Record<string, unknown>[] = [];
  await page.route('**/playground/query', async route => {
    const request = route.request().postDataJSON(); requests.push(request);
    const answers = Object.fromEntries(Object.entries(request.questions).map(([id, raw]) => {
      const q = raw as { type: string; criteria: Record<string, unknown> | unknown[] };
      if (q.type === 'noul') return [id, { type: 'noul', noul: .85 }];
      const keys = Object.keys(q.criteria), probabilities = Object.fromEntries(keys.map((k, i) => [k, id === 'years_of_experience' ? i === 3 ? .98 : i === 2 ? .02 : 0 : i === 0 ? 1 : 0]));
      return [id, q.type === 'choice' ? { type: 'choice', choice: keys[0], probabilities, confidence: 1 } : { type: 'score', score: id === 'years_of_experience' ? 2.98 : 0, legend: Object.fromEntries(keys.map((k, i) => [k, (q.criteria as unknown[])[i]])), probabilities, confidence: 1 }];
    }));
    const response = { model: 'jev-fixture', answers, usage: { input_tokens: 253, output_tokens: 40 } };
    await route.fulfill({ json: { request, response, raw_response: { ...response, provider: 'TypeSafe', id: 'raw-provider-id', usage: { ...response.usage, cost: .001 } }, latency_ms: 670.2 } });
  });
  await page.goto('/');
  await expect(page.getByRole('link', { name: 'GS1 Playground', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Run query', exact: true }).click();
  await expect(page.locator('.query-result-row')).toHaveCount(1);
  await expect(page.locator('.latency')).toHaveText('670 ms');
  await page.getByRole('tab', { name: 'JSON', exact: true }).click();
  await expect(page.getByLabel('Raw JSON response', { exact: true })).toContainText('raw-provider-id');
  await expect(page.getByLabel('Raw JSON response', { exact: true })).toContainText('"cost": 0.001');
  await expect(page.locator('.latency')).toHaveText('670 ms');
  await page.getByRole('button', { name: 'Resume screening', exact: true }).click();
  await page.getByRole('button', { name: 'Full request', exact: true }).click();
  await page.getByRole('button', { name: 'Run query', exact: true }).click();
  await expect.poll(() => requests.length).toBe(2);
  expect((requests[1].state as { resume: string }).resume).toContain('SASHA BERNOULLI\nSan Francisco, CA');
  const q = requests[1].questions as Record<string, { instructions: unknown }>;
  expect(Object.keys(q)).toHaveLength(7);
  expect(q.years_of_experience.instructions).toEqual({ question: 'How many years of professional experience does the candidate have, as of today?', today: 'September 15, 2026' });
  await expect(page.getByLabel('Raw JSON response', { exact: true })).toContainText('"score": 2.98');
  await page.getByRole('tab', { name: 'Overview', exact: true }).click();
  await expect(page.locator('.query-result-row')).toHaveCount(7);
  const years = page.locator('[data-question-id="years_of_experience"]');
  await expect(years.locator('.query-instructions')).toContainText('September 15, 2026');
  await expect(years.locator('.query-instructions')).not.toContainText('"question":');
  await expect(years.locator('.query-score-value')).toHaveText('2.98 of 5');
  await expect(years.locator('.query-primitive')).toContainText('6 levels · 0~5');
  await expect(years.locator('.query-level-description')).toHaveText(['None', '2 years', '4 years', '6 years', '8 years', '10+ years']);
  await expect(years.getByRole('meter', { name: 'years_of_experience expected level' })).toHaveAttribute('aria-valuenow', '2.98');
  await expect(years.locator('.query-score-track')).toHaveCount(6);
  const axis = await years.evaluate(element => {
    const ticks = [...element.querySelectorAll('.query-score-tick')].map(tick => { const r = tick.getBoundingClientRect(); return r.top + r.height / 2; });
    const marker = element.querySelector('.query-score-marker')!.getBoundingClientRect();
    return { expected: ticks[2] + .98 * (ticks[3] - ticks[2]), actual: marker.top + marker.height / 2 };
  });
  expect(Math.abs(axis.actual - axis.expected)).toBeLessThan(1);
  await expect(page.locator('.query-row-toggle, .query-results-heading')).toHaveCount(0);
  await expect(page.getByRole('meter', { name: 'mentorship_demonstrated probability of true' })).toHaveAttribute('aria-valuenow', '85');
  const profile = page.locator('[data-question-id="primary_talent_profile"]');
  await expect(profile.locator('.query-option')).toHaveCount(3);
  await page.getByRole('button', { name: 'Show all primary_talent_profile options', exact: true }).click();
  await expect(profile.locator('.query-option')).toHaveCount(10);
  await page.getByRole('button', { name: 'Show top 3 primary_talent_profile options', exact: true }).click();
  await page.locator('.response-scroll').evaluate(element => { element.scrollTop = 0; });
  await page.screenshot({ path: '../.local/output/playwright/query-primitive-results.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('.response-scroll').evaluate(element => { element.scrollTop = 0; });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: '../.local/output/playwright/query-primitive-mobile.png', fullPage: true });
  await expect(page.getByRole('button', { name: 'Run query', exact: true })).toBeEnabled();
  await page.getByLabel('Request native TypeSafe JSON').fill('{invalid');
  await page.getByRole('button', { name: 'Run query', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('invalid JSON'); expect(requests).toHaveLength(2);
  expect(errors).toEqual([]);
});

test('Go and Maze controls create real boards and regenerate a new maze', async ({ page }) => {
  await page.goto('/'); await page.getByRole('button', { name: 'Go 02', exact: true }).click();
  await expect(page.getByRole('img', { name: /5 by 5 Go board/ })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Run game', exact: true })).toBeEnabled();
  await page.getByLabel('Go board size').selectOption('7');
  await expect(page.getByRole('img', { name: /7 by 7 Go board/ })).toBeVisible();
  await page.getByRole('button', { name: 'Maze 03', exact: true }).click();
  await expect(page.getByRole('img', { name: /Maze. Explorer/ })).toBeVisible();
  const seed = await page.locator('.maze-legend code').textContent();
  await page.getByRole('button', { name: 'New maze', exact: true }).click();
  await expect(page.locator('.maze-legend code')).not.toHaveText(seed!);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('Go finishes after a placement settles owned and neutral points', async ({ page }) => {
  const game = new GoGame();
  game.state.board = ['BB.WW', '.B.W.', 'B..WW', '.B.W.', 'BB.WW'].join('').split('').map(s => s === 'B' ? 1 : s === 'W' ? 2 : 0);
  game.state.settlement = goSettlement(game.state.board, 5);
  const snapshot: GameSnapshot = { id: 'settlement-fixture', version: 0, state: game.state, moves: [], result: null };
  let calls = 0;
  await page.route('**/playground/games', route => route.fulfill({ json: snapshot }));
  await page.route('**/playground/games/settlement-fixture/step', async route => {
    calls++;
    const choice = 'B3', player = game.state.turn === 1 ? 'Black' : 'White';
    game.apply(choice); snapshot.version++;
    snapshot.moves.push({ number: snapshot.version, player, choice, latency_ms: 1 });
    await route.fulfill({ json: snapshot });
  });
  await page.goto('/'); await page.getByRole('button', { name: 'Go 02', exact: true }).click();
  await expect(page.getByText(/Every move places a stone; passing is disabled/)).toBeVisible();
  await page.getByRole('button', { name: 'Run game', exact: true }).click();
  await expect(page.locator('.completion')).toContainText('White wins by 0.5 points.');
  await expect(page.locator('.go-settlement')).toContainText('25 / 25 points settled');
  await expect(page.locator('.go-settlement')).toContainText('5 neutral · 0 unresolved');
  await expect(page.locator('.completion')).toContainText('Every point is owned or neutral');
  await expect(page.getByRole('button', { name: 'Run game', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Step', exact: true })).toBeDisabled();
  expect(calls).toBe(1);
  await page.screenshot({ path: '../.local/output/playwright/go-settled.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test.describe('live TypeSafe / Jev', () => {
  test.skip(process.env.GS1_LIVE !== '1', 'Set GS1_LIVE=1 to use the configured server-side key.');
  test('all query examples support native primitives and raw provider responses', async ({ page }) => {
    test.setTimeout(180_000);
    await page.goto('/'); await page.getByLabel('Decision model').selectOption('jev');
    for (const name of ['Multiple Choice', 'Score', 'Resume screening']) {
      await page.getByRole('button', { name, exact: true }).click();
      await page.getByRole('button', { name: 'Run query', exact: true }).click();
      await expect(page.locator('.response-footnote')).toContainText('jev-', { timeout: 35_000 });
      await expect(page.getByRole('alert')).toHaveCount(0);
      await expect(page.locator('.latency')).not.toHaveText('— ms');
      await page.getByRole('tab', { name: 'JSON', exact: true }).click();
      await expect(page.getByLabel('Raw JSON response', { exact: true })).toContainText('"model": "jev-');
      await expect(page.getByLabel('Raw JSON response', { exact: true })).toContainText('"usage"');
      await page.getByRole('tab', { name: 'Overview', exact: true }).click();
      if (name === 'Multiple Choice') await page.screenshot({ path: '../.local/output/playwright/redesigned-query.png', fullPage: true });
      if (name === 'Resume screening') await page.screenshot({ path: '../.local/output/playwright/query-primitive-live.png', fullPage: true });
    }
  });
  test('Jev plays both Go sides with ownership-based stopping', async ({ page }) => {
    test.setTimeout(180_000);
    await page.goto('/'); await page.getByLabel('Decision model').selectOption('jev'); await page.getByRole('button', { name: 'Go 02', exact: true }).click();
    for (let i = 0; i < 8; i++) {
      const response = page.waitForResponse(r => r.url().endsWith('/step') && r.request().method() === 'POST');
      await page.getByRole('button', { name: 'Step', exact: true }).click();
      const result = await response; expect(result.ok()).toBe(true);
      const snapshot = await result.json() as GameSnapshot;
      expect(snapshot.state.kind).toBe('go');
      if (snapshot.state.kind !== 'go') throw new Error('Expected Go');
      expect(snapshot.moves.at(-1)?.player).toBe(i % 2 === 0 ? 'Black' : 'White');
      expect(snapshot.moves.at(-1)?.choice).not.toBe('pass');
      expect(Object.keys(snapshot.result!.request.questions.move.criteria!)).not.toContain('pass');
      expect(snapshot.state.finished).toBe(snapshot.state.settlement.unsettled === 0);
      expect(snapshot.state.settlement).toEqual(goSettlement(snapshot.state.board, snapshot.state.size));
      const input = snapshot.result!.request.state as Record<string, unknown>;
      const ownership = input.ownership as string[][];
      expect(ownership).toHaveLength(5);
      expect(ownership.flat()).toHaveLength(25);
      expect(ownership.flat().every(p => ['black', 'white', 'neutral', 'unsettled'].includes(p))).toBe(true);
      if (snapshot.state.finished) {
        await expect(page.locator('.completion')).toContainText('wins by');
        await expect(page.getByRole('button', { name: 'Run game', exact: true })).toBeDisabled();
        break;
      }
      expect(snapshot.state.winner).toBeNull();
      await expect(page.locator('.completion')).toHaveCount(0);
      await expect(page.getByRole('button', { name: 'Run game', exact: true })).toBeEnabled();
    }
    await expect(page.locator('.response-footnote')).toContainText('jev-');
    await expect(page.getByRole('alert')).toHaveCount(0);
    await page.screenshot({ path: '../.local/output/playwright/redesigned-go.png', fullPage: true });
  });
  test('Jev navigates a generated maze to its exit and stops', async ({ page }) => {
    test.setTimeout(150_000);
    await page.goto('/'); await page.getByLabel('Decision model').selectOption('jev'); await page.getByRole('button', { name: 'Maze 03', exact: true }).click();
    await page.getByLabel('Maze size').selectOption('5');
    await page.getByRole('button', { name: 'Run maze', exact: true }).click();
    await expect(page.locator('.completion')).toContainText('Exit reached', { timeout: 120_000 });
    await expect(page.getByRole('button', { name: 'Run maze', exact: true })).toBeDisabled();
    await expect(page.locator('.response-footnote')).toContainText('jev-');
    await expect(page.getByRole('alert')).toHaveCount(0);
    await page.screenshot({ path: '../.local/output/playwright/redesigned-maze.png', fullPage: true });
  });
});
