import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }, testInfo) => {
  if (testInfo.titlePath.some(title => title === 'live TypeSafe / Jev')) return;
  await page.route('**/playground/backends', route => route.fulfill({ json: {
    default: 'gs1_local',
    backends: [{ id: 'gs1_local', label: 'GS1 · GS1', model: 'GS1', available: true, reason: '' }, { id: 'jev', label: 'Jev · TypeSafe', model: 'jev-latest', available: true, reason: '' }],
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
    const response = { model: request.model, answers, usage: { input_tokens: 253, output_tokens: 0 } };
    await route.fulfill({ json: { request, response, raw_response: { ...response, provider: 'TypeSafe', id: 'raw-provider-id', usage: { ...response.usage, cost: .001 } }, latency_ms: 670.2 } });
  });
  await page.goto('/');
  await expect(page.getByRole('link', { name: 'GS1 Playground', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Query', exact: true })).toBeVisible();
  await expect(page.getByRole('navigation', { name: 'Playgrounds' })).toHaveCount(0);
  await page.getByRole('button', { name: 'Run query', exact: true }).click();
  await expect(page.locator('.query-result-row')).toHaveCount(1);
  await expect(page.locator('.latency')).toHaveText('670 ms');
  await page.screenshot({ path: '../.local/output/playwright/query-playground.png', fullPage: true });
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

test('Query switches backends and discards a cancelled response', async ({ page }) => {
  let release!: () => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  const requests: Record<string, unknown>[] = [];
  await page.route('**/playground/query', async route => {
    const request = route.request().postDataJSON(); requests.push(request);
    const first = requests.length === 1;
    if (first) await held;
    const response = { model: request.model, answers: { department: { type: 'choice', choice: 'returns', probabilities: { returns: 1, billing: 0, shipping: 0 }, confidence: 1 } }, usage: { input_tokens: 10, output_tokens: 0 } };
    try { await route.fulfill({ json: { request, response, raw_response: response, latency_ms: 15 } }); }
    catch (error) { if (!first) throw error; /* The first browser request was cancelled. */ }
  });
  try {
    await page.goto('/');
    await page.getByLabel('Decision model').selectOption('jev');
    await expect(page.locator('.connection')).toContainText('Connected to TypeSafe');
    await expect(page.locator('.model-control code')).toHaveText('jev-latest');
    await page.getByRole('button', { name: 'Run query', exact: true }).click();
    await expect.poll(() => requests.length).toBe(1);
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Run query', exact: true })).toBeEnabled();
    await page.getByLabel('Decision model').selectOption('gs1_local');
    await expect(page.locator('.connection')).toContainText('Connected to model');
    await page.getByRole('button', { name: 'Run query', exact: true }).click();
    await expect(page.locator('.response-footnote')).toContainText('GS1');
    release();
    await expect(page.locator('.response-footnote')).toContainText('GS1');
    await expect(page.getByRole('alert')).toHaveCount(0);
    expect(requests.map(request => request.model)).toEqual(['jev-latest', 'GS1']);
  } finally { release(); }
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
});
