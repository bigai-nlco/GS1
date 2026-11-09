import { defineConfig } from '@playwright/test';
import { existsSync } from 'node:fs';

export default defineConfig({
  testDir: './tests',
  workers: 1,
  timeout: 45_000,
  webServer: process.env.GS1_URL ? undefined : {
    command: 'npm run server',
    url: 'http://127.0.0.1:3000',
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
  use: {
    baseURL: process.env.GS1_URL ?? 'http://127.0.0.1:3000',
    viewport: { width: 1600, height: 1100 },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
    launchOptions: {
      executablePath: process.env.CHROME_PATH ?? (existsSync('/usr/bin/google-chrome') ? '/usr/bin/google-chrome' : undefined),
      args: ['--enable-unsafe-swiftshader'],
    },
  },
});
