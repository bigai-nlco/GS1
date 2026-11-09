import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { parse as parseYaml } from 'yaml';
import { parse as parseEnv } from 'dotenv';
import { object } from './protocol.js';

export const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
export interface Endpoint { base_url: string; model: string }
export interface ServerOptions {
  host: string; port: number; runsDir: string; backendsConfig: string; envFile: string;
  defaultBackend: string; deadlineSeconds: number; realtimeTickSeconds: number;
}
export const defaults: ServerOptions = {
  host: '127.0.0.1', port: 3000, runsDir: resolve(REPO_ROOT, 'runs'),
  backendsConfig: resolve(REPO_ROOT, 'playground/backends.yaml'), envFile: resolve(REPO_ROOT, '.env'),
  defaultBackend: 'gs1_local', deadlineSeconds: 30, realtimeTickSeconds: 2,
};
export function readCatalog(path: string): Record<string, Endpoint> {
  const data = object(parseYaml(readFileSync(path, 'utf8')));
  if (Object.keys(data).some(k => k !== 'backends')) throw new Error('backends.yaml only accepts the endpoint catalog');
  const profiles = object(data.backends);
  if (!Object.keys(profiles).length) throw new Error('Empty backend catalog');
  return Object.fromEntries(Object.entries(profiles).map(([id, raw]) => {
    const profile = object(raw);
    if (Object.keys(profile).sort().join(',') !== 'base_url,model' || typeof profile.base_url !== 'string' || typeof profile.model !== 'string' || !profile.model) throw new Error(`Backend ${id} requires only base_url and model`);
    const url = new URL(profile.base_url);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error(`Invalid serving URL for ${id}`);
    return [id, { base_url: profile.base_url.replace(/\/$/, ''), model: profile.model }];
  }));
}
export function readKey(envFile: string): string {
  const values = existsSync(envFile) ? parseEnv(readFileSync(envFile)) : {};
  return process.env.TYPESAFE_API_KEY ?? values.TYPESAFE_API_KEY ?? '';
}
export function options(argv = process.argv.slice(2)): ServerOptions {
  const { values } = parseArgs({ args: argv, options: {
    host: { type: 'string' }, port: { type: 'string' }, 'runs-dir': { type: 'string' },
    'backends-config': { type: 'string' }, 'env-file': { type: 'string' },
    'default-backend': { type: 'string' }, 'deadline-seconds': { type: 'string' },
    'realtime-tick-seconds': { type: 'string' },
  } });
  const result = { ...defaults };
  if (values.host) result.host = values.host;
  if (values.port) result.port = Number(values.port);
  if (values['runs-dir']) result.runsDir = resolve(values['runs-dir']);
  if (values['backends-config']) result.backendsConfig = resolve(values['backends-config']);
  if (values['env-file']) result.envFile = resolve(values['env-file']);
  if (values['default-backend']) result.defaultBackend = values['default-backend'];
  if (values['deadline-seconds']) result.deadlineSeconds = Number(values['deadline-seconds']);
  if (values['realtime-tick-seconds']) result.realtimeTickSeconds = Number(values['realtime-tick-seconds']);
  if (!Number.isInteger(result.port) || result.port < 0 || result.port > 65535) throw new Error('Invalid port');
  for (const duration of [result.deadlineSeconds, result.realtimeTickSeconds]) if (!Number.isFinite(duration) || duration <= 0) throw new Error('Invalid deadline');
  return result;
}
