import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Registry, type BackendRegistry } from './backends.js';
import type { Profile } from '../src/types.js';
import { readCatalog, readKey, type ServerOptions } from './config.js';
import { HttpError, object } from './protocol.js';
import { Lab } from './lab.js';

const dist = resolve(fileURLToPath(new URL('../dist/', import.meta.url)));
const contentTypes: Record<string, string> = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon' };
async function body(request: IncomingMessage): Promise<Record<string, unknown>> {
  let size = 0; const chunks: Buffer[] = [];
  for await (const chunk of request) { size += chunk.length; if (size > 1024 * 1024) throw new HttpError('Request body too large', 413); chunks.push(chunk); }
  try { return object(JSON.parse(Buffer.concat(chunks).toString('utf8'))); } catch (error) { throw new HttpError(error instanceof HttpError ? error.message : 'Invalid JSON body'); }
}
function json(response: ServerResponse, value: unknown, status = 200) { response.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }); response.end(JSON.stringify(value)); }
export function createPlaygroundServer(settings: ServerOptions, registry: BackendRegistry & { profiles(): Promise<Profile[]> } = new Registry(readCatalog(settings.backendsConfig), readKey(settings.envFile), settings.deadlineSeconds)) {
  registry.profile(settings.defaultBackend);
  const lab = new Lab(registry, settings.defaultBackend, settings.runsDir);
  const pending = new Set<AbortController>();
  const server = createServer((request, response) => { void handle(request, response).catch(error => {
    if (response.headersSent) { response.end(); return; }
    json(response, { detail: error instanceof Error ? error.message : 'Server error' }, error instanceof HttpError ? error.status : 500);
  }); });
  async function handle(request: IncomingMessage, response: ServerResponse) {
    const url = new URL(request.url ?? '/', 'http://localhost'), path = url.pathname, method = request.method;
    if (method === 'GET' && path === '/health') return json(response, { status: 'ok', service: 'playground' });
    if (method === 'GET' && path === '/playground/backends') return json(response, { backends: await registry.profiles(), default: settings.defaultBackend });
    if (method === 'POST' && path === '/playground/games') return json(response, lab.create(await body(request)));
    const gameRoute = path.match(/^\/playground\/games\/([a-f0-9]{32})(?:\/(step))?$/);
    if (method === 'GET' && gameRoute && !gameRoute[2]) return json(response, lab.get(gameRoute[1]).snapshot());
    if (method === 'POST' && (path === '/playground/query' || gameRoute?.[2] === 'step')) {
      const controller = new AbortController(); pending.add(controller);
      const cancel = () => { if (!response.writableEnded) controller.abort(); };
      response.once('close', cancel);
      try {
        const value = await body(request);
        const result = path === '/playground/query' ? await lab.query(value, controller.signal) : await lab.step(gameRoute![1], value, controller.signal);
        if (!controller.signal.aborted) return json(response, result);
        return;
      } finally { response.off('close', cancel); pending.delete(controller); }
    }
    if (method === 'GET' && path === '/docs') return json(response, { playground: { 'POST /playground/query': '{state, questions, model?, backend?}', 'POST /playground/games': '{kind: go | maze | snake, size?, seed?}', 'GET /playground/games/:id': 'Authoritative game snapshot', 'POST /playground/games/:id/step': '{version, backend?}' }, primitives: 'https://docs.typesafe.ai/primitives', note: 'Credentials stay on the server. Query results contain the untouched provider JSON and model round-trip latency. Games apply only legal model-selected moves.' });
    if (method === 'GET' && (path === '/' || path.startsWith('/assets/'))) {
      const file = resolve(dist, path === '/' ? 'index.html' : '.' + decodeURIComponent(path));
      if (!file.startsWith(dist + sep)) throw new HttpError('Not found', 404);
      try { if (!(await stat(file)).isFile()) throw new Error(); response.writeHead(200, { 'Content-Type': contentTypes[extname(file)] ?? 'application/octet-stream' }); response.end(await readFile(file)); return; }
      catch { throw new HttpError('Build the playground with npm run build, or use the Vite development server.', 404); }
    }
    throw new HttpError('Not found', 404);
  }
  async function close() {
    for (const controller of pending) controller.abort();
    await new Promise<void>((resolveClose, reject) => server.close(error => error ? reject(error) : resolveClose()));
  }
  return { server, registry, close };
}
