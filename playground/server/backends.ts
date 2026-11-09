import type { DecisionRequest, DecisionResponse, Profile } from '../src/types.js';
import type { Endpoint } from './config.js';
import { HttpError, normalizeResponse, object } from './protocol.js';

export interface DecisionResult { response: DecisionResponse; telemetry: Record<string, unknown> }
export interface DecisionBackend {
  modelName: string;
  evaluate(request: DecisionRequest, signal: AbortSignal): Promise<DecisionResult>;
}
export interface BackendRegistry { profile(id: string): DecisionBackend }
export class RemoteBackend implements DecisionBackend {
  available = false; reason = 'Not connected';
  private openrouter: boolean;
  constructor(private endpoint: Endpoint, private key: string, private deadline: number, private fetcher: typeof fetch = fetch) {
    this.openrouter = new URL(endpoint.base_url).hostname === 'openrouter.ai';
  }
  get modelName() { return this.endpoint.model; }
  async probe() {
    if (this.openrouter) { this.available = !!this.key; this.reason = this.key ? '' : 'Set OPENROUTER_API_KEY in .env'; return; }
    try {
      const root = this.endpoint.base_url.replace(/\/v1\/systemone$/, '');
      const response = await this.fetcher(`${root}/v1/models`, { signal: AbortSignal.timeout(3000), redirect: 'error' });
      const raw = object(await response.json());
      this.available = response.ok && Array.isArray(raw.models) && raw.models.some(m => object(m).name === this.modelName);
      this.reason = this.available ? '' : 'Configured model is not served at this URL';
    } catch { this.available = false; this.reason = 'Model server unavailable'; }
  }
  async evaluate(request: DecisionRequest, signal: AbortSignal): Promise<DecisionResult> {
    if (this.openrouter && !this.key) throw new HttpError('Set OPENROUTER_API_KEY in .env', 401);
    const root = this.endpoint.base_url;
    const url = /\/(decisions|v1\/systemone)$/.test(root) ? root : `${root}/v1/systemone`;
    const telemetry: Record<string, unknown> = { attempts: 1, engine: this.openrouter ? 'openrouter-decisions' : 'remote-typesafe' };
    const started = performance.now();
    try {
      const result = await this.fetcher(url, {
        method: 'POST', redirect: 'error',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${this.openrouter ? this.key : 'local'}` },
        body: JSON.stringify({ ...request, model: this.modelName }),
        signal: AbortSignal.any([signal, AbortSignal.timeout(this.deadline * 1000)]),
      });
      const text = await result.text(); telemetry.raw_response_text = text;
      telemetry.server_timing = result.headers.get('server-timing');
      telemetry.request_id = result.headers.get('x-request-id');
      if (!result.ok) throw new HttpError(`Decision request failed (HTTP ${result.status})`, result.status);
      const raw = object(JSON.parse(text));
      const { response, derived, normalizedProbabilities } = normalizeResponse(raw, request, this.openrouter);
      telemetry.derived_scores = derived;
      telemetry.probability_normalizations = normalizedProbabilities;
      if (this.openrouter) { telemetry.provider = raw.provider; telemetry.gateway_request_id = raw.id; telemetry.cost = object(raw.usage).cost; }
      return { response, telemetry };
    } catch (error) {
      if (signal.aborted) throw new HttpError('Decision cancelled', 499, telemetry);
      throw new HttpError(error instanceof HttpError ? error.message : 'Decision request failed or exceeded its deadline', error instanceof HttpError ? error.status : 529, telemetry);
    } finally { telemetry.client_ms = performance.now() - started; }
  }
}
export class Registry implements BackendRegistry {
  private backends: Map<string, RemoteBackend>;
  constructor(catalog: Record<string, Endpoint>, key: string, deadline: number) {
    this.backends = new Map(Object.entries(catalog).map(([id, endpoint]) => [id, new RemoteBackend(endpoint, key, deadline)]));
  }
  profile(id: string) {
    const backend = this.backends.get(id);
    if (!backend) throw new HttpError(`Unknown backend: ${id}`);
    return backend;
  }
  async profiles(): Promise<Profile[]> {
    await Promise.all([...this.backends.values()].map(b => b.probe()));
    return [...this.backends.entries()].map(([id, b]) => ({ id, label: id === 'jev' ? 'Jev · OpenRouter' : id === 'gs1_local' ? `GS1 · ${b.modelName}` : id, model: b.modelName, available: b.available, reason: b.reason }));
  }
}
