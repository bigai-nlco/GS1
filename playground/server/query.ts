import type { DecisionRequest, RunResult } from '../src/types.js';
import { parseDecisionRequest } from '../src/request.js';
import type { BackendRegistry } from './backends.js';
import { HttpError } from './protocol.js';

export async function runQuery(registry: BackendRegistry, defaultBackend: string, value: Record<string, unknown>, signal: AbortSignal): Promise<RunResult> {
  const { backend: selected, ...input } = value;
  if (selected !== undefined && typeof selected !== 'string') throw new HttpError('Invalid backend.');
  const backend = registry.profile(selected as string | undefined ?? defaultBackend);
  let request: DecisionRequest;
  try { request = parseDecisionRequest(input, backend.modelName); }
  catch (error) { throw new HttpError((error as Error).message); }
  const started = performance.now();
  const result = await backend.evaluate(request, signal);
  if (signal.aborted) throw new HttpError('Request cancelled.', 499);
  return { request, response: result.response,
    raw_response: typeof result.telemetry.raw_response_text === 'string' ? JSON.parse(result.telemetry.raw_response_text) : result.response,
    latency_ms: typeof result.telemetry.client_ms === 'number' ? result.telemetry.client_ms : performance.now() - started };
}
