import { randomInt, randomUUID } from 'node:crypto';
import { mkdirSync, appendFileSync } from 'node:fs';
import { join } from 'node:path';
import type { DecisionRequest } from '../src/types.js';
import type { RunResult, GameSnapshot, GameMove } from '../src/lab-types.js';
import { parseDecisionRequest } from '../src/request.js';
import type { BackendRegistry } from './backends.js';
import { HttpError } from './protocol.js';
import { GoGame, MazeGame } from './games.js';
import { SnakeGame } from './snake.js';

export async function runDecision(registry: BackendRegistry, backend: string, request: DecisionRequest, signal: AbortSignal): Promise<RunResult> {
  const started = performance.now();
  const result = await registry.profile(backend).evaluate(request, signal);
  if (signal.aborted) throw new HttpError('Request cancelled.', 499);
  return { request, response: result.response,
    raw_response: typeof result.telemetry.raw_response_text === 'string' ? JSON.parse(result.telemetry.raw_response_text) : result.response,
    latency_ms: typeof result.telemetry.client_ms === 'number' ? result.telemetry.client_ms : performance.now() - started };
}

class GameSession {
  id = randomUUID().replaceAll('-', ''); version = 0; busy = false;
  result: RunResult | null = null; moves: GameMove[] = [];
  constructor(readonly game: GoGame | MazeGame | SnakeGame) {}
  snapshot(): GameSnapshot { return structuredClone({ id: this.id, version: this.version, state: this.game.state, result: this.result, moves: this.moves }); }
}

export class Lab {
  private games = new Map<string, GameSession>();
  constructor(private registry: BackendRegistry, private defaultBackend: string, private runsDir: string) {}
  private backend(value: unknown) {
    if (value !== undefined && typeof value !== 'string') throw new HttpError('Invalid backend.');
    const id = value as string | undefined ?? this.defaultBackend;
    this.registry.profile(id); return id;
  }
  async query(value: Record<string, unknown>, signal: AbortSignal) {
    const { backend: selected, ...input } = value, backend = this.backend(selected);
    let request: DecisionRequest;
    try { request = parseDecisionRequest(input, this.registry.profile(backend).modelName); }
    catch (error) { throw new HttpError((error as Error).message); }
    return runDecision(this.registry, backend, request, signal);
  }
  create(value: Record<string, unknown>) {
    if (Object.keys(value).some(k => !['kind', 'size', 'seed', 'guarded'].includes(k)) || !['go', 'maze', 'snake'].includes(String(value.kind))) throw new HttpError('Choose a Go, Maze, or Snake game.');
    if (value.guarded !== undefined && (value.kind !== 'snake' || typeof value.guarded !== 'boolean')) throw new HttpError('The guard setting must be a boolean and is only available for Snake.');
    if (value.size !== undefined && !Number.isInteger(value.size)) throw new HttpError('Size must be an integer.');
    if (value.seed !== undefined && !Number.isSafeInteger(value.seed)) throw new HttpError('Seed must be a safe integer.');
    if (this.games.size >= 64) {
      const oldest = [...this.games].find(([, g]) => !g.busy);
      if (!oldest) throw new HttpError('Too many active games.', 429);
      this.games.delete(oldest[0]);
    }
    const size = value.size as number | undefined, seed = value.seed as number | undefined ?? randomInt(2 ** 31);
    const game = value.kind === 'go' ? new GoGame(size) : value.kind === 'maze' ? new MazeGame(size, seed) : new SnakeGame(size, seed, value.guarded as boolean | undefined);
    const session = new GameSession(game);
    this.games.set(session.id, session); return session.snapshot();
  }
  get(id: string) { const game = this.games.get(id); if (!game) throw new HttpError('Game not found. Start a new game.', 404); return game; }
  async step(id: string, value: Record<string, unknown>, signal: AbortSignal) {
    const session = this.get(id), backend = this.backend(value.backend);
    if (Object.keys(value).some(k => !['version', 'backend'].includes(k))) throw new HttpError('Unexpected step option.');
    if (!Number.isInteger(value.version) || value.version !== session.version || session.busy) throw new HttpError('The game changed or already has a move in progress.', 409);
    if (session.game.state.finished) throw new HttpError('The game has ended. Start a new game.', 409);
    session.busy = true;
    try {
      const request = session.game.request(this.registry.profile(backend).modelName);
      const result = await runDecision(this.registry, backend, request, signal);
      const player = session.game.state.kind === 'go' ? session.game.state.turn === 1 ? 'Black' : 'White' : session.game.state.kind === 'maze' ? 'Explorer' : 'Snake';
      const choice = result.response.answers.move?.choice;
      if (typeof choice !== 'string') throw new HttpError('The model did not return a move.');
      if (signal.aborted) throw new HttpError('Move cancelled.', 499);
      let executed = choice;
      if (session.game instanceof SnakeGame) executed = session.game.apply(choice, result.response.answers.move.probabilities);
      else session.game.apply(choice);
      session.version++; session.result = result;
      const move = { number: session.version, player, choice: executed, ...(session.game.kind === 'snake' ? { proposed: choice } : {}), latency_ms: result.latency_ms };
      session.moves.push(move);
      const directory = join(this.runsDir, 'games', id); mkdirSync(directory, { recursive: true });
      appendFileSync(join(directory, 'moves.jsonl'), JSON.stringify({ ...move, version: session.version, ...result, state: session.game.state }) + '\n');
      return session.snapshot();
    } finally { session.busy = false; }
  }
}
