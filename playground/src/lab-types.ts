import type { DecisionRequest, DecisionResponse } from './types.js';

export type GameKind = 'go' | 'maze' | 'snake';
export type Tab = 'query' | GameKind;
export type Stone = 0 | 1 | 2;
export type GoOwnership = 'black' | 'white' | 'neutral' | 'unsettled';
export interface GoSettlement {
  ownership: GoOwnership[];
  neutral: number;
  unsettled: number;
}
export interface GoState {
  kind: 'go'; size: number; board: Stone[]; turn: 1 | 2; move: number;
  captures: { black: number; white: number }; komi: number;
  last: number | null; finished: boolean; winner: 'black' | 'white' | null;
  score: { black: number; white: number };
  settlement: GoSettlement;
}
export interface MazeState {
  kind: 'maze'; size: number; seed: number; grid: number[][];
  position: [number, number]; goal: [number, number]; trail: [number, number][];
  visits: number[][]; steps: number; finished: boolean;
}
export type SnakeDirection = 'UP' | 'DOWN' | 'LEFT' | 'RIGHT';
export interface SnakeState {
  kind: 'snake'; size: number; seed: number; body: [number, number][];
  food: [number, number] | null; direction: SnakeDirection; score: number; steps: number;
  finished: boolean; outcome: 'won' | 'lost' | null;
  death_reason: 'wall' | 'body' | 'reverse' | null;
  guarded: boolean; interventions: number;
  last: { proposed: SnakeDirection; executed: SnakeDirection; intervened: boolean } | null;
}
export interface RunResult {
  request: DecisionRequest; response: DecisionResponse;
  raw_response: DecisionResponse & Record<string, unknown>;
  latency_ms: number;
}
export interface GameMove { number: number; player: string; choice: string; proposed?: string; latency_ms: number }
export interface GameSnapshot {
  id: string; version: number; state: GoState | MazeState | SnakeState;
  result: RunResult | null; moves: GameMove[];
}
