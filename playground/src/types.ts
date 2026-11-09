export interface Answer { type: 'choice' | 'noul' | 'score'; choice?: string; probabilities?: Record<string, number>; confidence?: number; noul?: number; score?: number; legend?: Record<string, unknown> }
export type Question =
  | { type: 'choice'; instructions: unknown; criteria: Record<string, unknown> }
  | { type: 'noul'; instructions: unknown; criteria?: Record<string, unknown> | null }
  | { type: 'score'; instructions: unknown; criteria: unknown[] };
export interface DecisionResponse { model: string; answers: Record<string, Answer>; usage: { input_tokens: number; output_tokens: number } }
export interface DecisionRequest { state: unknown; model: string; questions: Record<string, Question> }
export interface Profile { id: string; label: string; available: boolean; reason: string; model: string; artifact?: { status: string; resolved_model: string } }
