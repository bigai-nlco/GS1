import type { DecisionRequest, Question } from './types.js';

const record = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const content = (v: unknown) => typeof v === 'string' || (v !== null && typeof v === 'object');

/** TypeSafe's public primitives contract. Nested JSON is preserved verbatim. */
export function parseDecisionRequest(value: unknown, model: string): DecisionRequest {
  if (!record(value)) throw new Error('Request must be a JSON object.');
  if (Object.keys(value).some(k => !['state', 'questions', 'model'].includes(k))) throw new Error('Request accepts state, questions, and an optional model.');
  if (!content(value.state)) throw new Error('State must be text, a JSON object, or an array.');
  if (value.model !== undefined && value.model !== model) throw new Error(`This playground is configured for ${model}. Use that model or omit the model field.`);
  if (!record(value.questions) || !Object.keys(value.questions).length) throw new Error('Questions must be a nonempty JSON object keyed by question ID.');
  for (const [id, q] of Object.entries(value.questions)) {
    const fail = (message: string): never => { throw new Error(`${id || '(empty ID)'}: ${message}`); };
    if (!id || !record(q)) fail('each question needs an ID and an object.');
    const question = q as Record<string, unknown>;
    if (Object.keys(question).some(k => !['type', 'instructions', 'criteria'].includes(k))) fail('only type, instructions, and criteria are supported.');
    if (!content(question.instructions)) fail('instructions must be text, an object, or an array.');
    if (question.type === 'choice') {
      if (!record(question.criteria) || !Object.keys(question.criteria).length || Object.keys(question.criteria).length > 255) fail('Choice needs 1–255 named criteria.');
      if (Object.entries(question.criteria as Record<string, unknown>).some(([key, v]) => !key || !(v === null || content(v)))) fail('Choice criteria must be text, objects, arrays, or null.');
    } else if (question.type === 'score') {
      if (!Array.isArray(question.criteria) || question.criteria.length < 2 || question.criteria.length > 10 || !question.criteria.every(content)) fail('Score needs an ordered array of 2–10 text, object, or array criteria.');
    } else if (question.type === 'noul') {
      if (question.criteria !== undefined && question.criteria !== null && (!record(question.criteria) || Object.entries(question.criteria).some(([key, v]) => !['true', 'false'].includes(key) || !content(v)))) fail('Noul criteria may describe true and false with text, objects, or arrays.');
    } else fail('type must be choice, score, or noul.');
  }
  return { state: value.state, questions: value.questions as Record<string, Question>, model };
}

export function parseState(text: string, format: 'text' | 'json'): unknown {
  if (format === 'text') return text;
  try { return JSON.parse(text); } catch { throw new Error('State contains invalid JSON.'); }
}
