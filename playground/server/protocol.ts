import type { Question, Answer, DecisionRequest, DecisionResponse } from '../src/types.js';

export class HttpError extends Error {
  constructor(message: string, public status = 422, public telemetry: Record<string, unknown> = {}) { super(message); }
}
export function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new HttpError('Expected an object');
  return value as Record<string, unknown>;
}
export function canonical(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  return `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([k, v]) => `${JSON.stringify(k)}:${canonical(v)}`).join(',')}}`;
}
export function candidateIds(question: Question): string[] {
  return question.type === 'choice' ? Object.keys(question.criteria) : question.type === 'noul' ? ['false', 'true'] : question.criteria.map((_, i) => String(i));
}
const sum = (v: number[]) => v.reduce((a, b) => a + b, 0);
const probability = (value: unknown): number => {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) throw new HttpError('Invalid probability');
  return value;
};
// TypeSafe formulas: https://docs.typesafe.ai/confidence
export function noulConfidence(p: number): number {
  return Math.abs(2 * p - 1);
}
export function choiceConfidence(probabilities: number[]): number {
  const n = probabilities.length;
  return n === 1 ? 1 : Math.max(0, Math.min(1, (Math.max(...probabilities) - 1 / n) / (1 - 1 / n)));
}
export function scoreConfidence(probabilities: number[]): number {
  const n = probabilities.length, m = probabilities.indexOf(Math.max(...probabilities));
  const spread = sum(probabilities.map((p, i) => p * Math.abs(i - m)));
  const evenSpread = sum(probabilities.map((_, i) => Math.abs(i - (n - 1) / 2))) / n;
  return Math.max(0, 1 - spread / evenSpread);
}
export function makeAnswer(question: Question, p: number[]): Answer {
  if (question.type === 'noul') return { type: 'noul', noul: p[1] };
  const keys = candidateIds(question), probabilities = Object.fromEntries(keys.map((k, i) => [k, p[i]]));
  const confidence = question.type === 'choice' ? choiceConfidence(p) : scoreConfidence(p);
  if (question.type === 'choice') return { type: 'choice', choice: keys[p.indexOf(Math.max(...p))], probabilities, confidence };
  return { type: 'score', score: sum(p.map((v, i) => v * i)), legend: Object.fromEntries(keys.map((k, i) => [k, question.criteria[i]])), probabilities, confidence };
}

/** Validate the exact request identities and numerical contract before actions apply. */
export function normalizeResponse(raw: unknown, request: DecisionRequest, openrouter = false) {
  const value = object(raw), source = object(value.answers), usage = object(value.usage);
  if (typeof value.model !== 'string' || !value.model) throw new HttpError('Missing resolved model');
  if (canonical(Object.keys(source).sort()) !== canonical(Object.keys(request.questions).sort())) throw new HttpError('Response question identities do not match');
  const answers: Record<string, Answer> = {}, derived: Record<string, unknown>[] = [], normalizedProbabilities: Record<string, unknown>[] = [];
  for (const [qid, question] of Object.entries(request.questions)) {
    const a = object(source[qid]);
    if (a.type !== question.type) throw new HttpError(`Primitive mismatch for ${qid}`);
    if (question.type === 'noul') { answers[qid] = { type: 'noul', noul: probability(a.noul) }; continue; }
    const map = object(a.probabilities), keys = candidateIds(question);
    if (canonical(Object.keys(map).sort()) !== canonical([...keys].sort())) throw new HttpError(`Candidate mismatch for ${qid}`);
    const reported = keys.map(k => probability(map[k])), total = sum(reported);
    let p = reported;
    if (Math.abs(total - 1) > 1e-6) {
      // Jev publishes each probability to two decimals, so a valid vector can
      // total 0.99 or 1.01. Accept only that bounded wire rounding, with an audit.
      const roundedJev = openrouter && value.model.startsWith('typesafe/jev-') && reported.every(v => Math.abs(v * 100 - Math.round(v * 100)) < 1e-8);
      if (!roundedJev || total <= 0 || Math.abs(total - 1) > keys.length * 0.005 + 1e-6) throw new HttpError('Probabilities must sum to one');
      p = reported.map(v => v / total);
      normalizedProbabilities.push({ question_id: qid, policy: 'jev-two-decimal-rounding-v1', reported_sum: total, reported_probabilities: { ...map }, served_probabilities: Object.fromEntries(keys.map((k, i) => [k, p[i]])) });
    }
    const confidence = probability(a.confidence), probabilities = Object.fromEntries(keys.map((k, i) => [k, p[i]]));
    if (question.type === 'choice') {
      // Preserve vendor tie choices across JSON floating-point representations
      // such as 0.27999999999999997 versus 0.28, without accepting a lower rank.
      if (typeof a.choice !== 'string' || !keys.includes(a.choice) || Math.max(...reported) - Number(map[a.choice]) > (openrouter ? 1e-12 : 0)) throw new HttpError('Choice must be an argmax');
      answers[qid] = { type: 'choice', choice: a.choice, probabilities, confidence };
    } else {
      const legend = Object.fromEntries(keys.map((k, i) => [k, question.criteria[i]]));
      if (canonical(a.legend) !== canonical(legend)) throw new HttpError('Score legend mismatch');
      if (typeof a.score !== 'number' || !Number.isFinite(a.score) || a.score < 0 || a.score > keys.length - 1) throw new HttpError('Invalid Score');
      const expected = sum(p.map((v, i) => i * v));
      if (Math.abs(a.score - expected) > 1e-6) {
        if (!openrouter) throw new HttpError('Score must equal the probability-weighted level');
        derived.push({ question_id: qid, reported_score: a.score, served_score: expected });
      }
      answers[qid] = { type: 'score', score: expected, legend, probabilities, confidence };
    }
  }
  const input = usage.input_tokens ?? usage.inputTokens, output = usage.output_tokens ?? usage.outputTokens;
  if (!Number.isSafeInteger(input) || !Number.isSafeInteger(output) || Number(input) < 0 || Number(output) < 0) throw new HttpError('Invalid usage counters');
  const response: DecisionResponse = { model: value.model, answers, usage: { input_tokens: Number(input), output_tokens: Number(output) } };
  return { response, derived, normalizedProbabilities };
}
