import type { DecisionRequest } from '../src/types.js';
import type { SnakeDirection, SnakeState } from '../src/lab-types.js';
import { HttpError } from './protocol.js';
import { Random } from './random.js';

type Cell = [number, number];
const vectors: Record<SnakeDirection, Cell> = { UP: [0, -1], DOWN: [0, 1], LEFT: [-1, 0], RIGHT: [1, 0] };
const directions = Object.keys(vectors) as SnakeDirection[];
const same = (a: Cell, b: Cell | null) => b !== null && a[0] === b[0] && a[1] === b[1];
const key = ([x, y]: Cell) => `${x},${y}`;

// A closed tour through every cell, used to initialize the body and by the
// optional execution guard. Its evaluations are never part of model input.
function cycle(size: number): Cell[] {
  const cells: Cell[] = Array.from({ length: size }, (_, y) => [0, y]);
  for (let y = size - 1; y >= 0; y--) {
    for (let i = 1; i < size; i++) cells.push([(size - y) % 2 ? i : size - i, y]);
  }
  return cells;
}

export class SnakeGame {
  readonly kind = 'snake';
  state: SnakeState;
  private rng: Random;
  private route: Cell[];
  private indices: Map<string, number>;

  constructor(size = 10, seed = 42, guarded = true) {
    if (![6, 8, 10, 12].includes(size) || !Number.isSafeInteger(seed) || typeof guarded !== 'boolean') throw new HttpError('Snake requires a 6, 8, 10, or 12 cell board, an integer seed, and a boolean guard setting.');
    this.rng = new Random(seed); this.route = cycle(size);
    this.indices = new Map(this.route.map((cell, i) => [key(cell), i]));
    const start = this.indices.get(key([size / 2, size / 2]))!;
    const body = Array.from({ length: 4 }, (_, i) => [...this.route[(start - i + size * size) % (size * size)]] as Cell);
    const direction = directions.find(d => same([body[1][0] + vectors[d][0], body[1][1] + vectors[d][1]], body[0]))!;
    this.state = { kind: 'snake', size, seed, body, food: null, direction, score: 0, steps: 0, finished: false, outcome: null, death_reason: null, guarded, interventions: 0, last: null };
    this.spawnFood();
  }

  private spawnFood() {
    const occupied = new Set(this.state.body.map(key));
    const empty = this.route.filter(cell => !occupied.has(key(cell)));
    this.state.food = empty.length ? [...this.rng.choice(empty)] : null;
  }
  private inside([x, y]: Cell) { return x >= 0 && y >= 0 && x < this.state.size && y < this.state.size; }
  private distance(from: Cell, to: Cell) { return (this.indices.get(key(to))! - this.indices.get(key(from))! + this.route.length) % this.route.length; }
  private ordered() {
    let span = 0;
    const body = this.state.body;
    for (let i = body.length - 1; i > 0; i--) {
      const distance = this.distance(body[i], body[i - 1]);
      if (!distance) return false;
      span += distance;
    }
    return span < this.route.length;
  }
  moves() {
    const { body, food, finished } = this.state;
    if (finished || !food) return [];
    const head = body[0], tail = body[body.length - 1], ordered = this.ordered();
    return directions.map(direction => {
      const [dx, dy] = vectors[direction], target: Cell = [head[0] + dx, head[1] + dy];
      const eats = same(target, food);
      const occupied = eats ? body : body.slice(0, -1);
      const collision: SnakeState['death_reason'] = !this.inside(target) ? 'wall' : same(target, body[1]) ? 'reverse' : occupied.some(cell => same(cell, target)) ? 'body' : null;
      const advance = this.inside(target) ? this.distance(head, target) : 0;
      const tailDistance = this.distance(head, tail);
      const safe = !collision && ordered && advance > 0 && advance <= this.distance(head, food) && (advance < tailDistance || advance === tailDistance && !eats);
      return { direction, target, eats, collision, legal: collision === null, safe, advance };
    });
  }
  request(model: string): DecisionRequest {
    const s = this.state;
    if (s.finished) throw new HttpError('This Snake game has ended.', 409);
    return {
      model,
      state: {
        game: 'Snake', size: s.size,
        coordinates: 'Zero-based [x,y]. x increases right; y increases down. Both coordinates range from 0 to size - 1.',
        body_order: 'head_to_tail', body: s.body.map(cell => [...cell]),
        food: s.food ? [...s.food] : null, direction: s.direction, score: s.score, steps: s.steps,
        rules: [
          'Move one cell per turn. Board edges are walls; there is no wrapping.',
          'Moving into the neck (the second body cell), another occupied body cell, or a wall ends the game.',
          'On a move without food, the tail vacates its cell at the same time as the head moves, so that cell may be entered unless it is the neck.',
          'Eating food grows the body by one cell without moving the tail. Filling the board wins.',
        ],
      },
      questions: {
        move: { type: 'choice', instructions: 'Choose the next direction from the board state. Collect food while avoiding collisions and leaving room for future moves.',
          criteria: { UP: 'Move one cell up: y decreases by 1.', DOWN: 'Move one cell down: y increases by 1.', LEFT: 'Move one cell left: x decreases by 1.', RIGHT: 'Move one cell right: x increases by 1.' } },
        risk: { type: 'noul', instructions: 'Does at least one direction allow the snake to move now without a collision? Apply the no-reverse and moving-tail rules.' },
        food: { type: 'noul', instructions: 'Is there a path from the head to the food through currently empty cells? Use orthogonal steps and treat all other body cells, including the current tail, as occupied. Do not simulate future body movement.' },
      },
    };
  }

  apply(choice: string, probabilities?: Record<string, number>): SnakeDirection {
    const s = this.state;
    if (s.finished) throw new HttpError('This Snake game has ended.', 409);
    const moves = this.moves(), proposed = moves.find(move => move.direction === choice);
    if (!proposed) throw new HttpError('The model selected an unknown Snake direction.');
    let executed = proposed;
    if (s.guarded && !proposed.safe) {
      const safe = moves.filter(move => move.safe);
      if (!safe.length) throw new HttpError('No certified safe route remains. Start a new Snake game.', 409);
      if (!probabilities || safe.some(move => !Number.isFinite(probabilities[move.direction]))) throw new HttpError('Snake guard requires model probabilities for every safe direction.');
      executed = safe.reduce((best, move) => probabilities[move.direction] > probabilities[best.direction] ? move : best);
    }
    const intervened = proposed !== executed;
    s.last = { proposed: proposed.direction, executed: executed.direction, intervened };
    s.interventions += Number(intervened); s.steps++; s.direction = executed.direction;
    if (executed.collision) {
      s.finished = true; s.outcome = 'lost'; s.death_reason = executed.collision;
    } else {
      s.body = [executed.target, ...s.body];
      if (executed.eats) {
        s.score++;
        if (s.body.length === s.size * s.size) { s.finished = true; s.outcome = 'won'; s.food = null; }
        else this.spawnFood();
      } else s.body.pop();
    }
    return executed.direction;
  }
}
