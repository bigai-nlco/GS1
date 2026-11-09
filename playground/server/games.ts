import type { DecisionRequest } from '../src/types.js';
import type { GoOwnership, GoSettlement, GoState, MazeState, Stone } from '../src/lab-types.js';
import { Random } from './random.js';
import { HttpError } from './protocol.js';

const neighbors = (i: number, size: number) => [i % size > 0 ? i - 1 : -1, i % size < size - 1 ? i + 1 : -1, i >= size ? i - size : -1, i < size * (size - 1) ? i + size : -1].filter(n => n >= 0);
const positionKey = (board: Stone[]) => board.join('');
export const pointName = (i: number, size: number) => `${'ABCDEFGHJ'[i % size]}${size - Math.floor(i / size)}`;
function group(board: Stone[], start: number, size: number) {
  const stones = new Set([start]), liberties = new Set<number>(), queue = [start];
  for (let i = 0; i < queue.length; i++) for (const n of neighbors(queue[i], size)) {
    if (!board[n]) liberties.add(n);
    else if (board[n] === board[start] && !stones.has(n)) { stones.add(n); queue.push(n); }
  }
  return { stones, liberties };
}
function placement(board: Stone[], index: number, color: 1 | 2, size: number) {
  if (index < 0 || index >= board.length || board[index]) return null;
  const next = [...board]; next[index] = color;
  let captured = 0;
  for (const n of neighbors(index, size)) if (next[n] && next[n] !== color) {
    const enemy = group(next, n, size);
    if (!enemy.liberties.size) for (const stone of enemy.stones) { next[stone] = 0; captured++; }
  }
  const own = group(next, index, size);
  return own.liberties.size ? { board: next, captured, liberties: own.liberties.size } : null;
}
export function areaScore(board: Stone[], size: number, komi = 0.5) {
  const score = { black: board.filter(s => s === 1).length, white: board.filter(s => s === 2).length + komi };
  const visited = new Set<number>();
  for (let start = 0; start < board.length; start++) {
    if (board[start] || visited.has(start)) continue;
    const region = [start], borders = new Set<Stone>(); visited.add(start);
    for (let i = 0; i < region.length; i++) for (const n of neighbors(region[i], size)) {
      if (board[n]) borders.add(board[n]);
      else if (!visited.has(n)) { visited.add(n); region.push(n); }
    }
    if (borders.size === 1) score[borders.has(1) ? 'black' : 'white'] += region.length;
  }
  return score;
}

/** Conservative Benson pass-alive analysis for suicide-forbidden Go.
 * A region supplies one sure liberty, regardless of how many points it has.
 * See playground/README.md for the algorithm, sources, and seki limitations.
 */
function passAliveOwnership(board: Stone[], size: number): Stone[] {
  const ownership = Array<Stone>(board.length).fill(0);
  for (const color of [1, 2] as const) {
    const chains: ReturnType<typeof group>[] = [], chainAt = Array<number>(board.length).fill(-1);
    for (let start = 0; start < board.length; start++) {
      if (board[start] !== color || chainAt[start] !== -1) continue;
      const chain = group(board, start, size), id = chains.length;
      chains.push(chain);
      for (const point of chain.stones) chainAt[point] = id;
    }
    const regions: { points: number[]; boundary: number[]; vitalFor: number[] }[] = [], visited = new Set<number>();
    for (let start = 0; start < board.length; start++) {
      if (board[start] === color || visited.has(start)) continue;
      // Opposing stones remain part of a region: an invasion does not split an eye.
      const points = [start], boundary = new Set<number>(); visited.add(start);
      for (let i = 0; i < points.length; i++) for (const n of neighbors(points[i], size)) {
        if (board[n] === color) boundary.add(chainAt[n]);
        else if (!visited.has(n)) { visited.add(n); points.push(n); }
      }
      const empty = points.filter(p => !board[p]);
      if (!empty.length || !boundary.size) continue;
      regions.push({ points, boundary: [...boundary], vitalFor: [...boundary].filter(id => empty.every(p => chains[id].liberties.has(p))) });
    }
    let alive = new Set(chains.map((_, id) => id));
    let enclosed = regions;
    for (;;) {
      enclosed = enclosed.filter(r => r.boundary.every(id => alive.has(id)));
      const next = new Set([...alive].filter(id => enclosed.filter(r => r.vitalFor.includes(id)).length >= 2));
      if (next.size === alive.size) break;
      alive = next;
    }
    for (const id of alive) for (const p of chains[id].stones) ownership[p] = color;
    for (const region of enclosed) {
      // An invader needs two internal eye points to live here. Larger regions
      // remain unresolved even when their enclosing stones are pass-alive.
      const internal = region.points.filter(p => !neighbors(p, size).some(n => board[n] === color));
      if (internal.length <= 1) for (const p of region.points) ownership[p] = color;
    }
  }
  return ownership;
}

/** A closed two-chain seki: neither side can start without losing its chain.
 * Two liberties per chain covers shared-two-liberty and eye-plus-shared-liberty
 * seki. Closure and actual capture replies exclude atari, escape, and ko fights.
 */
function sekiOwnership(board: Stone[], size: number, safe: Stone[], positions: ReadonlySet<string>) {
  const ownership = [...safe], chains: { color: 1 | 2; stones: Set<number>; liberties: Set<number> }[] = [];
  const chainAt = Array<number>(board.length).fill(-1);
  for (let start = 0; start < board.length; start++) {
    if (!board[start] || chainAt[start] !== -1) continue;
    const chain = { color: board[start] as 1 | 2, ...group(board, start, size) }, id = chains.length;
    chains.push(chain);
    for (const p of chain.stones) chainAt[p] = id;
  }
  for (let a = 0; a < chains.length; a++) for (let b = a + 1; b < chains.length; b++) {
    const pair = [chains[a], chains[b]];
    if (pair[0].color === pair[1].color || pair.some(c => c.liberties.size !== 2 || [...c.stones].some(p => safe[p]))) continue;
    if (![...pair[0].liberties].some(p => pair[1].liberties.has(p))) continue;
    const liberties = new Set(pair.flatMap(c => [...c.liberties]));
    // No move can connect to a third chain; neighboring outside chains must
    // already be unconditionally alive so their capture cannot open an escape.
    if ([...liberties].some(p => neighbors(p, size).some(n => board[n] && chainAt[n] !== a && chainAt[n] !== b))) continue;
    if (pair.some(c => [...c.stones].some(p => neighbors(p, size).some(n => board[n] && chainAt[n] !== a && chainAt[n] !== b && safe[n] !== board[n])))) continue;
    const mutuallyBlocked = pair.every(chain => {
      let tried = false;
      for (const p of liberties) {
        const move = placement(board, p, chain.color, size);
        if (!move) continue;
        tried = true;
        if (positions.has(positionKey(move.board)) || move.captured || move.liberties !== 1) return false;
        const replyPoint = [...group(move.board, p, size).liberties][0];
        const reply = placement(move.board, replyPoint, chain.color === 1 ? 2 : 1, size);
        if (!reply || positions.has(positionKey(reply.board)) || reply.liberties < 2 || [...chain.stones, p].some(s => reply.board[s] === chain.color)) return false;
      }
      return tried;
    });
    if (mutuallyBlocked) for (const chain of pair) for (const p of chain.stones) ownership[p] = chain.color;
  }
  return ownership;
}

function emptyRegions(board: Stone[], size: number) {
  const regions: { points: number[]; boundary: Set<number> }[] = [], visited = new Set<number>();
  for (let start = 0; start < board.length; start++) {
    if (board[start] || visited.has(start)) continue;
    const points = [start], boundary = new Set<number>(); visited.add(start);
    for (let i = 0; i < points.length; i++) for (const n of neighbors(points[i], size)) {
      if (board[n]) boundary.add(n);
      else if (!visited.has(n)) { visited.add(n); points.push(n); }
    }
    regions.push({ points, boundary });
  }
  return regions;
}

/** Prove a small enclosed eyespace can be secured even when the invader starts.
 * Attacker nodes require every legal invasion (including playing elsewhere) to
 * be answered; defender nodes need one continuation to certified territory.
 * Running out of depth/nodes never counts as a proof. No model or playout guess.
 */
function canSecureEyespace(board: Stone[], size: number, points: number[], chain: ReturnType<typeof group>, color: 1 | 2, positions: ReadonlySet<string>): boolean {
  let remainingNodes = 2000;
  const opponent = color === 1 ? 2 : 1, seen = new Set(positions);
  seen.add(positionKey(board));
  const target = [...chain.stones, ...points];
  function solve(current: Stone[], turn: 1 | 2, depth: number): boolean {
    if (--remainingNodes < 0 || depth === 0 || [...chain.stones].some(p => current[p] !== color)) return false;
    const proven = passAliveOwnership(current, size);
    if (target.every(p => proven[p] === color)) return true;
    const candidates = points.flatMap(p => {
      const move = placement(current, p, turn, size);
      return move ? [move.board] : [];
    });
    // An off-board ko threat changes the global position. Do not certify life
    // by silently excluding a repeating local attack from this bounded reading.
    if (turn !== color && candidates.some(next => seen.has(positionKey(next)))) return false;
    const moves = candidates.filter(next => !seen.has(positionKey(next)));
    const nextTurn = turn === 1 ? 2 : 1;
    function continueWith(next: Stone[]) {
      const key = positionKey(next); seen.add(key);
      const secured = solve(next, nextTurn, depth - 1); seen.delete(key);
      return secured;
    }
    // A defender passing cannot create the required two-eye certificate.
    if (turn === color) return moves.some(continueWith);
    return solve(current, color, depth - 1) && moves.every(continueWith);
  }
  return solve(board, opponent, 12);
}

export function goSettlement(board: Stone[], size: number, positions: ReadonlySet<string> = new Set()): GoSettlement {
  const safe = passAliveOwnership(board, size), alive = sekiOwnership(board, size, safe, positions);
  const regions = emptyRegions(board, size);
  for (const { points, boundary } of regions) {
    if (points.length < 4 || points.length > 8 || points.every(p => alive[p]) || !boundary.size) continue;
    const start = [...boundary][0], color = board[start] as 1 | 2, chain = group(board, start, size);
    if (![...boundary].every(p => chain.stones.has(p))) continue;
    if (canSecureEyespace(board, size, points, chain, color, positions)) {
      for (const p of [...chain.stones, ...points]) alive[p] = color;
    }
  }
  const names = ['unsettled', 'black', 'white'] as const;
  const ownership: GoOwnership[] = alive.map(owner => names[owner]);
  for (const { points, boundary } of regions) {
    if (!boundary.size || [...boundary].some(p => !alive[p])) continue;
    const colors = new Set([...boundary].map(p => alive[p]));
    if (colors.size === 1) {
      // A private one-point eye of a seki chain belongs to its owner under area
      // scoring. Larger enclosures still require the pass-alive territory test.
      if (points.length === 1) ownership[points[0]] = names[[...colors][0]];
    } else if (points.every(p => {
      const adjacent = new Set(neighbors(p, size).filter(n => board[n]).map(n => alive[n]));
      return adjacent.has(1) && adjacent.has(2);
    })) {
      // Only a closed shared boundary is neutral. Merely touching both colors
      // somewhere around a large open region is not a settlement certificate.
      for (const p of points) if (!alive[p]) ownership[p] = 'neutral';
    }
  }
  return { ownership, neutral: ownership.filter(owner => owner === 'neutral').length, unsettled: ownership.filter(owner => owner === 'unsettled').length };
}

export class GoGame {
  readonly kind = 'go';
  state: GoState;
  private positions = new Set<string>();
  constructor(size = 5) {
    if (![5, 7, 9].includes(size)) throw new HttpError('Go boards must be 5, 7, or 9 intersections wide.');
    const board = Array<Stone>(size * size).fill(0);
    this.state = { kind: 'go', size, board, turn: 1, move: 0, captures: { black: 0, white: 0 }, komi: 0.5, last: null, finished: false, winner: null, score: { black: 0, white: 0.5 }, settlement: goSettlement(board, size) };
    this.positions.add(positionKey(this.state.board));
  }
  legal() {
    const { board, size, turn } = this.state;
    return board.flatMap((_, i) => {
      const move = placement(board, i, turn, size);
      return move && !this.positions.has(positionKey(move.board)) ? [{ index: i, name: pointName(i, size), ...move }] : [];
    });
  }
  request(model: string): DecisionRequest {
    const { board, size, turn, move, komi, settlement } = this.state;
    const legal = this.legal();
    if (!legal.length) throw new HttpError('No legal Go placements remain for the current player. Passing is disabled; start a new game.', 409);
    return {
      model,
      state: { game: 'Go', size, to_play: turn === 1 ? 'black' : 'white', move, komi_for_white: komi,
        rules: 'Area scoring with settled neutral points left unclaimed. Stones and owned territory count for their owner; neutral points count for neither side. White receives komi. Capture groups with no liberties. Suicide and positional superko are forbidden. Every move must place a stone; passing is not allowed. Play ends as soon as every point is black, white, or neutral in the ownership map, with zero unsettled points. Living groups include proven mutual life (seki). Captures do not add extra area points.',
        coordinates: `Columns A to ${'ABCDEFGHJ'[size - 1]} left to right; rows ${size} to 1 top to bottom. .=empty, B=black, W=white.`,
        board: Array.from({ length: size }, (_, y) => board.slice(y * size, (y + 1) * size).map(s => ['.', 'B', 'W'][s]).join(' ')),
        captures: this.state.captures, provisional_area: areaScore(board, size, komi),
        ownership: Array.from({ length: size }, (_, y) => settlement.ownership.slice(y * size, (y + 1) * size)),
        ownership_legend: 'Same rows and columns as board. black/white=settled ownership, including living stones; neutral=settled shared points worth zero to both sides; unsettled=still contested or not established. Unsettled does not mean neutral.',
        settlement: { unsettled_points: settlement.unsettled, neutral_points: settlement.neutral, settled_points: board.length - settlement.unsettled } },
      questions: { move: { type: 'choice', instructions: 'Choose the best legal stone placement for the side to play. Capture enemies, defend groups in atari, connect your stones, and secure living groups. Resolve unsettled areas while preserving your eyes and mutual-life groups. Settled neutral points do not need filling for the game to end. The options contain only legal placements; select one of the supplied coordinates.',
        criteria: Object.fromEntries(legal.map(m => [m.name, { play_at: m.name, captures: m.captured, group_liberties_after: m.liberties, surrounded_by_own_stones: neighbors(m.index, size).every(n => board[n] === turn), point_status: settlement.ownership[m.index] }])) } },
    };
  }
  apply(choice: string) {
    const s = this.state;
    if (s.finished) throw new HttpError('This game has ended.', 409);
    const move = this.legal().find(m => m.name === choice);
    if (!move) throw new HttpError('The model selected an illegal Go move.');
    s.board = move.board; s.captures[s.turn === 1 ? 'black' : 'white'] += move.captured;
    s.last = move.index; this.positions.add(positionKey(s.board));
    s.move++; s.settlement = goSettlement(s.board, s.size, this.positions);
    s.finished = s.settlement.unsettled === 0;
    s.score = s.finished ? { black: s.settlement.ownership.filter(owner => owner === 'black').length, white: s.settlement.ownership.filter(owner => owner === 'white').length + s.komi } : areaScore(s.board, s.size, s.komi);
    if (s.finished) s.winner = s.score.black > s.score.white ? 'black' : 'white';
    else s.turn = s.turn === 1 ? 2 : 1;
  }
}

const directions = [{ id: 'north', dx: 0, dy: -1 }, { id: 'east', dx: 1, dy: 0 }, { id: 'south', dx: 0, dy: 1 }, { id: 'west', dx: -1, dy: 0 }];
export class MazeGame {
  readonly kind = 'maze';
  state: MazeState;
  constructor(size = 9, seed = 42) {
    if (![5, 7, 9, 11, 15].includes(size) || !Number.isSafeInteger(seed)) throw new HttpError('Invalid maze size or seed.');
    const rng = new Random(seed), grid = Array.from({ length: size }, () => Array<number>(size).fill(1));
    const stack: [number, number][] = [[1, 1]]; grid[1][1] = 0;
    while (stack.length) {
      const [x, y] = stack[stack.length - 1];
      const candidates = directions.filter(d => { const nx = x + d.dx * 2, ny = y + d.dy * 2; return nx > 0 && nx < size - 1 && ny > 0 && ny < size - 1 && grid[ny][nx]; });
      if (!candidates.length) { stack.pop(); continue; }
      const d = rng.choice(candidates), nx = x + d.dx * 2, ny = y + d.dy * 2;
      grid[y + d.dy][x + d.dx] = 0; grid[ny][nx] = 0; stack.push([nx, ny]);
    }
    const visits = grid.map(row => row.map(() => 0)); visits[1][1] = 1;
    this.state = { kind: 'maze', size, seed, grid, position: [1, 1], goal: [size - 2, size - 2], trail: [[1, 1]], visits, steps: 0, finished: false };
  }
  legal() {
    const { position: [x, y], grid } = this.state;
    return directions.filter(d => grid[y + d.dy]?.[x + d.dx] === 0);
  }
  request(model: string): DecisionRequest {
    const s = this.state;
    return { model, state: { game: 'maze', coordinates: 'Zero-based [x,y]. x increases east; y increases south. # is a wall; . is an open cell; P is you; G is the exit.',
      grid: s.grid.map((row, y) => row.map((v, x) => x === s.position[0] && y === s.position[1] ? 'P' : x === s.goal[0] && y === s.goal[1] ? 'G' : v ? '#' : '.').join('')),
      position: s.position, goal: s.goal, visits: s.visits, recent_path: s.trail.slice(-30), steps: s.steps },
      questions: { move: { type: 'choice', instructions: 'Choose the next one-cell move to reach G in this maze. Inspect the symbolic grid for a path, do not cross walls. Prefer an unvisited route toward the exit. At a dead end, backtrack until an unexplored route is available. Avoid repeatedly cycling between visited cells. All supplied moves are legal.',
        criteria: Object.fromEntries(this.legal().map(d => { const x = s.position[0] + d.dx, y = s.position[1] + d.dy; return [d.id, { next_position: [x, y], previous_visits: s.visits[y][x], reaches_exit: x === s.goal[0] && y === s.goal[1] }]; })) } } };
  }
  apply(choice: string) {
    const s = this.state;
    if (s.finished) throw new HttpError('This maze is complete.', 409);
    const d = this.legal().find(d => d.id === choice);
    if (!d) throw new HttpError('The model selected a blocked maze move.');
    s.position = [s.position[0] + d.dx, s.position[1] + d.dy];
    s.steps++; s.visits[s.position[1]][s.position[0]]++; s.trail.push([...s.position]);
    s.finished = s.position.every((v, i) => v === s.goal[i]);
  }
}
