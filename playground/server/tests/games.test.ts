import test from 'node:test';
import assert from 'node:assert/strict';
import { GoGame, MazeGame, areaScore, goSettlement } from '../games.js';
import type { Stone } from '../../src/lab-types.js';

const board = (...rows: string[]): Stone[] => rows.join('').split('').map(s => s === 'B' ? 1 : s === 'W' ? 2 : 0);

test('Go alternates turns, captures surrounded groups, and rejects suicide', () => {
  const game = new GoGame();
  game.apply('B5'); game.apply('A5'); game.apply('A4');
  assert.equal(game.state.board[0], 0);
  assert.equal(game.state.captures.black, 1);
  assert.equal(game.state.turn, 2);
  const suicide = new GoGame();
  for (const i of [7, 11, 13, 17]) suicide.state.board[i] = 2;
  assert.throws(() => suicide.apply('C3'), /illegal/);
  assert.equal(suicide.state.move, 0);
});
test('Go rejects immediate ko recapture and occupied points without mutation', () => {
  const game = new GoGame();
  for (const move of ['B3', 'C3', 'C4', 'B2', 'D3', 'D2', 'A5', 'C1', 'C2']) game.apply(move);
  assert.equal(game.state.board[12], 0);
  const before = structuredClone(game.state);
  assert.throws(() => game.apply('C3'), /illegal/);
  assert.throws(() => game.apply('B3'), /illegal/);
  assert.deepEqual(game.state, before);
});
test('Go rejects passing without changing either player’s turn or board', () => {
  const game = new GoGame();
  for (const move of ['C3', 'D3']) {
    const before = structuredClone(game.state);
    assert.throws(() => game.apply('pass'), /illegal/);
    assert.deepEqual(game.state, before);
    game.apply(move);
  }
  assert.equal(game.state.move, 2); assert.equal(game.state.turn, 1);
});
test('a provisional area advantage does not end an unsettled Go game', () => {
  const game = new GoGame(); game.apply('C3');
  assert.equal(game.state.finished, false); assert.equal(game.state.winner, null);
  assert.deepEqual(game.state.score, { black: 25, white: .5 });
  assert.equal(game.state.settlement.unsettled, 25);
  game.apply('D3'); assert.equal(game.state.move, 2);
  assert.deepEqual(areaScore([1, 0, 2, 1, 0, 2, 1, 0, 2], 3), { black: 3, white: 3.5 });
});
test('Benson analysis counts independent vital regions and removes false-eye dependencies', () => {
  const oneEye = board('..BBB', 'BBBBB', 'BBBBB', 'BBBBB', 'BBBBB');
  assert.equal(goSettlement(oneEye, 5).unsettled, 25, 'a two-point eye is only one vital region');
  const falseEye = board('BBBBB', 'B...B', 'B.B.B', 'B...B', '.BBBB');
  assert.equal(goSettlement(falseEye, 5).unsettled, 25, 'removing the center chain invalidates the outer chain’s second eye');
  const twoEyes = board('.BBBB', 'BBBBB', 'BBBBB', 'BBBBB', 'BBBB.');
  assert.deepEqual(goSettlement(twoEyes, 5), { ownership: Array(25).fill('black'), neutral: 0, unsettled: 0 });
});
test('settled shared boundaries are neutral and do not block completion', () => {
  const neutral = board('BB.WW', '.B.W.', 'BB.WW', '.B.W.', 'BB.WW');
  const settlement = goSettlement(neutral, 5);
  assert.equal(settlement.unsettled, 0); assert.equal(settlement.neutral, 5);
  assert.deepEqual(settlement.ownership, Array.from({ length: 5 }, () => ['black', 'black', 'neutral', 'white', 'white']).flat());
  const game = new GoGame(); game.state.board = [...neutral]; game.state.board[11] = 0;
  game.apply('B3');
  assert.equal(game.state.finished, true);
  assert.deepEqual(game.state.score, { black: 10, white: 10.5 });
});
test('enclosed territory can settle through a proven response to every invasion', () => {
  const fourPointEye = board('BBBBB', '....B', 'BBBBB', 'BBBBB', 'BBBBB');
  assert.deepEqual(goSettlement(fourPointEye, 5), { ownership: Array(25).fill('black'), neutral: 0, unsettled: 0 });
  const white = fourPointEye.map(s => s === 1 ? 2 : s);
  assert.deepEqual(goSettlement(white, 5), { ownership: Array(25).fill('white'), neutral: 0, unsettled: 0 });
  const killableEyes = [board('BBBBB', '...BB', 'BBBBB', 'BBBBB', 'BBBBB'), board('BBBBB', 'B..BB', 'B..BB', 'BBBBB', 'BBBBB')];
  for (const position of killableEyes) assert.ok(goSettlement(position, 5).unsettled > 0, 'a killable eye must not become owned territory');
  const priorInvasion = [...fourPointEye]; priorInvasion[6] = 2;
  assert.ok(goSettlement(fourPointEye, 5, new Set([priorInvasion.join('')])).unsettled > 0, 'bounded reading must not certify territory by excluding a local attack on repetition alone');
});
test('closed seki settles both living chains and their shared neutral liberties', () => {
  // The two chains share their last two liberties. Either side starting loses.
  for (const seki of [board('BBBWW', 'BB.WW', 'BBBWW', 'BB.WW', 'BBWWW'), board('BBWWW', '.BWW.', 'BBBWW', 'BB.WW', 'BBWWW')]) {
    const settlement = goSettlement(seki, 5);
    assert.equal(settlement.unsettled, 0);
    assert.equal(settlement.neutral, seki[5] === 0 ? 1 : 2);
    for (const [i, stone] of seki.entries()) if (stone) assert.equal(settlement.ownership[i], stone === 1 ? 'black' : 'white');
    if (!seki[5]) {
      assert.equal(settlement.ownership[5], 'black', 'a private seki eye belongs to Black');
      assert.equal(settlement.ownership[9], 'white', 'a private seki eye belongs to White');
    }
    const game = new GoGame(); game.state.board = [...seki]; game.state.board[0] = 0; game.apply('A5');
    assert.equal(game.state.finished, true);
    assert.equal(game.state.score.black + game.state.score.white, 25 - settlement.neutral + .5);
  }
});
test('open mixed-color regions, capture fights, and invadable enclosures stay unsettled', () => {
  const opening = board('.....', '.....', '.BW..', '.....', '.....');
  assert.equal(goSettlement(opening, 5).unsettled, 25);
  assert.equal(goSettlement(opening, 5).neutral, 0);
  const fight = board('..WW.', '.BWB.', '.BBW.', 'BBB..', 'BBBB.');
  assert.ok(goSettlement(fight, 5).unsettled > 0, 'a capturing fight is not neutral merely because both colors border empty points');
  const atari = board('BBBWW', 'BBBWW', 'BBBWW', 'BB.WW', 'BBWWW');
  assert.equal(goSettlement(atari, 5).neutral, 0, 'one shared liberty is a capture, not seki');
  assert.ok(goSettlement(atari, 5).unsettled > 0);
  const large = Array<Stone>(81).fill(1); large[0] = 0; large[80] = 0;
  for (let y = 2; y <= 4; y++) for (let x = 2; x <= 5; x++) large[y * 9 + x] = 0;
  assert.equal(areaScore(large, 9).black, 81);
  assert.equal(goSettlement(large, 9).unsettled, 12, 'living boundary stones do not prove that a large enclosure cannot be invaded');
});
test('seki is not certified when its required capture reply is forbidden by superko', () => {
  const position = board('BBBWW', 'BB.WW', 'BBBWW', 'BB.WW', 'BBWWW');
  const game = new GoGame(); game.state.board = position;
  game.apply('C4'); game.apply('C2');
  const settlement = goSettlement(position, 5, new Set([game.state.board.join('')]));
  assert.ok(settlement.unsettled > 0); assert.equal(settlement.neutral, 0);
});
test('Go ends immediately after a placement proves all ownership', () => {
  for (const color of [1, 2] as const) {
    const game = new GoGame();
    game.state.board = board('BBWWW', '.BW.W', 'B.WWW', '.BW.W', 'BBWWW').map(s => !s ? 0 : color === 1 ? s : s === 1 ? 2 : 1);
    game.state.turn = color;
    assert.ok(goSettlement(game.state.board, 5).unsettled > 0);
    game.apply('B3');
    assert.equal(game.state.finished, true);
    assert.equal(game.state.settlement.unsettled, 0);
    assert.equal(game.state.winner, color === 1 ? 'white' : 'black');
    assert.deepEqual(game.state.score, color === 1 ? { black: 10, white: 15.5 } : { black: 15, white: 10.5 });
    assert.throws(() => game.apply('A4'), /ended/);
  }
});
test('final scoring assigns provably dead stones to their territory owner', () => {
  const game = new GoGame();
  game.state.board = board('..BBB', 'BBBBB', 'B.WBB', 'BBBBB', 'BBBB.');
  assert.equal(areaScore(game.state.board, 5).white, 1.5);
  game.apply('B5');
  assert.equal(game.state.finished, true);
  assert.deepEqual(game.state.score, { black: 25, white: .5 });
  assert.equal(game.state.settlement.ownership[12], 'black');
});
test('model requests offer only legal placements and explain automatic settlement', () => {
  const game = new GoGame(); game.apply('C3');
  const request = game.request('jev-latest');
  const state = request.state as Record<string, unknown>;
  assert.match(String(state.rules), /passing is not allowed/);
  assert.equal('consecutive_passes' in state, false);
  assert.deepEqual(Object.keys(request.questions.move.criteria!), game.legal().map(move => move.name));
  assert.equal('pass' in request.questions.move.criteria!, false);
  assert.ok(state.provisional_area); assert.equal(state.current_area, undefined);
  assert.match(JSON.stringify(state.settlement), /unsettled_points":25/);
  assert.deepEqual(state.ownership, Array.from({ length: 5 }, () => Array(5).fill('unsettled')));
  game.state.board = board('BB.WW', '.B.W.', 'BB.WW', '.B.W.', 'BB.WW');
  game.state.settlement = goSettlement(game.state.board, 5);
  const settledState = game.request('jev-latest').state as Record<string, unknown>;
  assert.deepEqual(settledState.ownership, Array.from({ length: 5 }, () => ['black', 'black', 'neutral', 'white', 'white']));
  assert.deepEqual(settledState.settlement, { unsettled_points: 0, neutral_points: 5, settled_points: 25 });
});
test('generated mazes are deterministic, connected, and finish only at the exit', () => {
  for (const size of [5, 7, 9, 11, 15]) for (const seed of [1, 42, 73]) {
    const maze = new MazeGame(size, seed);
    assert.deepEqual(maze.state.grid, new MazeGame(size, seed).state.grid);
    assert.throws(() => maze.apply('north'), /blocked/);
    const queue: { x: number; y: number; route: string[] }[] = [{ x: 1, y: 1, route: [] }], seen = new Set(['1,1']);
    const dirs = [['north', 0, -1], ['east', 1, 0], ['south', 0, 1], ['west', -1, 0]] as const;
    let route: string[] = [];
    for (let i = 0; i < queue.length; i++) {
      const item = queue[i];
      if (item.x === size - 2 && item.y === size - 2) route = item.route;
      for (const [name, dx, dy] of dirs) {
        const x = item.x + dx, y = item.y + dy;
        if (maze.state.grid[y]?.[x] !== 0 || seen.has(`${x},${y}`)) continue;
        seen.add(`${x},${y}`); queue.push({ x, y, route: [...item.route, name] });
      }
    }
    assert.equal(seen.size, maze.state.grid.flat().filter(v => !v).length);
    assert.ok(route.length > 0);
    for (const move of route) { assert.equal(maze.state.finished, false); maze.apply(move); }
    assert.equal(maze.state.finished, true); assert.equal(maze.state.steps, route.length);
    assert.throws(() => maze.apply('west'), /complete/);
  }
  assert.notDeepEqual(new MazeGame(9, 1).state.grid, new MazeGame(9, 2).state.grid);
});
