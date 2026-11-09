import type { GoState, MazeState, SnakeState } from './lab-types';
export function GoBoard({ state }: { state: GoState }) {
  const n = state.size, margin = 38, gap = 58, end = margin + (n - 1) * gap, width = end + margin;
  return <svg className="go-board" viewBox={`0 0 ${width} ${width}`} role="img" aria-label={`${n} by ${n} Go board. Move ${state.move}.`}>
    <rect width={width} height={width} rx="8" fill="#f4f2eb" />
    {Array.from({ length: n }, (_, i) => <g key={i}>
      <line x1={margin} y1={margin + i * gap} x2={end} y2={margin + i * gap} stroke="#c2bfb4" />
      <line y1={margin} x1={margin + i * gap} y2={end} x2={margin + i * gap} stroke="#c2bfb4" />
      <text x={margin + i * gap} y={width - 12} textAnchor="middle" className="board-coordinate">{'ABCDEFGHJ'[i]}</text>
      <text x="16" y={margin + i * gap + 4} textAnchor="middle" className="board-coordinate">{n - i}</text>
    </g>)}
    <circle cx={width / 2} cy={width / 2} r="3" fill="#8a877c" />
    {state.board.map((stone, i) => stone ? <g key={i} className="go-stone">
      <circle cx={margin + i % n * gap} cy={margin + Math.floor(i / n) * gap + 2} r="24" fill="#000" opacity=".09" />
      <circle cx={margin + i % n * gap} cy={margin + Math.floor(i / n) * gap} r="24" fill={stone === 1 ? '#242424' : '#fff'} stroke={stone === 1 ? '#161616' : '#d5d3cd'} />
      {state.last === i && <circle cx={margin + i % n * gap} cy={margin + Math.floor(i / n) * gap} r="5" fill="none" stroke={stone === 1 ? '#e8e7e1' : '#555'} strokeWidth="1.5" />}
    </g> : null)}
  </svg>;
}
export function MazeBoard({ state }: { state: MazeState }) {
  const cell = 30, size = state.size * cell;
  return <svg className="maze-board" viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`Maze. Explorer at ${state.position.join(', ')}. Exit at ${state.goal.join(', ')}. ${state.steps} steps.`}>
    <rect width={size} height={size} rx="5" fill="#fafaf8" />
    {state.grid.flatMap((row, y) => row.map((wall, x) => <rect key={`${x}:${y}`} x={x * cell} y={y * cell} width={cell + .1} height={cell + .1} fill={wall ? '#353936' : state.visits[y][x] ? '#e5e9e1' : '#fafaf8'} />))}
    {state.trail.length > 1 && <polyline points={state.trail.map(([x, y]) => `${(x + .5) * cell},${(y + .5) * cell}`).join(' ')} fill="none" stroke="#a9b7a0" strokeWidth="2" strokeLinejoin="round" />}
    <rect x={state.goal[0] * cell + 7} y={state.goal[1] * cell + 7} width="16" height="16" rx="3" fill="none" stroke="#4b6640" strokeWidth="2" strokeDasharray="3 2" />
    <circle cx={(state.position[0] + .5) * cell} cy={(state.position[1] + .5) * cell} r="8" fill="#314b2c" stroke="#fff" strokeWidth="2" />
  </svg>;
}

export function SnakeBoard({ state }: { state: SnakeState }) {
  const cell = 28, width = state.size * cell, [hx, hy] = state.body[0];
  const angle = { UP: -90, DOWN: 90, LEFT: 180, RIGHT: 0 }[state.direction];
  return <svg className="snake-board" viewBox={`0 0 ${width} ${width}`} role="img" aria-label={`${state.size} by ${state.size} Snake board. Score ${state.score}. Length ${state.body.length}. Step ${state.steps}. ${state.finished ? state.outcome === 'won' ? 'Board filled.' : 'Game over.' : `Moving ${state.direction.toLowerCase()}.`} ${state.food ? `Food at ${state.food.join(', ')}.` : ''}`}>
    <rect width={width} height={width} fill="#fafbf7" />
    {Array.from({ length: state.size * state.size }, (_, i) => <rect key={i} x={i % state.size * cell} y={Math.floor(i / state.size) * cell} width={cell} height={cell} fill={(i % state.size + Math.floor(i / state.size)) % 2 ? '#f1f4ec' : '#fafbf7'} />)}
    <polyline points={state.body.map(([x, y]) => `${(x + .5) * cell},${(y + .5) * cell}`).join(' ')} fill="none" stroke="#81966b" strokeWidth="19" strokeLinejoin="round" strokeLinecap="round" />
    {state.body.map(([x, y], i) => <rect key={`${x}:${y}`} x={x * cell + 3} y={y * cell + 3} width={22} height={22} rx={6} fill={i === 0 ? state.outcome === 'lost' ? '#a56952' : '#344d2b' : `hsl(91 20% ${46 + 20 * i / state.body.length}%)`} />)}
    <g transform={`translate(${(hx + .5) * cell} ${(hy + .5) * cell}) rotate(${angle})`} fill="white"><circle cx="4" cy="-5" r="2" /><circle cx="4" cy="5" r="2" /></g>
    {state.food && <g transform={`translate(${(state.food[0] + .5) * cell} ${(state.food[1] + .5) * cell})`}><circle cy="2" r="7" fill="#c57951" /><path d="M0 -5q0 -5 5 -5" fill="none" stroke="#718952" strokeWidth="2" strokeLinecap="round" /></g>}
  </svg>;
}

export function SnakeDetails({ state, guarded, busy, onGuardChange }: { state: SnakeState; guarded: boolean; busy: boolean; onGuardChange: (guarded: boolean) => void }) {
  return <>
    <div className="game-metrics snake-metrics"><div><span>Score</span><strong>{state.score}</strong><em>Food eaten</em></div><div><span>Length</span><strong>{state.body.length}<small> / {state.size * state.size}</small></strong><em>Cells occupied</em></div></div>
    <div className="maze-legend snake-legend"><span><i className="explorer-dot" />Snake</span><span><i className="food-mark" />Food</span><code>seed {state.seed}</code></div>
    <div className="snake-guard"><label><input type="checkbox" checked={guarded} disabled={busy} onChange={e => onGuardChange(e.target.checked)} />Cycle guard</label><span>{state.interventions} {state.interventions === 1 ? 'intervention' : 'interventions'}</span><small>Changing the guard starts a new game.</small></div>
    {state.last && <div className={`snake-decision${state.last.intervened ? ' intervened' : ''}`} role="status"><span>Model <code>{state.last.proposed}</code></span><span>Played <code>{state.last.executed}</code></span>{state.last.intervened && <small>Guard chose the highest-probability direction on the safe route.</small>}</div>}
    <p className="game-help">Eat food to grow. Fill the board to win; hitting a wall or your body ends the game. Each step sends the board and fixed direction descriptions. The model chooses a move and assesses collision-free movement and food reachability.{state.guarded ? ' The cycle guard can redirect unsafe moves; original model probabilities stay visible.' : ' The model’s chosen direction is executed directly.'}</p>
  </>;
}
