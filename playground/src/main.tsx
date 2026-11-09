import { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import type { Answer, Profile } from './types';
import type { GameKind, GameSnapshot, RunResult, Tab } from './lab-types';
import { parseDecisionRequest, parseState } from './request';
import { examples } from './examples';
import { GoBoard, MazeBoard, SnakeBoard, SnakeDetails } from './Boards';
import { QueryResults } from './QueryResults';
import './style.css';

async function api<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { method: body === undefined ? 'GET' : 'POST', headers: body === undefined ? undefined : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body), signal });
  const value = await response.json();
  if (!response.ok) throw new Error(value.detail ?? 'Request failed.');
  return value;
}
const pretty = (v: unknown) => JSON.stringify(v, null, 2);
const describe = (v: unknown) => typeof v === 'string' ? v : JSON.stringify(v);
function Icon({ name }: { name: 'arrow' | 'play' | 'pause' | 'reset' | 'copy' | 'code' | 'go' | 'maze' | 'snake' }) {
  const paths = { arrow: 'M3 8h10M9 4l4 4-4 4', play: 'M5 3l8 5-8 5z', pause: 'M5 3v10M11 3v10', reset: 'M3 6a5.5 5.5 0 1 1 0 5M3 2v4h4', copy: 'M6 5V2h7v9h-3M3 5h7v9H3z', code: 'M5 3L1 8l4 5M11 3l4 5-4 5', go: 'M3 1v14M8 1v14M13 1v14M1 3h14M1 8h14M1 13h14', maze: 'M1 1h14v14H1V5h6v6h4V5M7 1v4M1 11h6', snake: 'M3 13h7a3 3 0 0 0 0-6H6a2 2 0 0 1 0-4h5M11 2v2h3V2z' };
  return <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}
function AnswerCard({ id, answer, instructions }: { id: string; answer: Answer; instructions: unknown }) {
  const entries: [string, number][] = answer.type === 'noul' ? [['true', answer.noul ?? 0], ['false', 1 - (answer.noul ?? 0)]] : Object.entries(answer.probabilities ?? {});
  if (answer.type !== 'score') entries.sort((a, b) => b[1] - a[1]);
  const chosen = answer.type === 'choice' ? answer.choice : answer.type === 'score' ? Number(answer.score).toFixed(2) : `${Math.round((answer.noul ?? 0) * 100)}% yes`;
  const max = Math.max(...entries.map(([, p]) => p));
  return <article className="answer-card"><div className="answer-description"><div className="answer-id"><span title={id}>{id}</span><span>· {answer.type}</span></div><p title={describe(instructions)}>{describe(instructions)}</p><div className="answer-value">{chosen}</div>{answer.confidence !== undefined && <span className="confidence">confidence {answer.confidence.toFixed(2)}</span>}</div><div className="probability-list">{entries.map(([label, value]) => <div className="probability-row" key={label} title={answer.legend ? describe(answer.legend[label]) : label}><span className="probability-label">{label}</span><div className="probability-track"><span style={{ width: `${Math.min(1, Math.max(0, value)) * 100}%` }} className={value === max ? 'highest' : ''} /></div><span className="probability-value">{value.toFixed(2)}</span></div>)}</div></article>;
}

function App() {
  const [tab, setTab] = useState<Tab>('query');
  const [profiles, setProfiles] = useState<Profile[]>([]), [backend, setBackend] = useState('gs1_local');
  const [stateText, setStateText] = useState(String(examples[0].state)), [stateFormat, setStateFormat] = useState<'text' | 'json'>('text');
  const [questions, setQuestions] = useState(pretty(examples[0].questions));
  const [full, setFull] = useState(false), [requestText, setRequestText] = useState('');
  const [queryResult, setQueryResult] = useState<RunResult | null>(null);
  const [games, setGames] = useState<Partial<Record<GameKind, GameSnapshot>>>({});
  const [sizes, setSizes] = useState({ go: 5, maze: 9, snake: 10 });
  const [snakeGuard, setSnakeGuard] = useState(true);
  const [view, setView] = useState<'overview' | 'json'>('overview');
  const [busy, setBusy] = useState(false), [running, setRunning] = useState(false), [pausing, setPausing] = useState(false);
  const [error, setError] = useState(''), [notice, setNotice] = useState(''), [copied, setCopied] = useState(false);
  const serial = useRef(0), controller = useRef<AbortController | null>(null), auto = useRef(false);
  const cache = useRef(games); cache.current = games;
  const actionRef = useRef<() => void>(() => {});
  const profile = profiles.find(p => p.id === backend), model = profile?.model ?? 'GS1';
  const game = tab === 'query' ? undefined : games[tab];
  const result = tab === 'query' ? queryResult : game?.result;
  const raw = result?.raw_response, answers = raw?.answers ?? {};
  const elapsed = result ? `${Math.round(result.latency_ms).toLocaleString()} ms` : '— ms';
  function store(next: GameSnapshot) { cache.current = { ...cache.current, [next.state.kind]: next }; setGames(cache.current); if (next.state.kind === 'snake') setSnakeGuard(next.state.guarded); }
  function cancel() { serial.current++; auto.current = false; controller.current?.abort(); setBusy(false); setRunning(false); setPausing(false); }
  function inputChanged() { setQueryResult(null); setError(''); setNotice(''); }
  function request() {
    let value: unknown;
    try { value = full ? JSON.parse(requestText) : { state: parseState(stateText, stateFormat), questions: JSON.parse(questions) }; }
    catch (e) { throw new Error(e instanceof SyntaxError ? 'Questions or request contain invalid JSON.' : (e as Error).message); }
    return parseDecisionRequest(value, model);
  }
  async function loadGame(kind: GameKind, fresh = false, size = sizes[kind], guarded = snakeGuard) {
    cancel(); const token = serial.current; setBusy(true); setError(''); setNotice('');
    const abort = new AbortController(); controller.current = abort;
    try {
      const existing = cache.current[kind];
      const next = existing && !fresh ? await api<GameSnapshot>(`/playground/games/${existing.id}`, undefined, abort.signal) : await api<GameSnapshot>('/playground/games', { kind, size, ...(kind === 'snake' ? { guarded } : {}) }, abort.signal);
      if (token === serial.current) store(next);
    } catch (e) {
      if (token === serial.current && !abort.signal.aborted) {
        setError((e as Error).message);
        if (kind === 'snake') { const previous = cache.current.snake?.state; setSnakeGuard(previous?.kind === 'snake' ? previous.guarded : true); }
      }
    }
    finally { if (token === serial.current) setBusy(false); }
  }
  function switchTab(next: Tab) { cancel(); setTab(next); setError(''); setNotice(''); if (next !== 'query') void loadGame(next); }
  async function runQuery() {
    cancel(); const token = serial.current; setError(''); setNotice('');
    let input; try { input = request(); } catch (e) { setError((e as Error).message); return; }
    const abort = new AbortController(); controller.current = abort; setBusy(true); setQueryResult(null);
    try { const next = await api<RunResult>('/playground/query', { ...input, backend }, abort.signal); if (token === serial.current) setQueryResult(next); }
    catch (e) { if (token === serial.current && !abort.signal.aborted) setError((e as Error).message); }
    finally { if (token === serial.current) setBusy(false); }
  }
  async function runGame(single = false) {
    if (tab === 'query') return;
    cancel(); const token = serial.current; let current = cache.current[tab];
    if (!current || current.state.finished) return;
    setBusy(true); setRunning(!single); auto.current = !single; setError(''); setNotice('');
    try {
      for (let i = 0; i < 120; i++) {
        if (token !== serial.current) break;
        const abort = new AbortController(); controller.current = abort;
        current = await api<GameSnapshot>(`/playground/games/${current.id}/step`, { version: current.version, backend }, abort.signal);
        if (token !== serial.current) break;
        store(current);
        if (single || current.state.finished || !auto.current) break;
        if (i === 119) setNotice('Paused after 120 model calls. Continue to keep playing.');
        await new Promise(resolve => setTimeout(resolve, 220));
        if (!auto.current) break;
      }
    } catch (e) {
      if (token === serial.current) {
        setError((e as Error).message);
        try { const next = await api<GameSnapshot>(`/playground/games/${current.id}`); if (token === serial.current) store(next); } catch { /* Preserve the last confirmed board. */ }
      }
    } finally { if (token === serial.current) { auto.current = false; setBusy(false); setRunning(false); setPausing(false); } }
  }
  function pause() { auto.current = false; setPausing(true); }
  function selectExample(id: string) {
    const preset = examples.find(e => e.id === id)!; inputChanged();
    setStateText(typeof preset.state === 'string' ? preset.state : pretty(preset.state)); setStateFormat(typeof preset.state === 'string' ? 'text' : 'json');
    setQuestions(pretty(preset.questions)); setRequestText(pretty({ state: preset.state, questions: preset.questions }));
  }
  function toggleEditor() {
    try { const value = request(); if (!full) setRequestText(pretty(value)); else { setStateFormat(typeof value.state === 'string' ? 'text' : 'json'); setStateText(typeof value.state === 'string' ? value.state : pretty(value.state)); setQuestions(pretty(value.questions)); } setFull(!full); setError(''); }
    catch (e) { setError((e as Error).message); }
  }
  function formatEditor() { try { if (full) setRequestText(pretty(JSON.parse(requestText))); else { setQuestions(pretty(JSON.parse(questions))); if (stateFormat === 'json') setStateText(pretty(JSON.parse(stateText))); } setError(''); } catch { setError('Fix the JSON syntax before formatting.'); } }
  async function copyResponse() { try { await navigator.clipboard.writeText(pretty(raw)); setCopied(true); setTimeout(() => setCopied(false), 1600); } catch { setError('Clipboard unavailable. Select and copy the JSON directly.'); } }
  useEffect(() => { const abort = new AbortController(); void api<{ backends: Profile[]; default: string }>('/playground/backends', undefined, abort.signal).then(v => { setProfiles(v.backends); setBackend(v.default); }).catch(e => { if (!abort.signal.aborted) setError(e.message); }); return () => { abort.abort(); controller.current?.abort(); auto.current = false; }; }, []);
  actionRef.current = () => { if (!busy && profile?.available) { if (tab === 'query') void runQuery(); else void runGame(true); } };
  useEffect(() => { const handler = (e: KeyboardEvent) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); actionRef.current(); } }; window.addEventListener('keydown', handler); return () => window.removeEventListener('keydown', handler); }, []);
  return <div className="app-shell">
    <header className="topbar"><a href="/" className="brand">GS1 Playground</a></header>
    <main>
      <nav className="main-tabs" aria-label="Playgrounds">{(['query', 'go', 'maze', 'snake'] as Tab[]).map((id, i) => <button key={id} className={tab === id ? 'active' : ''} aria-current={tab === id ? 'page' : undefined} onClick={() => switchTab(id)}><Icon name={id === 'query' ? 'code' : id} />{id === 'query' ? 'Query' : id === 'go' ? 'Go' : id === 'maze' ? 'Maze' : 'Snake'}<span className="tab-index">0{i + 1}</span></button>)}<span className="connection"><i className={profile?.available ? 'online' : ''} />{profile?.available ? backend === 'jev' ? 'Connected to OpenRouter' : 'Connected to model' : profile ? 'Model unavailable' : 'Connecting to model'}</span></nav>
      <div className="workspace"><section className="input-panel" aria-label="State and questions">
        <div className="panel-heading"><span>{tab === 'query' ? 'Input' : tab === 'maze' ? 'The maze' : 'The board'}</span><div className="heading-tools">{tab === 'query' ? <><button disabled={busy} onClick={formatEditor}>Format</button><button disabled={busy} onClick={toggleEditor}>{full ? 'Split editor' : 'Full request'}</button></> : <><label className="size-select">Size<select aria-label={tab === 'go' ? 'Go board size' : tab === 'maze' ? 'Maze size' : 'Snake board size'} value={sizes[tab]} disabled={busy} onChange={e => { const size = Number(e.target.value); setSizes(v => ({ ...v, [tab]: size })); void loadGame(tab, true, size); }}>{(tab === 'go' ? [5, 7, 9] : tab === 'maze' ? [5, 7, 9, 11, 15] : [6, 8, 10, 12]).map(n => <option key={n} value={n}>{n} × {n}</option>)}</select></label><button disabled={busy} onClick={() => void loadGame(tab, true)}><Icon name="reset" />{tab === 'maze' ? 'New maze' : 'New game'}</button></>}</div></div>
        <div className="input-scroll">{tab === 'query' ? <>
          {full ? <div className="editor-field full-editor"><label htmlFor="request-json">Request <span>native TypeSafe JSON</span></label><textarea id="request-json" className="code-input" spellCheck={false} value={requestText} disabled={busy} onChange={e => { setRequestText(e.target.value); inputChanged(); }} /></div> : <><div className="editor-field"><div className="field-heading"><label htmlFor="state-input">State <span>the shared context</span></label><div className="mini-switch">{(['text', 'json'] as const).map(f => <button key={f} disabled={busy} className={stateFormat === f ? 'active' : ''} onClick={() => { if (stateFormat === f) return; try { const current = parseState(stateText, stateFormat); setStateText(f === 'json' ? pretty(current) : typeof current === 'string' ? current : pretty(current)); setStateFormat(f); inputChanged(); } catch (e) { setError((e as Error).message); } }}>{f === 'text' ? 'Text' : 'JSON'}</button>)}</div></div><textarea id="state-input" className="code-input state-input" spellCheck={false} value={stateText} disabled={busy} onChange={e => { setStateText(e.target.value); inputChanged(); }} /></div><div className="editor-field questions-editor"><label htmlFor="questions-input">Questions <span>choice, noul, score</span></label><textarea id="questions-input" className="code-input" spellCheck={false} value={questions} disabled={busy} onChange={e => { setQuestions(e.target.value); inputChanged(); }} /></div></>}
          <section className="query-examples" aria-labelledby="examples-heading"><h2 id="examples-heading">Examples</h2><div className="example-buttons">{examples.map(p => <button key={p.id} disabled={busy} title={p.description} onClick={() => selectExample(p.id)}>{p.label}</button>)}</div></section>
        </> : game ? <>
          <div className="game-status"><span><i className={game.state.kind === 'go' ? game.state.turn === 1 ? 'stone black' : 'stone white' : 'explorer-dot'} />{game.state.finished ? game.state.kind === 'snake' && game.state.outcome === 'lost' ? 'Game over' : 'Game complete' : game.state.kind === 'go' ? `${game.state.turn === 1 ? 'Black' : 'White'} to play` : game.state.kind === 'maze' ? 'Finding a way through' : `Moving ${game.state.direction.toLowerCase()}`}</span><span>{game.state.kind === 'go' ? `Move ${game.state.move}` : `Step ${game.state.steps}`}</span></div>
          <div className="board-wrap">{game.state.kind === 'go' ? <GoBoard state={game.state} /> : game.state.kind === 'maze' ? <MazeBoard state={game.state} /> : <SnakeBoard state={game.state} />}</div>
          {game.state.finished && <div className={`completion${game.state.kind === 'snake' && game.state.outcome === 'lost' ? ' lost' : ''}`} role="status">{game.state.kind === 'go' ? `${game.state.winner === 'black' ? 'Black' : 'White'} wins by ${Math.abs(game.state.score.black - game.state.score.white).toFixed(1)} points.` : game.state.kind === 'maze' ? `Exit reached in ${game.state.steps} steps.` : game.state.outcome === 'won' ? `Board filled. You win with ${game.state.score} food eaten.` : `Game over: ${game.state.death_reason === 'wall' ? 'hit the wall' : game.state.death_reason === 'reverse' ? 'reversed into the neck' : 'hit the body'}. Score ${game.state.score}.`}<span>{game.state.kind === 'go' ? 'Every point is owned or neutral. Neutral points count for neither side.' : game.state.kind === 'maze' ? 'Generate a new maze to try again.' : 'Start a new game to play again.'}</span></div>}
          {game.state.kind === 'go' ? <><div className="game-metrics"><div><span><i className="stone black" />Black</span><strong>{game.state.score.black}<small> {game.state.finished ? 'final area' : 'area estimate'}</small></strong><em>{game.state.captures.black} captures</em></div><div><span><i className="stone white" />White</span><strong>{game.state.score.white}<small> {game.state.finished ? 'final area' : 'area estimate'}</small></strong><em>{game.state.captures.white} captures · +0.5 komi</em></div></div><div className="go-settlement" role="status"><span>{game.state.board.length - game.state.settlement.unsettled} / {game.state.board.length} points settled</span><span>{game.state.settlement.neutral} neutral · {game.state.settlement.unsettled} unresolved</span></div><p className="game-help">The game ends when every point belongs to Black, White, or is settled neutral. Neutral points count for neither side. Every move places a stone; passing is disabled. The selected model receives the ownership map with every move.</p></> : game.state.kind === 'snake' ? <SnakeDetails state={game.state} guarded={snakeGuard} busy={busy} onGuardChange={guarded => { setSnakeGuard(guarded); void loadGame('snake', true, sizes.snake, guarded); }} /> : <><div className="maze-legend"><span><i className="explorer-dot" />Explorer</span><span><i className="exit-mark" />Exit</span><span><i className="visited-mark" />Visited</span><code>seed {game.state.seed}</code></div><p className="game-help">Each call sends the grid, position, visit history, and legal directions to the selected model.</p></>}
          {!!game.moves.length && <div className="move-history"><span>RECENT MOVES</span><div>{game.moves.slice(-10).map(m => <code key={m.number} title={`${m.player} · ${Math.round(m.latency_ms)} ms`}>{m.number}. {game.state.kind === 'go' && <i className={`stone ${m.player.toLowerCase()}`} />}{m.choice}</code>)}</div></div>}
          {game.result && <details className="request-details"><summary>Inspect last state & questions <Icon name="code" /></summary><pre>{pretty(game.result.request)}</pre></details>}
        </> : <div className="loading-board">{busy ? 'Preparing the board…' : 'Start a new game to begin.'}</div>}</div>
        {error && <div className="error-message" role="alert">{error}<button onClick={() => setError('')} aria-label="Dismiss error">×</button></div>}{notice && <div className="notice" role="status">{notice}</div>}
        <div className="run-bar"><div className="model-control"><span className="model-label">MODEL</span><select aria-label="Decision model" value={backend} disabled={busy} onChange={e => { setBackend(e.target.value); inputChanged(); }}>{profiles.length ? profiles.map(p => <option key={p.id} value={p.id}>{p.label}</option>) : <option value="jev">Jev · OpenRouter</option>}</select><code title={model}>{model}</code></div><div className="run-actions">{tab !== 'query' && <button className="step-button" disabled={busy || !game || game.state.finished || !profile?.available} onClick={() => void runGame(true)}>Step</button>}{running ? <button className="run-button" disabled={pausing} onClick={pause}><Icon name="pause" />{pausing ? 'Pausing…' : 'Pause'}</button> : tab === 'query' && busy ? <button className="run-button" onClick={cancel}>Cancel</button> : <button className="run-button" disabled={busy || !profile?.available || (tab !== 'query' && (!game || game.state.finished))} onClick={() => tab === 'query' ? void runQuery() : void runGame()}>{busy ? <span className="spinner" /> : <Icon name={tab === 'query' ? 'arrow' : 'play'} />}{busy ? 'Thinking…' : tab === 'query' ? 'Run query' : tab === 'maze' ? 'Run maze' : tab === 'snake' ? 'Run snake' : 'Run game'}</button>}</div></div>
      </section>
      <section className="response-panel" aria-label="Model response" aria-busy={busy}>
        <div className="panel-heading response-heading"><div className="response-tabs" role="tablist" aria-label="Response view"><button role="tab" aria-selected={view === 'overview'} className={view === 'overview' ? 'active' : ''} onClick={() => setView('overview')}>Overview</button><button role="tab" aria-selected={view === 'json'} className={view === 'json' ? 'active' : ''} onClick={() => setView('json')}>JSON</button></div><div className="latency" aria-label="Model latency"><span className={busy ? 'pulse-dot' : ''} />{elapsed}</div></div>
        <div className="response-meta"><span>{result ? `${result.response.usage.input_tokens.toLocaleString()} input tokens · ${Object.keys(answers).length} answers` : 'The model’s response will appear here'}</span>{result && <button aria-label="Copy raw JSON response" onClick={() => void copyResponse()}><Icon name="copy" />{copied ? 'Copied' : 'Copy'}</button>}</div>
        <div className="response-scroll" role="tabpanel">{result ? view === 'json' ? <pre className="json-response" aria-label="Raw JSON response">{pretty(raw)}</pre> : tab === 'query' ? <QueryResults result={result} /> : <div className="answer-list">{Object.entries(answers).map(([id, a]) => <AnswerCard key={id} id={tab === 'snake' && id === 'risk' ? 'can_move' : tab === 'snake' && id === 'food' ? 'food_reachable' : id} answer={a} instructions={result.request.questions[id]?.instructions} />)}<div className="response-footnote"><span className="online-indicator" />{raw?.model}{game && <span>Turn {game.version}</span>}</div></div> : <div className="empty-response"><div className="empty-bars"><i /><i /><i /></div><h2>{busy ? 'Reading the state…' : 'A little context. A clear decision.'}</h2><p>{busy ? 'The model is evaluating the questions. The response and model latency will appear here.' : tab === 'query' ? 'Run the example, or bring your own state and questions. See the probabilities behind every answer.' : 'Start the model to see its next move, the alternatives, and their probabilities.'}</p><span>CHOICE <b>·</b> NOUL <b>·</b> SCORE</span></div>}</div>
      </section></div>
      <footer className="page-footer"><span>Shared state. Independent questions.</span><span>{tab === 'query' ? '⌘ / Ctrl + Enter to run' : 'Pause finishes the current move'}<b>·</b>Powered by GS1</span></footer>
    </main>
  </div>;
}
createRoot(document.getElementById('root')!).render(<App />);
