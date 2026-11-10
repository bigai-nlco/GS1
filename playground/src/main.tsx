import { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import type { Profile, RunResult } from './types';
import { parseDecisionRequest, parseState } from './request';
import { examples } from './examples';
import { QueryResults } from './QueryResults';
import './style.css';

async function api<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { method: body === undefined ? 'GET' : 'POST', headers: body === undefined ? undefined : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body), signal });
  const value = await response.json();
  if (!response.ok) throw new Error(value.detail ?? 'Request failed.');
  return value;
}
const pretty = (v: unknown) => JSON.stringify(v, null, 2);
function Icon({ name }: { name: 'arrow' | 'copy' | 'code' }) {
  const paths = { arrow: 'M3 8h10M9 4l4 4-4 4', copy: 'M6 5V2h7v9h-3M3 5h7v9H3z', code: 'M5 3L1 8l4 5M11 3l4 5-4 5' };
  return <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}

function App() {
  const [profiles, setProfiles] = useState<Profile[]>([]), [backend, setBackend] = useState('gs1_local');
  const [stateText, setStateText] = useState(String(examples[0].state)), [stateFormat, setStateFormat] = useState<'text' | 'json'>('text');
  const [questions, setQuestions] = useState(pretty(examples[0].questions));
  const [full, setFull] = useState(false), [requestText, setRequestText] = useState('');
  const [result, setResult] = useState<RunResult | null>(null);
  const [view, setView] = useState<'overview' | 'json'>('overview');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(''), [copied, setCopied] = useState(false);
  const serial = useRef(0), controller = useRef<AbortController | null>(null);
  const actionRef = useRef<() => void>(() => {});
  const profile = profiles.find(p => p.id === backend), model = profile?.model ?? 'GS1';
  const raw = result?.raw_response, answers = raw?.answers ?? {};
  const elapsed = result ? `${Math.round(result.latency_ms).toLocaleString()} ms` : '— ms';
  function cancel() { serial.current++; controller.current?.abort(); setBusy(false); }
  function inputChanged() { setResult(null); setError(''); setCopied(false); }
  function request() {
    let value: unknown;
    try { value = full ? JSON.parse(requestText) : { state: parseState(stateText, stateFormat), questions: JSON.parse(questions) }; }
    catch (e) { throw new Error(e instanceof SyntaxError ? 'Questions or request contain invalid JSON.' : (e as Error).message); }
    return parseDecisionRequest(value, model);
  }
  async function runQuery() {
    cancel(); const token = serial.current; setError('');
    let input; try { input = request(); } catch (e) { setError((e as Error).message); return; }
    const abort = new AbortController(); controller.current = abort; setBusy(true); setResult(null);
    try { const next = await api<RunResult>('/playground/query', { ...input, backend }, abort.signal); if (token === serial.current) setResult(next); }
    catch (e) { if (token === serial.current && !abort.signal.aborted) setError((e as Error).message); }
    finally { if (token === serial.current) setBusy(false); }
  }
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
  useEffect(() => { const abort = new AbortController(); void api<{ backends: Profile[]; default: string }>('/playground/backends', undefined, abort.signal).then(v => { setProfiles(v.backends); setBackend(v.default); }).catch(e => { if (!abort.signal.aborted) setError(e.message); }); return () => { abort.abort(); controller.current?.abort(); }; }, []);
  actionRef.current = () => { if (!busy && profile?.available) void runQuery(); };
  useEffect(() => { const handler = (e: KeyboardEvent) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); actionRef.current(); } }; window.addEventListener('keydown', handler); return () => window.removeEventListener('keydown', handler); }, []);
  return <div className="app-shell">
    <header className="topbar"><a href="/" className="brand">GS1 Playground</a></header>
    <main>
      <div className="query-toolbar"><h1><Icon name="code" />Query</h1><span className="connection"><i className={profile?.available ? 'online' : ''} />{profile?.available ? backend === 'jev' ? 'Connected to TypeSafe' : 'Connected to model' : profile ? 'Model unavailable' : 'Connecting to model'}</span></div>
      <div className="workspace"><section className="input-panel" aria-label="State and questions">
        <div className="panel-heading"><span>Input</span><div className="heading-tools"><button disabled={busy} onClick={formatEditor}>Format</button><button disabled={busy} onClick={toggleEditor}>{full ? 'Split editor' : 'Full request'}</button></div></div>
        <div className="input-scroll">
          {full ? <div className="editor-field full-editor"><label htmlFor="request-json">Request <span>native TypeSafe JSON</span></label><textarea id="request-json" className="code-input" spellCheck={false} value={requestText} disabled={busy} onChange={e => { setRequestText(e.target.value); inputChanged(); }} /></div> : <><div className="editor-field"><div className="field-heading"><label htmlFor="state-input">State <span>the shared context</span></label><div className="mini-switch">{(['text', 'json'] as const).map(f => <button key={f} disabled={busy} className={stateFormat === f ? 'active' : ''} onClick={() => { if (stateFormat === f) return; try { const current = parseState(stateText, stateFormat); setStateText(f === 'json' ? pretty(current) : typeof current === 'string' ? current : pretty(current)); setStateFormat(f); inputChanged(); } catch (e) { setError((e as Error).message); } }}>{f === 'text' ? 'Text' : 'JSON'}</button>)}</div></div><textarea id="state-input" className="code-input state-input" spellCheck={false} value={stateText} disabled={busy} onChange={e => { setStateText(e.target.value); inputChanged(); }} /></div><div className="editor-field questions-editor"><label htmlFor="questions-input">Questions <span>choice, noul, score</span></label><textarea id="questions-input" className="code-input" spellCheck={false} value={questions} disabled={busy} onChange={e => { setQuestions(e.target.value); inputChanged(); }} /></div></>}
          <section className="query-examples" aria-labelledby="examples-heading"><h2 id="examples-heading">Examples</h2><div className="example-buttons">{examples.map(p => <button key={p.id} disabled={busy} title={p.description} onClick={() => selectExample(p.id)}>{p.label}</button>)}</div></section>
        </div>
        {error && <div className="error-message" role="alert">{error}<button onClick={() => setError('')} aria-label="Dismiss error">×</button></div>}
        <div className="run-bar"><div className="model-control"><span className="model-label">MODEL</span><select aria-label="Decision model" value={backend} disabled={busy} onChange={e => { setBackend(e.target.value); inputChanged(); }}>{profiles.length ? profiles.map(p => <option key={p.id} value={p.id}>{p.label}</option>) : <option value="gs1_local">GS1</option>}</select><code title={model}>{model}</code></div><div className="run-actions">{busy ? <button className="run-button" onClick={cancel}><span className="spinner" />Cancel</button> : <button className="run-button" disabled={!profile?.available} onClick={() => void runQuery()}><Icon name="arrow" />Run query</button>}</div></div>
      </section>
      <section className="response-panel" aria-label="Model response" aria-busy={busy}>
        <div className="panel-heading response-heading"><div className="response-tabs" role="tablist" aria-label="Response view"><button role="tab" aria-selected={view === 'overview'} className={view === 'overview' ? 'active' : ''} onClick={() => setView('overview')}>Overview</button><button role="tab" aria-selected={view === 'json'} className={view === 'json' ? 'active' : ''} onClick={() => setView('json')}>JSON</button></div><div className="latency" aria-label="Model latency"><span className={busy ? 'pulse-dot' : ''} />{elapsed}</div></div>
        <div className="response-meta"><span>{result ? `${result.response.usage.input_tokens.toLocaleString()} input tokens · ${Object.keys(answers).length} answers` : 'The model’s response will appear here'}</span>{result && <button aria-label="Copy raw JSON response" onClick={() => void copyResponse()}><Icon name="copy" />{copied ? 'Copied' : 'Copy'}</button>}</div>
        <div className="response-scroll" role="tabpanel">{result ? view === 'json' ? <pre className="json-response" aria-label="Raw JSON response">{pretty(raw)}</pre> : <QueryResults result={result} /> : <div className="empty-response"><div className="empty-bars"><i /><i /><i /></div><h2>{busy ? 'Reading the state…' : 'A little context. A clear decision.'}</h2><p>{busy ? 'The model is evaluating the questions. The response and model latency will appear here.' : 'Run the example, or bring your own state and questions. See the probabilities behind every answer.'}</p><span>CHOICE <b>·</b> NOUL <b>·</b> SCORE</span></div>}</div>
      </section></div>
      <footer className="page-footer"><span>Shared state. Independent questions.</span><span>⌘ / Ctrl + Enter to run<b>·</b>Powered by GS1</span></footer>
    </main>
  </div>;
}
createRoot(document.getElementById('root')!).render(<App />);
