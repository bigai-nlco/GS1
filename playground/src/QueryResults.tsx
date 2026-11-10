import { useState } from 'react';
import type { Answer, Question, RunResult } from './types';
import './query-results.css';

const percent = new Intl.NumberFormat('en', { style: 'percent', maximumFractionDigits: 1 });
const scoreFormat = new Intl.NumberFormat('en', { maximumFractionDigits: 2 });
const primitiveNames = { choice: 'Choice', noul: 'Noul', score: 'Score' };

function labelText(value: unknown): string {
  if (Array.isArray(value)) return value.map(labelText).join(' · ');
  if (value !== null && typeof value === 'object') return Object.entries(value).map(([key, item]) => `${key}: ${labelText(item)}`).join('; ');
  return String(value ?? 'null');
}

/** Keep structured instructions and criteria readable without flattening their fields. */
function StructuredValue({ value }: { value: unknown }) {
  if (Array.isArray(value)) return <ul className="structured-list">{value.map((item, i) => <li key={i}><StructuredValue value={item} /></li>)}</ul>;
  if (value !== null && typeof value === 'object') {
    const entries = Object.entries(value);
    if (!entries.length) return <span>{'{}'}</span>;
    return <dl className="structured-fields">{entries.map(([key, item]) => <div key={key}><dt>{key}: </dt><dd><StructuredValue value={item} /></dd></div>)}</dl>;
  }
  return <>{String(value ?? 'null')}</>;
}

function ProbabilityRing({ value }: { value: number }) {
  return <svg className="query-probability-ring" width="12" height="12" viewBox="0 0 16 16" aria-hidden="true">
    <circle cx="8" cy="8" r="5.5" fill="none" stroke="#dee6fb" strokeWidth="4" />
    <circle cx="8" cy="8" r="5.5" fill="none" stroke="#7a9cf5" strokeWidth="4" pathLength="100" strokeDasharray={`${value * 100} 100`} transform="rotate(-90 8 8)" />
  </svg>;
}

function ResultRow({ id, question, answer }: {
  id: string; question: Question; answer: Answer;
}) {
  const [showAll, setShowAll] = useState(false);
  const entries = Object.entries(answer.probabilities ?? {});
  if (answer.type === 'choice') entries.sort((a, b) => b[1] - a[1]);
  else if (answer.type === 'score') entries.sort((a, b) => Number(a[0]) - Number(b[0]));
  const shown = answer.type === 'choice' && !showAll ? entries.slice(0, 3) : entries;
  const levelLabel = (key: string) => answer.legend?.[key] ?? (question.type === 'score' ? question.criteria[Number(key)] : undefined) ?? key;
  const optionCount = question.type === 'choice' ? Object.keys(question.criteria).length : question.type === 'score' ? question.criteria.length : 2;
  const score = answer.score ?? 0;
  const markerLevel = Math.floor(score);

  return <article className="query-result-row" data-primitive={answer.type} data-question-id={id} aria-label={`${id} ${primitiveNames[answer.type]}`}>
    <div className="query-question">
      <h3>{id}</h3>
      <div className={`query-instructions${typeof question.instructions === 'string' ? ' is-plain' : ''}`}><StructuredValue value={question.instructions} /></div>
    </div>
    <div className="query-answer">
      {answer.type === 'noul' ? <>
        <div className="query-noul-value"><strong>{percent.format(answer.noul ?? 0)}</strong> <span>true</span></div>
        <div className="query-noul-meter" role="meter" aria-label={`${id} probability of true`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={(answer.noul ?? 0) * 100} aria-valuetext={`${percent.format(answer.noul ?? 0)} true`}>
          <i style={{ left: `${(answer.noul ?? 0) * 100}%` }} />
        </div>
      </> : answer.type === 'score' ? <div className="query-score-value"><strong>{scoreFormat.format(score)}</strong><span> of {optionCount - 1}</span></div> : <>
        <ol className="query-distribution" aria-label={`${id} probabilities`}>
          {shown.map(([key, value]) => <li key={key} className={`query-option${key === answer.choice ? ' selected' : ''}`}>
            <div className="query-option-label" title={question.type === 'choice' && question.criteria[key] != null ? labelText(question.criteria[key]) : undefined}><code>{key}</code></div>
            <span className="query-option-probability">{percent.format(value)}</span><ProbabilityRing value={value} />
          </li>)}
        </ol>
        {optionCount > 3 && <button className="query-show-options" aria-label={`${showAll ? 'Show top 3' : 'Show all'} ${id} options`} aria-expanded={showAll} onClick={() => setShowAll(!showAll)}>{showAll ? 'Show top 3' : `Show all ${optionCount} options`}</button>}
        {answer.confidence !== undefined && <div className="query-confidence">Confidence: {percent.format(answer.confidence)}</div>}
      </>}
    </div>
    <div className="query-primitive"><span>{primitiveNames[answer.type]}</span>{answer.type !== 'noul' && <small>{optionCount} {answer.type === 'choice' ? 'options' : `levels · 0~${optionCount - 1}`}</small>}</div>
    {answer.type === 'score' && <div className="query-score-chart" aria-label={`${id} probabilities`}>
      {entries.map(([key, probability]) => <div key={key} className="query-score-level">
        <div className="query-level-description" title={labelText(levelLabel(key))}>{labelText(levelLabel(key))}</div>
        <div className="query-score-measure">
          <span className="query-level-index">{key}</span>
          <span className="query-score-tick">
            {Number(key) === markerLevel && <i className="query-score-marker" role="meter" aria-label={`${id} expected level`} aria-valuemin={0} aria-valuemax={optionCount - 1} aria-valuenow={score} aria-valuetext={`${scoreFormat.format(score)} of ${optionCount - 1}`} style={{ top: `${(score - markerLevel + .5) * 100}%` }} />}
          </span>
          <span className="query-score-track"><i style={{ width: `${probability * 100}%` }} /></span>
          <span className={`query-score-probability${probability === 0 ? ' is-zero' : ''}`}>{percent.format(probability)}</span>
        </div>
      </div>)}
      {answer.confidence !== undefined && <div className="query-score-footer"><div className="query-confidence">Confidence: {percent.format(answer.confidence)}</div></div>}
    </div>}
  </article>;
}

export function QueryResults({ result }: { result: RunResult }) {
  return <section className="query-results" aria-label="Query results">
    {Object.entries(result.raw_response.answers).map(([id, answer]) => <ResultRow key={id} id={id} question={result.request.questions[id]} answer={answer} />)}
    <div className="response-footnote"><span className="online-indicator" />{result.raw_response.model}</div>
  </section>;
}
