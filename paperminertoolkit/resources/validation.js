/* Local human review. Suggestions are never scoring decisions. */
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
}[char]));
const url = path => path + '?token=' + encodeURIComponent(window.PMT_TOKEN);
const has = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
const empty = () => ({paperPairs: {}, materialPairs: {}, fields: {}, items: {}, extra: {}, run: {}});
const ABSENT = Symbol('absent');
let session = null;
let decisions = null;
let currentPaper = null;
let currentEntry = null;
let saveTimer = null;
let saveChain = Promise.resolve();

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  };
  const response = await fetch(url(path), options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed');
  return data;
}

function refreshReviewSelector(reviews) {
  const wrap = $('resume-wrap');
  const select = $('resume-select');
  if (!reviews?.length) { wrap.classList.add('hidden'); return; }
  wrap.classList.remove('hidden');
  select.innerHTML = '<option value="">Choose a saved review…</option>' + reviews.map(review => {
    const stamp = new Date(review.saved_at * 1000).toLocaleString();
    const label = `${review.validation_name} + ${review.scraped_name} · ${review.recipe} · ${stamp}` +
      (review.resumable ? '' : ' · select files once to enable resume');
    return `<option value="${esc(review.id)}" ${review.resumable ? '' : 'disabled'}>${esc(label)}</option>`;
  }).join('');
  $('resume-review').disabled = true;
}

async function fileSha256(file) {
  const bytes = await file.arrayBuffer();
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}

function items(raw) {
  const value = String(raw ?? '').trim();
  if (!value || ['none', 'null', 'nan'].includes(value.toLowerCase())) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : [parsed];
  } catch { return [value]; }
}

function valueText(raw) {
  const text = String(raw ?? '').trim();
  if (!text) return '<span class="muted">Empty</span>';
  try {
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) {
      if (!parsed.length) return '<span class="muted">[]</span>';
      return '<ul>' + parsed.map(item => '<li>' + itemText(item) + '</li>').join('') + '</ul>';
    }
    if (parsed && typeof parsed === 'object') return itemText(parsed);
  } catch { /* Show original text below. */ }
  return esc(text).replace(/\n/g, '<br>');
}

function itemText(item) {
  if (item && typeof item === 'object') {
    return Object.entries(item).map(([key, value]) =>
      '<div><code>' + esc(key) + '</code> ' + esc(value) + '</div>').join('');
  }
  return esc(item);
}

function paperName(paper) {
  return paper.meta.title || paper.meta.doi || paper.meta.paper_id || 'Unknown paper';
}

function identity(row) {
  return session.identity_fields.map(field => row.values[field] || '').filter(Boolean).join(' · ') || 'Unnamed entry';
}

function scrapedPaper(id) { return session.scraped.find(paper => paper.id === id); }
function goldPaper(id) { return session.validation.find(paper => paper.id === id); }
function pairedPaper(gold) {
  return has(decisions.paperPairs, gold.id) ? decisions.paperPairs[gold.id] : gold.suggested;
}

function pairedRow(gold, row) {
  const paper = scrapedPaper(pairedPaper(gold));
  if (!paper) return null;
  if (has(decisions.materialPairs, row.id)) {
    const id = decisions.materialPairs[row.id];
    return paper.rows.some(candidate => candidate.id === id) ? id : null;
  }
  return paper.id === gold.suggested ? gold.material_suggestions[row.id] || null : null;
}

function fieldDecision(row, field) { return decisions.fields[row.id]?.[field] || {}; }
function structuredDecision(row, field) { return decisions.parts?.[row.id]?.[field] || {}; }
function extraStatus(mark) {
  return mark === true ? 'incorrect' :
    mark && typeof mark === 'object' ? mark.status || 'unreviewed' : 'unreviewed';
}
function extraRowState(row) {
  const mark = decisions.extra[row.id];
  if (!mark) return {complete: false, incorrect: false};
  let state = {complete: true, incorrect: false};
  let populated = false;
  for (const field of session.fields) {
    const value = answer(row.values[field]);
    if (value === ABSENT) continue;
    populated = true;
    const status = decisions.fields[row.id]?.[field]?.status || extraStatus(mark);
    state = mergeStructureState(state, extraTreeState(value, '', status,
      decisions.parts?.[row.id]?.[field] || {}));
  }
  return populated ? state : {complete: extraStatus(mark) !== 'unreviewed',
    incorrect: extraStatus(mark) === 'incorrect'};
}
function extraReviewed(row) { return extraRowState(row).complete; }
function extraLabel(row) {
  if (!decisions.extra[row.id]) return 'unreviewed';
  const state = extraRowState(row);
  return `extra · ${state.complete ? state.incorrect ? 'incorrect' : 'correct' : 'unreviewed'}`;
}
function extraChoices(attribute, status) {
  return `<div class="decision">${['correct', 'incorrect', 'unreviewed'].map(choice =>
    `<button class="choice ${status === choice ? 'selected ' + (choice === 'correct' ? 'good' : choice === 'incorrect' ? 'bad' : 'neutral') : ''}" ${attribute} data-status="${choice}">${choice}</button>`
  ).join('')}</div>`;
}
function presentScalar(value) {
  return !['', 'none', 'null', 'nan'].includes(String(value ?? '').trim().toLowerCase());
}

function answer(raw) {
  const value = String(raw ?? '').trim();
  if (!presentScalar(value)) return ABSENT;
  try { const parsed = JSON.parse(value); return parsed === null ? ABSENT : parsed; }
  catch { return value; }
}

function partPath(parent, part) {
  return parent + '/' + String(part).replace(/~/g, '~0').replace(/\//g, '~1');
}

function partRecord(field, rowId = currentEntry) {
  decisions.parts ||= {};
  decisions.parts[rowId] ||= {};
  decisions.parts[rowId][field] ||= {lists: {}, values: {}, extraKeys: {}};
  const record = decisions.parts[rowId][field];
  record.lists ||= {}; record.values ||= {}; record.extraKeys ||= {};
  return record;
}

function listPart(field, path) {
  const record = partRecord(field);
  record.lists[path] ||= {matches: {}, extra: {}};
  record.lists[path].matches ||= {};
  record.lists[path].extra ||= {};
  return record.lists[path];
}

function extraTreeState(value, path, status, review) {
  if (value && typeof value === 'object') {
    const children = Array.isArray(value) ? value.map((item, index) => [index, item]) : Object.entries(value);
    return children.reduce((state, [part, item]) => mergeStructureState(state,
      extraTreeState(item, partPath(path, part), status, review)),
    {complete: true, incorrect: false});
  }
  if (!presentScalar(value)) return {complete: true, incorrect: false};
  const choice = review.values?.[path]?.status || status;
  return {complete: choice !== 'unreviewed', incorrect: choice === 'incorrect'};
}

function structureState(gold, scraped, path, review) {
  if (gold === null) gold = ABSENT;
  if (scraped === null) scraped = ABSENT;
  if (gold && typeof gold === 'object' && !Array.isArray(gold) ||
      scraped && typeof scraped === 'object' && !Array.isArray(scraped)) {
    if (gold !== ABSENT && (typeof gold !== 'object' || Array.isArray(gold)) ||
        scraped !== ABSENT && (typeof scraped !== 'object' || Array.isArray(scraped))) {
      return mergeStructureState(structureState(gold, ABSENT, path, review),
        structureState(ABSENT, scraped, path, review));
    }
    const left = gold === ABSENT ? {} : gold;
    const right = scraped === ABSENT ? {} : scraped;
    return [...new Set([...Object.keys(left), ...Object.keys(right)])].reduce((state, key) =>
      mergeStructureState(state, structureState(has(left, key) ? left[key] : ABSENT,
        has(right, key) ? right[key] : ABSENT, partPath(path, key), review)),
    {complete: true, incorrect: false});
  }
  if (Array.isArray(gold) || Array.isArray(scraped)) {
    if (gold !== ABSENT && !Array.isArray(gold) || scraped !== ABSENT && !Array.isArray(scraped)) {
      return mergeStructureState(structureState(gold, ABSENT, path, review),
        structureState(ABSENT, scraped, path, review));
    }
    const left = gold === ABSENT ? [] : gold;
    const right = scraped === ABSENT ? [] : scraped;
    const list = review.lists?.[path] || {};
    const used = new Set();
    let state = {complete: true, incorrect: false};
    left.forEach((item, index) => {
      const choice = list.matches?.[index] || {};
      const other = choice.scraped;
      if (choice.status === 'missing' && other === null) {
        state.incorrect = true;
      } else if (Number.isInteger(other) && other >= 0 && other < right.length && !used.has(other)) {
        used.add(other);
        state = mergeStructureState(state, structureState(item, right[other], partPath(path, index), review));
      } else state.complete = false;
    });
    right.forEach((item, index) => {
      if (!used.has(index)) {
        if (!list.extra?.[index]) state.complete = false;
        else state = mergeStructureState(state, extraTreeState(item, partPath(path, index),
          extraStatus(list.extra[index]), review));
      }
    });
    return state;
  }
  const a = gold !== ABSENT && presentScalar(gold);
  const b = scraped !== ABSENT && presentScalar(scraped);
  if (!a && !b) return {complete: true, incorrect: false};
  const status = review.values?.[path]?.status;
  if (a && b) return {complete: ['correct', 'incorrect'].includes(status), incorrect: status === 'incorrect'};
  if (a) return {complete: status === 'missing', incorrect: status === 'missing'};
  const extra = review.values?.[path]?.status || extraStatus(review.extraKeys?.[path]);
  return {complete: extra !== 'unreviewed', incorrect: extra === 'incorrect'};
}

function mergeStructureState(a, b) {
  return {complete: a.complete && b.complete, incorrect: a.incorrect || b.incorrect};
}

function entryStatus(gold, row) {
  const pair = pairedRow(gold, row);
  if (has(decisions.materialPairs, row.id) && pair === null) return 'missing';
  if (!pair) return 'unreviewed';
  const scraped = scrapedPaper(pairedPaper(gold))?.rows.find(candidate => candidate.id === pair);
  if (!scraped) return 'unreviewed';
  let incorrect = false;
  let complete = true;
  for (const field of session.fields) {
    if (session.structure_fields?.[field]) {
      const state = structureState(answer(row.values[field]), answer(scraped.values[field]),
        '', structuredDecision(row, field));
      if (!state.complete) complete = false;
      if (state.incorrect) incorrect = true;
    } else {
      const status = fieldDecision(row, field).status;
      if (status === 'incorrect') incorrect = true;
      if (!['correct', 'incorrect'].includes(status) &&
          (presentScalar(row.values[field]) || presentScalar(scraped.values[field]))) complete = false;
    }
  }
  return !complete ? 'unreviewed' : incorrect ? 'incorrect' : 'correct';
}

function usedRows(gold) {
  return new Set(gold.rows.map(row => pairedRow(gold, row)).filter(Boolean));
}

function paperStats(gold) {
  const tally = {correct: 0, incorrect: 0, missing: 0, extra: 0, unreviewed: 0};
  for (const row of gold.rows) tally[entryStatus(gold, row)]++;
  const scraped = scrapedPaper(pairedPaper(gold));
  const used = usedRows(gold);
  if (scraped) for (const row of scraped.rows) {
    if (!used.has(row.id)) tally[extraReviewed(row) ? 'extra' : 'unreviewed']++;
  }
  return tally;
}

function markDirty() {
  $('save-state').textContent = 'Saving…';
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, 300);
}

function save() {
  if (!session) return saveChain;
  const snapshot = JSON.parse(JSON.stringify(decisions));
  const fingerprint = session.fingerprint;
  saveChain = saveChain.then(() => api('/api/save', {fingerprint, decisions: snapshot}))
    .then(() => { $('save-state').textContent = 'Saved'; })
    .catch(error => { $('save-state').textContent = 'Save failed: ' + error.message; });
  return saveChain;
}

function start(data) {
  session = data;
  decisions = Object.assign(empty(), data.decisions || {});
  decisions.run ||= {};
  decisions.items ||= {};
  decisions.parts ||= {};
  for (const [rowId, fields] of Object.entries(decisions.items)) {
    for (const [field, old] of Object.entries(fields)) {
      if (decisions.parts[rowId]?.[field]) continue;
      decisions.parts[rowId] ||= {};
      const values = {};
      const row = session.validation.flatMap(paper => paper.rows).find(candidate => candidate.id === rowId);
      const oldItems = row ? items(row.values[field]) : [];
      for (const [index, choice] of Object.entries(old.matches || {})) {
        if (!oldItems[index] || typeof oldItems[index] !== 'object') {
          if (['correct', 'incorrect'].includes(choice.status)) {
            values[partPath('', index)] = {status: choice.status, note: choice.note || ''};
          }
        }
      }
      decisions.parts[rowId][field] = {lists: {'': old}, values, extraKeys: {}};
    }
  }
  currentPaper = session.validation[0]?.id || 'extra:' + session.scraped[0]?.id;
  currentEntry = null;
  $('setup').classList.add('hidden');
  $('review').classList.remove('hidden');
  $('score').classList.add('hidden');
  for (const id of ['change-inputs', 'show-score', 'export']) $(id).classList.remove('hidden');
  $('save-state').textContent = 'Review ready';
  $('save-state').title = data.review_path;
  render();
}

async function showSetup() {
  if (session) { clearTimeout(saveTimer); await save(); }
  session = null;
  decisions = null;
  $('review').classList.add('hidden');
  $('score').classList.add('hidden');
  $('setup').classList.remove('hidden');
  for (const id of ['change-inputs', 'show-score', 'export']) $(id).classList.add('hidden');
  $('save-state').textContent = '';
  try { refreshReviewSelector(await api('/api/reviews')); }
  catch (error) { $('setup-error').textContent = error.message; $('setup-error').classList.remove('hidden'); }
}

function render() {
  if (!session) return;
  const linked = new Set(session.validation.map(paper => pairedPaper(paper)).filter(Boolean));
  const extraPapers = session.scraped.filter(paper => !linked.has(paper.id));
  const total = {correct: 0, incorrect: 0, missing: 0, extra: 0, unreviewed: 0};
  for (const paper of session.validation) {
    const stats = paperStats(paper);
    for (const key of Object.keys(total)) total[key] += stats[key];
  }
  for (const paper of extraPapers) for (const row of paper.rows) {
    total[extraReviewed(row) ? 'extra' : 'unreviewed']++;
  }
  $('paper-count').textContent = session.validation.length + ' gold';
  $('overall').textContent = `${total.correct} correct · ${total.incorrect} incorrect · ${total.missing} missing · ${total.extra} extra · ${total.unreviewed} to review`;
  $('paper-list').innerHTML = session.validation.map(paper => {
    const stats = paperStats(paper);
    return `<button class="paper-button ${paper.id === currentPaper ? 'active' : ''}" data-paper="${esc(paper.id)}"><strong>${esc(paperName(paper))}</strong><small>${esc(paper.meta.doi || paper.meta.paper_id)} · ${paper.rows.length} expected · ${stats.correct} correct · ${stats.incorrect} incorrect${stats.missing ? ' · ' + stats.missing + ' missing' : ''}</small></button>`;
  }).join('') + (extraPapers.length ? '<div class="tiny muted" style="padding:12px 9px">Unmatched scraped papers</div>' : '') +
    extraPapers.map(paper => `<button class="paper-button ${currentPaper === 'extra:' + paper.id ? 'active' : ''}" data-paper="extra:${esc(paper.id)}"><strong>${esc(paperName(paper))}</strong><small>${paper.rows.length} scraped entries · no paired validation paper</small></button>`).join('');
  const gold = goldPaper(currentPaper);
  if (gold) renderGold(gold);
  else renderExtraPaper(scrapedPaper(String(currentPaper).replace(/^extra:/, '')));
}

function renderExtraLeaves(value, path, rowId, field, inherited) {
  if (value === ABSENT || value === null) return '';
  if (value && typeof value === 'object') {
    const entries = Array.isArray(value) ? value.map((item, index) => [index, item]) : Object.entries(value);
    return entries.map(([part, item]) => renderExtraLeaves(item, partPath(path, part), rowId, field, inherited)).join('');
  }
  if (!presentScalar(value)) return '';
  const status = decisions.parts?.[rowId]?.[field]?.values?.[path]?.status || inherited;
  const label = path.split('/').slice(1).join(' / ') || field;
  return `<div class="part-leaf"><div class="inline-label">${esc(label)}</div><div class="value">${esc(value)}</div>${extraChoices(`data-extra-leaf="${esc(rowId)}" data-field-name="${esc(field)}" data-path="${esc(path)}"`, status)}</div>`;
}

function renderExtraRowFields(row) {
  const inherited = extraStatus(decisions.extra[row.id]);
  return session.fields.map(field => {
    const value = answer(row.values[field]);
    if (value === ABSENT) return '';
    const status = decisions.fields[row.id]?.[field]?.status || inherited;
    const raw = row.raw_values?.[field] ?? row.values[field];
    return `<section class="field"><div class="field-name">${esc(field)}</div><div class="value">${valueText(row.values[field])}</div><details><summary>Raw cell</summary><pre>${esc(raw)}</pre></details>${extraChoices(`data-extra-field="${esc(row.id)}" data-field-name="${esc(field)}"`, status)}${value && typeof value === 'object' ? renderExtraLeaves(value, '', row.id, field, status) : ''}</section>`;
  }).join('');
}

function extraCards(rows) {
  return rows.length ? '<h3>Unpaired scraped entries</h3>' + rows.map(row =>
    `<div class="extra-card"><div class="row"><strong>${esc(identity(row))}</strong><button class="secondary ${decisions.extra[row.id] ? 'danger' : ''}" data-mark-extra="${esc(row.id)}">${decisions.extra[row.id] ? 'Undo extra' : 'Mark extra'}</button></div><small class="muted">${esc(row.meta.doi || row.meta.paper_id)}</small>${decisions.extra[row.id] ? `<p class="muted tiny">Accuracy of this extra entry. Field and part choices below override this choice.</p>${extraChoices(`data-extra-row="${esc(row.id)}"`, extraStatus(decisions.extra[row.id]))}${renderExtraRowFields(row)}` : ''}</div>`).join('') : '';
}

function renderGold(gold) {
  const scrapedPaperValue = scrapedPaper(pairedPaper(gold));
  const used = usedRows(gold);
  const extra = scrapedPaperValue ? scrapedPaperValue.rows.filter(row => !used.has(row.id)) : [];
  if (!currentEntry || !gold.rows.some(row => row.id === currentEntry)) currentEntry = gold.rows[0]?.id || null;
  $('entry-summary').textContent = `${gold.rows.length} validation · ${scrapedPaperValue?.rows.length || 0} scraped`;
  $('entry-list').innerHTML = gold.rows.map((row, index) => {
    const status = entryStatus(gold, row);
    const paired = scrapedPaperValue?.rows.find(candidate => candidate.id === pairedRow(gold, row));
    return `<button class="entry-button ${currentEntry === row.id ? 'active' : ''}" data-entry="${esc(row.id)}"><strong>${index + 1}. ${esc(identity(row))}</strong><small><span class="badge ${status === 'correct' ? 'good' : status === 'incorrect' || status === 'missing' ? 'bad' : ''}">${esc(status)}</span> ${esc(paired ? identity(paired) : 'No scraped pair')}</small></button>`;
  }).join('') + (extra.length ? '<div class="tiny muted" style="padding:12px 9px">Unpaired scraped entries</div>' : '') +
    extra.map(row => `<button class="entry-button" data-extra-entry="${esc(row.id)}"><strong>${esc(identity(row))}</strong><small><span class="badge ${extraRowState(row).complete ? extraRowState(row).incorrect ? 'bad' : 'good' : 'warn'}">${esc(extraLabel(row))}</span></small></button>`).join('');
  const stats = paperStats(gold);
  const paperChoices = session.scraped.map(paper => `<option value="${esc(paper.id)}" ${pairedPaper(gold) === paper.id ? 'selected' : ''}>${esc(paperName(paper))}${paperName(paper) === paper.meta.doi || paperName(paper) === paper.meta.paper_id ? '' : ' · ' + esc(paper.meta.doi || paper.meta.paper_id)}</option>`).join('');
  const top = `<h2>${esc(paperName(gold))}</h2><div class="meta">DOI: ${esc(gold.meta.doi || '—')} · Paper ID: ${esc(gold.meta.paper_id || '—')}</div><div class="summary">${stats.correct} correct · ${stats.incorrect} incorrect · ${stats.missing} missing · ${stats.extra} extra · ${stats.unreviewed} unreviewed</div>${gold.warning ? `<div class="warning">${esc(gold.warning)}</div>` : ''}<label for="paper-pair">Scraped paper ${!has(decisions.paperPairs, gold.id) && gold.suggested ? '(suggested from ' + esc(gold.match_reason) + ')' : ''}</label><select id="paper-pair"><option value="">No scraped paper</option>${paperChoices}</select>`;
  if (!gold.rows.length) {
    $('detail').innerHTML = top + '<div class="placeholder">This paper has zero expected material entries. Review any scraped entries below.</div>' + extraCards(extra);
    return;
  }
  const row = gold.rows.find(candidate => candidate.id === currentEntry);
  const paired = pairedRow(gold, row);
  const scraped = scrapedPaperValue?.rows.find(candidate => candidate.id === paired);
  const choices = (scrapedPaperValue?.rows || []).map(candidate => `<option value="${esc(candidate.id)}" ${paired === candidate.id ? 'selected' : ''}>${esc(identity(candidate))}</option>`).join('');
  const status = entryStatus(gold, row);
  $('detail').innerHTML = top + `<div class="pairing"><div><label>Validation entry</label><div class="value">${esc(identity(row))}</div></div><div><label for="material-pair">Scraped entry ${!has(decisions.materialPairs, row.id) && paired ? '(suggested)' : ''}</label><select id="material-pair"><option value="">Missing / no pair</option>${choices}</select></div></div><div class="row"><button class="secondary" data-action="missing">Mark missing</button><button class="primary" data-action="mark-all" ${!scraped ? 'disabled' : ''}>Mark entry correct</button><span class="badge ${status === 'correct' ? 'good' : status === 'incorrect' || status === 'missing' ? 'bad' : ''}">${esc(status)}</span></div>` +
    session.fields.map(field => renderField(row, scraped, field)).join('') +
    '<div class="footer-actions"><button class="secondary" data-action="prev">Previous entry</button><button class="secondary" data-action="next">Next entry</button></div>' + extraCards(extra);
}

function renderField(row, scraped, field) {
  const left = row.values[field];
  const right = scraped?.values[field] || '';
  const rawLeft = row.raw_values?.[field] ?? left;
  const rawRight = scraped?.raw_values?.[field] ?? right;
  const sideBySide = `<div class="compare"><div><div class="inline-label">Validation</div><div class="value">${valueText(left)}</div><details><summary>Raw cell</summary><pre>${esc(rawLeft)}</pre></details></div><div><div class="inline-label">Scraped</div><div class="value">${valueText(right)}</div><details><summary>Raw cell</summary><pre>${esc(rawRight)}</pre></details></div></div>`;
  if (session.structure_fields?.[field]) {
    const review = structuredDecision(row, field);
    return `<section class="field"><div class="field-name">${esc(field)} · part review</div>${sideBySide}<p class="muted tiny">Pair list items, then score each value or dictionary key separately. Mark missing and extra parts explicitly.</p>${renderStructureNode(answer(left), answer(right), '', field, review, field)}</section>`;
  }
  const decision = fieldDecision(row, field);
  return `<section class="field"><div class="field-name">${esc(field)}</div>${sideBySide}<div class="decision">${['correct', 'incorrect', 'unreviewed'].map(status => `<button class="choice ${decision.status === status ? 'selected ' + (status === 'correct' ? 'good' : status === 'incorrect' ? 'bad' : 'neutral') : ''}" data-field="${esc(field)}" data-status="${status}">${status}</button>`).join('')}</div><textarea class="note" data-note="${esc(field)}" placeholder="Optional note for this field">${esc(decision.note || '')}</textarea></section>`;
}

function renderStructureNode(gold, scraped, path, field, review, label) {
  if (gold === null) gold = ABSENT;
  if (scraped === null) scraped = ABSENT;
  const goldDict = gold && typeof gold === 'object' && !Array.isArray(gold);
  const scrapedDict = scraped && typeof scraped === 'object' && !Array.isArray(scraped);
  if (goldDict || scrapedDict) {
    if (gold !== ABSENT && !goldDict || scraped !== ABSENT && !scrapedDict) {
      return renderStructureNode(gold, ABSENT, path, field, review, label + ' (validation)') +
        renderStructureNode(ABSENT, scraped, path, field, review, label + ' (scraped)');
    }
    const left = gold === ABSENT ? {} : gold;
    const right = scraped === ABSENT ? {} : scraped;
    const keys = [...new Set([...Object.keys(left), ...Object.keys(right)])];
    return keys.length ? `<div class="part-group">${keys.map(key => renderStructureNode(
      has(left, key) ? left[key] : ABSENT, has(right, key) ? right[key] : ABSENT,
      partPath(path, key), field, review, key)).join('')}</div>` : '<p class="muted tiny">Empty dictionary.</p>';
  }
  if (Array.isArray(gold) || Array.isArray(scraped)) {
    if (gold !== ABSENT && !Array.isArray(gold) || scraped !== ABSENT && !Array.isArray(scraped)) {
      return renderStructureNode(gold, ABSENT, path, field, review, label + ' (validation)') +
        renderStructureNode(ABSENT, scraped, path, field, review, label + ' (scraped)');
    }
    const left = gold === ABSENT ? [] : gold;
    const right = scraped === ABSENT ? [] : scraped;
    const list = review.lists?.[path] || {};
    const used = new Set(Object.values(list.matches || {}).filter(choice =>
      Number.isInteger(choice.scraped) && choice.status !== 'missing').map(choice => choice.scraped));
    const references = left.map((item, index) => {
      const choice = list.matches?.[index] || {};
      const selected = choice.scraped;
      const valid = Number.isInteger(selected) && selected >= 0 && selected < right.length;
      const options = right.map((value, other) => `<option value="${other}" ${selected === other ? 'selected' : ''}>${other + 1}. ${esc(itemLabel(value))}</option>`).join('');
      const child = valid && choice.status !== 'missing' ? renderStructureNode(
        item, right[selected], partPath(path, index), field, review, `Item ${index + 1}`) : '';
      return `<div class="item-card"><div class="inline-label">${esc(label)} · reference item ${index + 1}</div><div class="value">${itemText(item)}</div><label>Scraped item</label><select data-struct-list-pair="${esc(field)}" data-path="${esc(path)}" data-index="${index}"><option value="">No scraped item</option>${options}</select><div class="decision"><button class="choice ${choice.status === 'missing' ? 'selected bad' : ''}" data-struct-list-missing="${esc(field)}" data-path="${esc(path)}" data-index="${index}">Mark missing</button></div><textarea class="note" data-struct-list-note="${esc(field)}" data-path="${esc(path)}" data-index="${index}" placeholder="Optional item note">${esc(choice.note || '')}</textarea>${child}</div>`;
    }).join('');
    const extras = right.map((item, index) => used.has(index) ? '' :
      `<div class="item-card"><div class="row"><strong>${esc(label)} · unpaired scraped item ${index + 1}</strong><button class="secondary ${list.extra?.[index] ? 'danger' : ''}" data-struct-list-extra="${esc(field)}" data-path="${esc(path)}" data-index="${index}">${list.extra?.[index] ? 'Undo extra' : 'Mark extra'}</button></div><div class="value">${itemText(item)}</div>${list.extra?.[index] ? extraChoices(`data-struct-extra-item="${esc(field)}" data-path="${esc(path)}" data-index="${index}"`, extraStatus(list.extra[index])) + renderExtraLeaves(item, partPath(path, index), currentEntry, field, extraStatus(list.extra[index])) : ''}</div>`).join('');
    return references + extras || '<p class="muted tiny">Empty list.</p>';
  }
  const a = gold !== ABSENT && presentScalar(gold);
  const b = scraped !== ABSENT && presentScalar(scraped);
  if (!a && !b) return `<div class="part-leaf"><strong>${esc(label)}</strong><span class="muted tiny"> No assessable value</span></div>`;
  const value = review.values?.[path] || {};
  if (!a) {
    const marked = review.extraKeys?.[path];
    return `<div class="part-leaf"><div class="inline-label">${esc(label)} · scraped only</div><div class="value">${itemText(scraped)}</div><button class="secondary ${marked ? 'danger' : ''}" data-struct-extra-key="${esc(field)}" data-path="${esc(path)}">${marked ? 'Undo extra' : 'Mark extra'}</button>${marked ? extraChoices(`data-struct-extra-key-status="${esc(field)}" data-path="${esc(path)}"`, review.values?.[path]?.status || extraStatus(marked)) : ''}</div>`;
  }
  const choices = b ? ['correct', 'incorrect', 'unreviewed'] : ['missing', 'unreviewed'];
  return `<div class="part-leaf"><div class="inline-label">${esc(label)}</div><div class="compare"><div><div class="inline-label">Validation</div><div class="value">${itemText(gold)}</div></div><div><div class="inline-label">Scraped</div><div class="value">${b ? itemText(scraped) : '<span class="muted">Missing</span>'}</div></div></div><div class="decision">${choices.map(status => `<button class="choice ${value.status === status ? 'selected ' + (status === 'correct' ? 'good' : status === 'unreviewed' ? 'neutral' : 'bad') : ''}" data-struct-value="${esc(field)}" data-path="${esc(path)}" data-status="${status}">${status}</button>`).join('')}</div><textarea class="note" data-struct-note="${esc(field)}" data-path="${esc(path)}" placeholder="Optional note for this part">${esc(value.note || '')}</textarea></div>`;
}

function itemLabel(item) {
  if (item && typeof item === 'object') return String(item.value ?? JSON.stringify(item)).slice(0, 100);
  return String(item).slice(0, 100);
}

function renderExtraPaper(paper) {
  if (!paper) { $('entry-list').innerHTML = ''; $('detail').innerHTML = ''; return; }
  $('entry-summary').textContent = `${paper.rows.length} scraped · no validation pair`;
  $('entry-list').innerHTML = paper.rows.map(row => `<button class="entry-button" data-extra-entry="${esc(row.id)}"><strong>${esc(identity(row))}</strong><small>${esc(extraLabel(row))}</small></button>`).join('');
  $('detail').innerHTML = `<h2>${esc(paperName(paper))}</h2><div class="warning">Pair this paper using a validation paper’s “Scraped paper” menu, or mark its entries extra.</div>${extraCards(paper.rows)}`;
}

function reviewed(row) {
  return Object.keys(decisions.fields[row.id] || {}).length ||
    Object.keys(decisions.items[row.id] || {}).length ||
    Object.keys(decisions.parts[row.id] || {}).length;
}

function updatePaperPair(id) {
  const gold = goldPaper(currentPaper);
  if (!gold) return;
  const affected = [gold, ...session.validation.filter(paper => paper.id !== gold.id && id && pairedPaper(paper) === id)];
  if (affected.some(paper => paper.rows.some(reviewed)) &&
      !confirm('Changing this paper pairing clears its existing field and item decisions. Continue?')) { render(); return; }
  for (const paper of affected) for (const row of paper.rows) {
    delete decisions.fields[row.id]; delete decisions.items[row.id];
    delete decisions.parts[row.id]; delete decisions.materialPairs[row.id];
  }
  if (id) for (const paper of affected) if (paper.id !== gold.id) decisions.paperPairs[paper.id] = null;
  decisions.paperPairs[gold.id] = id || null;
  currentEntry = null;
  markDirty(); render();
}

function updateMaterialPair(id) {
  const gold = goldPaper(currentPaper);
  if (!gold) return;
  const affected = [currentEntry, ...gold.rows.filter(row => row.id !== currentEntry && id && pairedRow(gold, row) === id).map(row => row.id)];
  if (affected.some(rowId => reviewed({id: rowId})) &&
      !confirm('Changing this entry pairing clears its existing field and item decisions. Continue?')) { render(); return; }
  for (const rowId of affected) {
    delete decisions.fields[rowId]; delete decisions.items[rowId]; delete decisions.parts[rowId];
  }
  for (const rowId of affected) if (rowId !== currentEntry) decisions.materialPairs[rowId] = null;
  decisions.materialPairs[currentEntry] = id || null;
  if (id) delete decisions.extra[id];
  markDirty(); render();
}

function structuresAlign(gold, scraped) {
  if (Array.isArray(gold) || Array.isArray(scraped)) {
    return Array.isArray(gold) && Array.isArray(scraped) &&
      gold.length === scraped.length && gold.every((item, index) => structuresAlign(item, scraped[index]));
  }
  const a = gold && typeof gold === 'object';
  const b = scraped && typeof scraped === 'object';
  if (a || b) return a && b && Object.keys(gold).length === Object.keys(scraped).length &&
    Object.keys(gold).every(key => has(scraped, key) && structuresAlign(gold[key], scraped[key]));
  return presentScalar(gold === ABSENT ? '' : gold) === presentScalar(scraped === ABSENT ? '' : scraped);
}

function markAllNode(gold, scraped, path, record) {
  if (Array.isArray(gold)) {
    const matches = {};
    gold.forEach((item, index) => {
      matches[index] = {scraped: index, note: ''};
      markAllNode(item, scraped[index], partPath(path, index), record);
    });
    record.lists[path] = {matches, extra: {}};
  } else if (gold && typeof gold === 'object') {
    for (const [key, value] of Object.entries(gold)) {
      markAllNode(value, scraped[key], partPath(path, key), record);
    }
  } else if (gold !== ABSENT && presentScalar(gold)) {
    record.values[path] = {status: 'correct', note: record.values[path]?.note || ''};
  }
}

document.addEventListener('click', event => {
  const target = event.target.closest('[data-paper],[data-entry],[data-extra-entry],[data-field],[data-action],[data-mark-extra],[data-extra-row],[data-extra-field],[data-extra-leaf],[data-struct-value],[data-struct-extra-key],[data-struct-extra-key-status],[data-struct-list-missing],[data-struct-list-extra],[data-struct-extra-item]');
  if (!target || !session || $('review').classList.contains('hidden')) return;
  if (target.dataset.paper) { currentPaper = target.dataset.paper; currentEntry = null; render(); return; }
  if (target.dataset.entry) { currentEntry = target.dataset.entry; render(); return; }
  if (target.dataset.extraEntry) {
    document.querySelector(`[data-mark-extra="${CSS.escape(target.dataset.extraEntry)}"]`)?.scrollIntoView({block: 'center'});
    return;
  }
  if (target.dataset.markExtra) {
    const id = target.dataset.markExtra;
    if (decisions.extra[id]) {
      delete decisions.extra[id]; delete decisions.fields[id]; delete decisions.parts[id];
    } else decisions.extra[id] = {status: 'unreviewed'};
    markDirty(); render(); return;
  }
  if (target.dataset.extraRow) {
    decisions.extra[target.dataset.extraRow] = {status: target.dataset.status};
    markDirty(); render(); return;
  }
  if (target.dataset.extraField) {
    const id = target.dataset.extraField;
    decisions.fields[id] ||= {};
    decisions.fields[id][target.dataset.fieldName] = {status: target.dataset.status};
    markDirty(); render(); return;
  }
  if (target.dataset.extraLeaf) {
    const record = partRecord(target.dataset.fieldName, target.dataset.extraLeaf);
    record.values[target.dataset.path] = {status: target.dataset.status};
    markDirty(); render(); return;
  }
  if (target.dataset.field) {
    const field = target.dataset.field;
    decisions.fields[currentEntry] ||= {};
    decisions.fields[currentEntry][field] ||= {};
    decisions.fields[currentEntry][field].status = target.dataset.status;
    decisions.fields[currentEntry][field].note = document.querySelector(`textarea[data-note="${CSS.escape(field)}"]`)?.value || '';
    markDirty(); render(); return;
  }
  if (target.dataset.structValue) {
    const record = partRecord(target.dataset.structValue);
    record.values[target.dataset.path] ||= {};
    record.values[target.dataset.path].status = target.dataset.status;
    markDirty(); render(); return;
  }
  if (target.dataset.structExtraKey) {
    const record = partRecord(target.dataset.structExtraKey);
    if (record.extraKeys[target.dataset.path]) {
      delete record.extraKeys[target.dataset.path];
      delete record.values[target.dataset.path];
    } else record.extraKeys[target.dataset.path] = {status: 'unreviewed'};
    markDirty(); render(); return;
  }
  if (target.dataset.structExtraKeyStatus) {
    const record = partRecord(target.dataset.structExtraKeyStatus);
    record.extraKeys[target.dataset.path] = {status: target.dataset.status};
    record.values[target.dataset.path] = {status: target.dataset.status};
    markDirty(); render(); return;
  }
  if (target.dataset.structListMissing) {
    const list = listPart(target.dataset.structListMissing, target.dataset.path);
    list.matches[target.dataset.index] ||= {};
    list.matches[target.dataset.index].status = 'missing';
    list.matches[target.dataset.index].scraped = null;
    markDirty(); render(); return;
  }
  if (target.dataset.structListExtra) {
    const list = listPart(target.dataset.structListExtra, target.dataset.path);
    if (list.extra[target.dataset.index]) {
      delete list.extra[target.dataset.index];
      const values = partRecord(target.dataset.structListExtra).values;
      const prefix = partPath(target.dataset.path, target.dataset.index);
      for (const path of Object.keys(values)) if (path === prefix || path.startsWith(prefix + '/')) delete values[path];
    } else list.extra[target.dataset.index] = {status: 'unreviewed'};
    markDirty(); render(); return;
  }
  if (target.dataset.structExtraItem) {
    const list = listPart(target.dataset.structExtraItem, target.dataset.path);
    list.extra[target.dataset.index] = {status: target.dataset.status};
    markDirty(); render(); return;
  }
  if (!target.dataset.action) return;
  const gold = goldPaper(currentPaper);
  const row = gold?.rows.find(candidate => candidate.id === currentEntry);
  if (!row) return;
  if (target.dataset.action === 'missing') { updateMaterialPair(''); return; }
  if (target.dataset.action === 'mark-all') {
    const scraped = scrapedPaper(pairedPaper(gold))?.rows.find(candidate => candidate.id === pairedRow(gold, row));
    if (!scraped) return;
    if (session.fields.some(field => !session.structure_fields?.[field] &&
        presentScalar(row.values[field]) !== presentScalar(scraped.values[field]))) {
      alert('A scalar field is present on only one side. Review its fields individually.'); return;
    }
    if (session.fields.some(field => session.structure_fields?.[field] &&
        !structuresAlign(answer(row.values[field]), answer(scraped.values[field])))) {
      alert('Structured parts differ. Review their items and keys individually.'); return;
    }
    decisions.materialPairs[row.id] = scraped.id;
    decisions.fields[row.id] ||= {};
    for (const field of session.fields) {
      if (session.structure_fields?.[field]) {
        decisions.parts[row.id] ||= {};
        const record = {lists: {}, values: {}, extraKeys: {}};
        markAllNode(answer(row.values[field]), answer(scraped.values[field]), '', record);
        decisions.parts[row.id][field] = record;
      } else {
        decisions.fields[row.id][field] = {status: 'correct', note: decisions.fields[row.id][field]?.note || ''};
      }
    }
    markDirty(); render(); return;
  }
  if (target.dataset.action === 'prev' || target.dataset.action === 'next') {
    const index = gold.rows.findIndex(candidate => candidate.id === row.id);
    const step = target.dataset.action === 'next' ? 1 : -1;
    currentEntry = gold.rows[Math.max(0, Math.min(gold.rows.length - 1, index + step))].id;
    render();
  }
});

document.addEventListener('change', event => {
  if (event.target.id === 'paper-pair') updatePaperPair(event.target.value);
  if (event.target.id === 'material-pair') updateMaterialPair(event.target.value);
  if (event.target.id === 'recipe-select') $('custom-recipe-wrap').classList.toggle('hidden', event.target.value !== '__custom__');
  if (event.target.dataset.structListPair) {
    const list = listPart(event.target.dataset.structListPair, event.target.dataset.path);
    list.matches[event.target.dataset.index] ||= {};
    list.matches[event.target.dataset.index].scraped = event.target.value === '' ? null : Number(event.target.value);
    list.matches[event.target.dataset.index].status = 'unreviewed';
    markDirty(); render();
  }
});

document.addEventListener('input', event => {
  if (!session) return;
  if (event.target.dataset.note && currentEntry) {
    const field = event.target.dataset.note;
    decisions.fields[currentEntry] ||= {};
    decisions.fields[currentEntry][field] ||= {status: 'unreviewed'};
    decisions.fields[currentEntry][field].note = event.target.value;
    markDirty();
  }
  if (event.target.dataset.structNote && currentEntry) {
    const record = partRecord(event.target.dataset.structNote);
    record.values[event.target.dataset.path] ||= {status: 'unreviewed'};
    record.values[event.target.dataset.path].note = event.target.value;
    markDirty();
  }
  if (event.target.dataset.structListNote && currentEntry) {
    const list = listPart(event.target.dataset.structListNote, event.target.dataset.path);
    list.matches[event.target.dataset.index] ||= {};
    list.matches[event.target.dataset.index].note = event.target.value;
    markDirty();
  }
  if (event.target.dataset.run) {
    decisions.run[event.target.dataset.run] = event.target.value;
    markDirty();
  }
});

$('load').onclick = async () => {
  const error = $('setup-error'); error.classList.add('hidden');
  const validation = $('validation-file').files[0];
  const scraped = $('scraped-file').files[0];
  const recipe = $('recipe-select').value;
  const custom = $('recipe-file').files[0];
  if (!validation || !scraped || !recipe || (recipe === '__custom__' && !custom)) {
    error.textContent = 'Choose both CSV files and a recipe.'; error.classList.remove('hidden'); return;
  }
  try {
    $('load').disabled = true;
    start(await api('/api/load', {
      validation: await validation.text(), scraped: await scraped.text(),
      recipe: recipe === '__custom__' ? '' : recipe,
      recipe_text: recipe === '__custom__' ? await custom.text() : '',
      validation_name: validation.name, scraped_name: scraped.name,
      validation_sha256: await fileSha256(validation),
      scraped_sha256: await fileSha256(scraped)
    }));
  } catch (exception) {
    error.textContent = exception.message; error.classList.remove('hidden');
  } finally { $('load').disabled = false; }
};

$('resume-select').onchange = () => {
  const option = $('resume-select').selectedOptions[0];
  $('resume-review').disabled = !option?.value || option.disabled;
};

$('resume-review').onclick = async () => {
  const error = $('setup-error'); error.classList.add('hidden');
  const reviewId = $('resume-select').value;
  if (!reviewId) return;
  try {
    $('resume-review').disabled = true;
    start(await api('/api/resume', {review_id: reviewId}));
  } catch (exception) {
    error.textContent = exception.message; error.classList.remove('hidden');
  } finally { $('resume-review').disabled = false; }
};

$('change-inputs').onclick = showSetup;
$('show-score').onclick = async () => {
  if (!session) return;
  $('review').classList.add('hidden');
  $('score').classList.remove('hidden');
  await renderScore();
};

const runLabels = {
  extraction_mode: 'Extraction mode', model_identifier: 'Model identifier',
  provider: 'Provider', context_length: 'Context length',
  software_commit: 'Software commit', run_date: 'Run date',
  numeric_matching_rule: 'Numeric matching rule / tolerance'
};

async function renderScore() {
  let report;
  try {
    report = await api('/api/score', {fingerprint: session.fingerprint, decisions});
  } catch (error) {
    $('score').innerHTML = `<div class="error">${esc(error.message)}</div><button id="back-review" class="secondary">Back to review</button>`;
    $('back-review').onclick = () => { $('score').classList.add('hidden'); $('review').classList.remove('hidden'); };
    return;
  }
  const score = $('score');
  const metadata = Object.entries(runLabels).map(([key, label]) =>
    `<div><label for="run-${key}">${label}</label>${key === 'numeric_matching_rule' ?
      `<textarea id="run-${key}" data-run="${key}" placeholder="For example: reported values in eV, with absolute tolerance …">${esc(decisions.run[key] || '')}</textarea>` :
      `<input id="run-${key}" data-run="${key}" type="${key === 'run_date' ? 'date' : key === 'context_length' ? 'number' : 'text'}" value="${esc(decisions.run[key] || '')}">`}</div>`).join('');
  const label = field => ({'Material system': 'Material system', 'Band gap': 'Band gap, this study',
    'All band gaps': 'All gaps, this study',
    'Cited literature band gaps': 'Gaps cited from literature'})[field] || field;
  const ratio = (value, complete) => value === null ? (complete ? 'undefined' : 'pending') : (100 * value).toFixed(1) + '%';
  const rows = Object.entries(report.fields).map(([field, result]) => {
    const main = `<tr><th>${esc(label(field))}</th><td>${esc(result.type)}</td><td>${result.tp}</td><td>${result.fp}</td><td>${result.fn}</td><td>${result.pending}</td><td>${ratio(result.precision, result.complete)}</td><td>${ratio(result.recall, result.complete)}</td><td>${ratio(result.f1, result.complete)}</td></tr>`;
    if (!session.structure_fields?.[field]) return main;
    const parts = Object.entries(result.parts || {}).map(([name, part]) =>
      `<tr><th>${esc(name)}</th><td>${part.tp}</td><td>${part.fp}</td><td>${part.fn}</td><td>${part.pending}</td><td>${ratio(part.precision, part.complete)}</td><td>${ratio(part.recall, part.complete)}</td><td>${ratio(part.f1, part.complete)}</td></tr>`).join('');
    return main + `<tr><td colspan="9" class="part-stat-cell"><details><summary>Individual parts of ${esc(label(field))}</summary><table><thead><tr><th>Part</th><th>TP</th><th>FP</th><th>FN</th><th>Pending</th><th>Precision</th><th>Recall</th><th>F₁</th></tr></thead><tbody>${parts || '<tr><td colspan="8">No populated parts</td></tr>'}</tbody></table></details></td></tr>`;
  }).join('');
  const sources = Object.entries(report.sources).map(([key, source]) =>
    `<li>${esc(key)}: ${esc(source.name)} <code>SHA-256 ${esc(source.sha256)}</code></li>`).join('');
  score.innerHTML = `<div class="toolbar"><div><h2>Scoring run</h2><p class="muted">${report.complete ? 'Complete and ready to archive.' : 'Draft: finish pending decisions and run details before reporting final metrics.'}</p></div><div class="row"><button id="refresh-score" class="secondary">Refresh totals</button><button id="back-review" class="secondary">Back to review</button></div></div><div class="summary">${report.papers_evaluated} validation papers · ${report.pending_pairs} unresolved entry/paper decisions · ${report.missing_metadata.length} missing run details</div><h3>Run details</h3><div class="setup-grid">${metadata}</div><h3>Field metrics</h3><div class="table-wrap"><table><thead><tr><th>Field</th><th>Type</th><th>TP</th><th>FP</th><th>FN</th><th>Pending</th><th>Precision</th><th>Recall</th><th>F₁</th></tr></thead><tbody>${rows}</tbody></table></div><p class="muted tiny">Each populated scalar part is scored separately. Incorrect paired parts count one FP and one FN; missing parts count FN. A correct extra counts TP and an incorrect extra counts FP. Undefined denominators display as undefined after review is complete.</p><h3>Archived inputs</h3><ul class="source-list">${sources}</ul><p class="muted tiny">${esc(report.protocol)}</p>`;
  $('back-review').onclick = () => { $('score').classList.add('hidden'); $('review').classList.remove('hidden'); render(); };
  $('refresh-score').onclick = renderScore;
}

$('export').onclick = async () => {
  if (!session) return;
  clearTimeout(saveTimer);
  await save();
  const score = await api('/api/score', {fingerprint: session.fingerprint, decisions});
  const blob = new Blob([JSON.stringify({score, decisions}, null, 2) + '\n'], {type: 'application/json'});
  const anchor = document.createElement('a');
  anchor.href = URL.createObjectURL(blob);
  anchor.download = 'pmt-validation-' + (score.complete ? 'scoring-run-' : 'draft-') + session.fingerprint.slice(0, 12) + '.json';
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
};

api('/api/config').then(data => {
  const selected = $('recipe-select').value;
  $('recipe-select').innerHTML = '<option value="">Choose recipe…</option>' +
    data.recipes.map(recipe => `<option value="${esc(recipe)}">${esc(recipe)}</option>`).join('') +
    '<option value="__custom__">Custom recipe JSON…</option>';
  $('recipe-select').value = selected || '';
  refreshReviewSelector(data.reviews || []);
  if (data.session) start(data.session);
}).catch(error => {
  $('setup-error').textContent = error.message;
  $('setup-error').classList.remove('hidden');
});
