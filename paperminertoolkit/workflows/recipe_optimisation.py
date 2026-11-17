"""Iteratively refine a recipe against a hand-curated validation CSV.

Each round scrapes the validation papers with the current recipe, scores the
records automatically against the validation set, and asks a language model to
revise the recipe's wording using the observed mismatches. A revision is kept
only when it improves the selection score, so the best recipe never regresses.

The automatic score approximates ``pmt validate``: records are paired within a
paper by the recipe's identity fields, and each populated scalar value, list
item, or dictionary key contributes one true positive, false positive, or false
negative. Unlike the human review, every unpaired scraped value counts as a
false positive because extras cannot be judged automatically.
"""

from __future__ import annotations

import copy
import csv
import json
import random
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from difflib import SequenceMatcher
from os import PathLike
from pathlib import Path
from typing import Any

from paperminertoolkit.corpus.database import connect, paper_rows
from paperminertoolkit.corpus.documents import read_document_text
from paperminertoolkit.extraction.extract import (_json_decoder_scan,
                                                  _strip_fences,
                                                  build_scrape_prompt,
                                                  scrape_text)
from paperminertoolkit.extraction.models import ModelConfig, query_text
from paperminertoolkit.extraction.recipes import _validate_recipe, load_recipe
from paperminertoolkit.extraction.scrape import _paper_asset_paths, _text_chunks
from paperminertoolkit.workflows.validation import (_csv_rows,
                                                    _material_suggestions,
                                                    _name,
                                                    _paper_suggestions,
                                                    _papers)
from paperminertoolkit.workflows.validation_scoring import _ABSENT, _decode

OPTIMISE_MODES = {'abstract', 'text'}
_NUMBER = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
_DECIMAL = re.compile(r'\d+\.\d+')
_TEXT_NUMBER = re.compile(r'\d+(?:\.\d+)?')
_ALT_DECIMAL = re.compile(r'\d+[·,]\d+')
_EDITABLE_FIELD_KEYS = ('prompt', 'example')

OPTIMISER_PROMPT = """You improve extraction recipes for PaperMinerToolkit, a tool that asks a language model to extract structured records from scientific papers.

A recipe is a JSON object with:
- "record definition": "subject", "singular", "plural", "unit" (what one output record represents), and "identity fields".
- "additional prompts": recipe-wide extraction rules inserted verbatim into the extraction prompt.
- "search fields": the output schema. Each field has a "prompt" describing what belongs in it and an "example" value whose JSON type shows the required output shape.

You will receive the current recipe, the exact extraction prompt it produces, field-level scores against a hand-curated validation set, mismatches between validation and scraped records, and earlier attempts with their scores.

Revise the recipe so that the extraction model reproduces the validation records more faithfully. Rules:
- Return the complete recipe. Keep every "search fields" key, in the same order, and keep "identity fields" unchanged. Do not add or remove fields.
- You may edit "subject", "singular", "plural", "unit", "additional prompts", and each field's "prompt" and "example". Other keys are restored automatically.
- Write general extraction rules. Do not copy paper-specific materials, values, or identifiers from the mismatches into the recipe; the recipe must work on unseen papers.
- Address the largest sources of error first: missing records, extra records, wrong record boundaries, and value formats that differ from the validation set.
- Prefer targeted edits over rewriting text that already works. Do not repeat an earlier attempt that scored worse.
- Expected values whose numbers never appear in the paper text are listed under "not_in_text" and excluded from the scores. They probably come from tables, figures, or supporting information that the extraction model cannot see, so do not add rules aimed at recovering them.

Return only JSON of the form {"rationale": "<one short paragraph>", "recipe": {...}}. Do not include markdown fences or prose outside the JSON."""


def _counts() -> dict[str, int]:
    """Create one field's count accumulator."""
    return {'tp': 0, 'fp': 0, 'fn': 0, 'unreachable': 0}


def _metrics(count: Mapping[str, int]) -> dict[str, Any]:
    """Calculate precision, recall, and F1 from counts.

    Parameters
    ----------
    count : Mapping[str, int]
        True-positive, false-positive, false-negative, and unreachable counts.

    Returns
    -------
    dict[str, Any]
        The counts with precision, recall, and F1, or ``None`` where a ratio
        has no denominator. ``f1`` ignores unreachable expected values;
        ``f1_including_unreachable`` counts them as false negatives.
    """
    tp, fp, fn = count['tp'], count['fp'], count['fn']
    unreachable = count.get('unreachable', 0)

    def f1(recall: float | None) -> float | None:
        """Combine precision with a recall value."""
        if precision is None or recall is None:
            return None
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    recall_all = tp / (tp + fn + unreachable) if tp + fn + unreachable else None
    return {'tp': tp, 'fp': fp, 'fn': fn, 'unreachable': unreachable, 'precision': precision,
            'recall': recall, 'f1': f1(recall), 'f1_including_unreachable': f1(recall_all)}


def _leaves(value: object, prefix: str = '') -> list[str]:
    """Flatten a decoded cell into comparable scalar parts.

    Parameters
    ----------
    value : object
        Decoded CSV cell or scraped value.
    prefix : str, optional
        Dictionary key path for nested values.

    Returns
    -------
    list[str]
        One entry per populated scalar, labelled with its dictionary key path.
    """
    if value is _ABSENT or value is None:
        return []
    if isinstance(value, list):
        return [leaf for item in value for leaf in _leaves(item, prefix)]
    if isinstance(value, dict):
        return [leaf for key, item in value.items() for leaf in _leaves(item, f'{prefix}{key}=')]
    text = str(value).strip()
    if text.casefold() in {'', 'none', 'null', 'nan'}:
        return []
    return [f'{prefix}{text}']


def _text_numbers(text: str | None) -> set[float] | None:
    """Collect every number printed in a source text, or ``None`` without text.

    Decimals written with a middle dot or comma, such as ``2·70``, are read as
    decimals too.
    """
    if text is None:
        return None
    numbers = {float(number) for number in _TEXT_NUMBER.findall(text)}
    numbers.update(float(re.sub('[·,]', '.', number)) for number in _ALT_DECIMAL.findall(text))
    return numbers


def _in_text(number: float, numbers: set[float], tolerance: float) -> bool:
    """Whether the text prints a number within the scoring tolerance."""
    return number in numbers or any(abs(number - candidate) <= tolerance * max(abs(number), abs(candidate))
                                    for candidate in numbers)


def _gold_parts(value: object, numbers: set[float] | None,
                tolerance: float = 0.0) -> list[tuple[str, bool]]:
    """Flatten a validation cell and mark which parts the source text supports.

    Parameters
    ----------
    value : object
        Decoded validation cell.
    numbers : set[float] or None
        Numbers printed in the paper text, or ``None`` to treat every part as
        reachable.
    tolerance : float, optional
        Relative tolerance, matching the scorer's numeric tolerance, so a value
        printed rounded in the text still counts as reachable.

    Returns
    -------
    list[tuple[str, bool]]
        Each scalar part with whether it is reachable. A part is unreachable
        when a decimal number it contains never appears in the text. All parts
        of one dictionary item share a flag, so a band-gap item whose value is
        missing from the text does not leave its method and gap type behind as
        false negatives.
    """
    items = value if isinstance(value, list) else [value]
    parts = []
    for item in items:
        leaves = _leaves(item)
        decimals = [float(d) for leaf in leaves for d in _DECIMAL.findall(leaf)]
        reachable = numbers is None or all(_in_text(number, numbers, tolerance) for number in decimals)
        parts.extend((leaf, reachable) for leaf in leaves)
    return parts


def values_match(gold: str, scraped: str, numeric_tolerance: float = 0.02,
                 text_threshold: float = 0.9) -> bool:
    """Decide whether two scalar values agree.

    Parameters
    ----------
    gold : str
        Validation value.
    scraped : str
        Scraped value.
    numeric_tolerance : float, optional
        Relative tolerance applied when both values contain the same count of
        numbers and otherwise agree in their wording.
    text_threshold : float, optional
        Minimum normalized string similarity for a textual match.

    Returns
    -------
    bool
        Whether the values are treated as the same answer.
    """
    left, right = _name(gold), _name(scraped)
    if left == right:
        return True
    left_numbers, right_numbers = _NUMBER.findall(gold), _NUMBER.findall(scraped)
    if left_numbers and len(left_numbers) == len(right_numbers):
        close = all(abs(float(a) - float(b)) <= numeric_tolerance * max(abs(float(a)), abs(float(b)), 1e-12)
                    for a, b in zip(left_numbers, right_numbers))
        left_words = _name(_NUMBER.sub(' ', gold))
        right_words = _name(_NUMBER.sub(' ', scraped))
        if close and (left_words == right_words or not left_words or not right_words):
            return True
        if not close:
            return False
    return SequenceMatcher(None, left, right).ratio() >= text_threshold


def _compare_field(gold: object, scraped: object, numeric_tolerance: float, text_threshold: float,
                   numbers: set[float] | None = None) -> tuple[dict[str, int], list[str]]:
    """Score one paired field by greedy one-to-one matching of its parts.

    Returns the counts and the unmatched expected parts that are unreachable.
    A matched unreachable part still counts as a true positive.
    """
    unused = _leaves(_decode(scraped))
    count = _counts()
    missed = []
    for part, reachable in _gold_parts(_decode(gold), numbers, numeric_tolerance):
        for index, candidate in enumerate(unused):
            if values_match(part, candidate, numeric_tolerance, text_threshold):
                count['tp'] += 1
                del unused[index]
                break
        else:
            if reachable:
                count['fn'] += 1
            else:
                count['unreachable'] += 1
                missed.append(part)
    count['fp'] = len(unused)
    return count, missed


def _cell(value: object) -> str:
    """Render a scraped value as the CSV text used by the validation reader."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return '' if value is None else str(value)


def _pairing_rows(rows: Sequence[Mapping[str, Any]], identity: Sequence[str]) -> list[dict[str, Any]]:
    """Sort identity tokens so pairing ignores word order, as in ``X (monoclinic)``."""
    return [{'id': row['id'],
             'values': {field: ' '.join(sorted(_name(row['values'].get(field, '')).split())) for field in identity}}
            for row in rows]


def _clip(value: str, limit: int = 300) -> str:
    """Shorten long values for the optimiser prompt."""
    return value if len(value) <= limit else value[:limit] + '…'


def score_paper_records(gold_rows: Sequence[Mapping[str, Any]],
                        scraped_records: Sequence[Mapping[str, Any]],
                        recipe: Mapping[str, Any],
                        numeric_tolerance: float = 0.02,
                        text_threshold: float = 0.9,
                        source_text: str | None = None) -> dict[str, Any]:
    """Score one paper's scraped records against its validation rows.

    Parameters
    ----------
    gold_rows : Sequence[Mapping[str, Any]]
        Validation rows from :func:`paperminertoolkit.workflows.validation._csv_rows`.
    scraped_records : Sequence[Mapping[str, Any]]
        Records returned by the extraction model.
    recipe : Mapping[str, Any]
        Recipe whose search fields are scored.
    numeric_tolerance : float, optional
        Relative tolerance for numeric values.
    text_threshold : float, optional
        Minimum normalized similarity for textual values.
    source_text : str or None, optional
        Paper text given to the extraction model. When supplied, expected
        values whose numbers never appear in it count as unreachable rather
        than as false negatives.

    Returns
    -------
    dict[str, Any]
        Per-field counts, record pairing counts, and mismatch examples.
    """
    fields = list(recipe['search fields'])
    numbers = _text_numbers(source_text)
    identity = list(recipe['record definition']['identity fields']) or fields
    scraped_rows = [{'id': f's:{index}', 'values': {field: _cell(record.get(field)) for field in fields}}
                    for index, record in enumerate(scraped_records)]
    pairs = _material_suggestions(_pairing_rows(gold_rows, identity), _pairing_rows(scraped_rows, identity), identity)
    by_id = {row['id']: row for row in scraped_rows}
    counts = {field: _counts() for field in fields}
    mismatches: list[dict[str, Any]] = []

    def label(row: Mapping[str, Any]) -> str:
        """Describe a record by its identity values for the optimiser."""
        return ', '.join(_clip(str(row['values'].get(field) or ''), 80) for field in identity)

    for gold in gold_rows:
        scraped = by_id.get(pairs.get(gold['id'], ''))
        if scraped is None:
            for field in fields:
                for _, reachable in _gold_parts(_decode(gold['values'][field]), numbers, numeric_tolerance):
                    counts[field]['fn' if reachable else 'unreachable'] += 1
            mismatches.append({'kind': 'missing record', 'record': label(gold)})
            continue
        for field in fields:
            result, missed = _compare_field(gold['values'][field], scraped['values'][field],
                                            numeric_tolerance, text_threshold, numbers)
            for key, value in result.items():
                counts[field][key] += value
            if result['fp'] or result['fn']:
                mismatch = {'kind': 'field mismatch', 'record': label(gold), 'field': field,
                            'expected': _clip(str(gold['values'][field])),
                            'scraped': _clip(scraped['values'][field])}
                if missed:
                    mismatch['not_in_text'] = [_clip(part, 80) for part in missed]
                mismatches.append(mismatch)
    paired = set(pairs.values())
    for row in scraped_rows:
        if row['id'] in paired:
            continue
        for field in fields:
            counts[field]['fp'] += len(_leaves(_decode(row['values'][field])))
        mismatches.append({'kind': 'extra record', 'record': label(row)})
    return {'counts': counts, 'expected': len(gold_rows), 'scraped': len(scraped_rows),
            'paired': len(pairs), 'mismatches': mismatches}


def _summarise(paper_scores: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    """Aggregate per-paper scores into field and overall metrics."""
    totals = {field: _counts() for field in fields}
    records = {'expected': 0, 'scraped': 0, 'paired': 0}
    for score in paper_scores:
        for field in fields:
            for key in totals[field]:
                totals[field][key] += score['counts'][field][key]
        for key in records:
            records[key] += score[key]
    overall = _counts()
    for count in totals.values():
        for key in overall:
            overall[key] += count[key]
    return {'overall': _metrics(overall), 'fields': {field: _metrics(totals[field]) for field in fields},
            'records': records, 'papers': len(paper_scores)}


def _merge_recipe(original: Mapping[str, Any], proposed: object) -> dict[str, Any]:
    """Apply only the editable parts of a proposed recipe.

    Parameters
    ----------
    original : Mapping[str, Any]
        Recipe being revised.
    proposed : object
        Recipe returned by the optimiser model.

    Returns
    -------
    dict[str, Any]
        Validated recipe with the original schema, identity fields, aliases,
        and units, and the proposed wording.

    Raises
    ------
    ValueError
        If the proposal is not a recipe or changes the search-field keys.
    """
    if not isinstance(proposed, dict):
        raise ValueError('Optimiser did not return a recipe object.')
    proposed_fields = proposed.get('search fields', {})
    if not isinstance(proposed_fields, dict):
        raise ValueError('Optimiser returned "search fields" that is not an object; the proposal was rejected.')
    unknown = [field for field in proposed_fields if field not in original['search fields']]
    if unknown:
        raise ValueError(f'Optimiser added or renamed search field(s) {", ".join(map(repr, unknown))}; '
                         'the proposal was rejected.')
    merged = copy.deepcopy(dict(original))
    definition = proposed.get('record definition')
    if isinstance(definition, dict):
        for key in ('subject', 'singular', 'plural', 'unit'):
            if isinstance(definition.get(key), str) and definition[key].strip():
                merged['record definition'][key] = definition[key]
    if isinstance(proposed.get('additional prompts'), str):
        merged['additional prompts'] = proposed['additional prompts']
    for field, config in proposed_fields.items():
        if not isinstance(config, dict):
            continue
        for key in _EDITABLE_FIELD_KEYS:
            if key in config:
                merged['search fields'][field][key] = config[key]
    return _validate_recipe(merged, 'optimiser proposal')


def _parse_proposal(response: str) -> tuple[object, str]:
    """Extract the proposed recipe and rationale from the optimiser response."""
    text = _strip_fences(response)
    try:
        candidates = [json.loads(text)]
    except json.JSONDecodeError:
        candidates = _json_decoder_scan(text)
    for candidate in candidates:
        if isinstance(candidate, dict) and 'recipe' in candidate:
            return candidate['recipe'], str(candidate.get('rationale') or '')
        if isinstance(candidate, dict) and 'search fields' in candidate:
            return candidate, ''
    raise ValueError('Optimiser response did not contain a recipe.')


def propose_recipe(recipe: Mapping[str, Any],
                   report: Mapping[str, Any],
                   mismatches: Sequence[Mapping[str, Any]],
                   history: Sequence[Mapping[str, Any]],
                   model_config: ModelConfig | None = None,
                   max_examples: int = 40) -> tuple[dict[str, Any], str]:
    """Ask a language model for a revised recipe.

    Parameters
    ----------
    recipe : Mapping[str, Any]
        Best recipe found so far.
    report : Mapping[str, Any]
        Score summary for ``recipe`` on the papers the optimiser may see.
    mismatches : Sequence[Mapping[str, Any]]
        Mismatch examples for ``recipe`` on those papers.
    history : Sequence[Mapping[str, Any]]
        Earlier attempts with their rationale and selection score.
    model_config : ModelConfig or None, optional
        Optimiser model configuration.
    max_examples : int, optional
        Maximum mismatch examples included in the request.

    Returns
    -------
    tuple[dict[str, Any], str]
        Validated revised recipe and the model's rationale.
    """
    payload = {
        'current_recipe': recipe,
        'extraction_prompt': build_scrape_prompt(recipe, source='text'),
        'scores': report,
        'mismatches': list(mismatches[:max_examples]),
        'omitted_mismatches': max(0, len(mismatches) - max_examples),
        'earlier_attempts': [{key: attempt.get(key) for key in ('iteration', 'score', 'accepted', 'rationale', 'error')}
                             for attempt in history],
    }
    messages = [
        {'role': 'system', 'content': OPTIMISER_PROMPT},
        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, indent=1)},
    ]
    try:
        response = query_text(messages, config=model_config, max_output_tokens=16000)
    except Exception as error:
        raise RuntimeError(f'Optimiser model request failed: {error}') from error
    proposed, rationale = _parse_proposal(response)
    return _merge_recipe(recipe, proposed), rationale


def _load_validation_papers(validation_path: str | PathLike[str],
                            recipe: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read the validation CSV and group its rows by paper."""
    content = Path(validation_path).read_bytes().decode('utf-8-sig')
    fields = list(recipe['search fields'])
    identity = list(recipe['record definition']['identity fields'])
    rows = _csv_rows(content, 'Validation', recipe)
    return _papers(rows, 'v', fields, identity)


def _match_corpus(papers: list[dict[str, Any]], corpus: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Pair validation papers with corpus rows using the reviewer's metadata rules."""
    candidates = [{'id': str(row['paper_id']),
                   'meta': {'doi': str(row.get('doi') or ''),
                            'paper_id': str(row['paper_id']),
                            'title': str(row.get('title') or '')}}
                  for row in corpus]
    _paper_suggestions(papers, candidates)
    by_id = {str(row['paper_id']): row for row in corpus}
    return {paper['id']: by_id[paper['suggested']] for paper in papers if paper.get('suggested')}


def _write_scraped_csv(path: Path, papers: Sequence[Mapping[str, Any]],
                       records: Mapping[str, Sequence[Mapping[str, Any]]],
                       fields: Sequence[str]) -> None:
    """Write scraped records in a CSV that ``pmt validate gui`` can open."""
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Identifier', 'Title', 'DOI', *fields])
        for paper in papers:
            meta = paper['corpus']
            title = paper['meta']['title'] or ' '.join(str(meta.get('title') or '').split())
            for record in records.get(paper['id'], []):
                writer.writerow([meta['paper_id'], title, meta.get('doi') or '',
                                 *(_cell(record.get(field)) for field in fields)])


def _dump(path: Path, payload: object) -> None:
    """Write indented UTF-8 JSON."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def optimise_recipe(db_path: str | PathLike[str],
                    recipe: str,
                    validation_path: str | PathLike[str],
                    output_dir: str | PathLike[str] = 'recipe_optimisation',
                    mode: str = 'text',
                    iterations: int = 5,
                    patience: int = 2,
                    holdout: float = 0.0,
                    seed: int = 0,
                    numeric_tolerance: float = 0.02,
                    text_threshold: float = 0.9,
                    max_examples: int = 40,
                    include_unreachable: bool = False,
                    model: str | None = None,
                    provider: str | None = None,
                    base_url: str | None = None,
                    optimiser_model: str | None = None,
                    optimiser_provider: str | None = None,
                    optimiser_base_url: str | None = None,
                    echo: Callable[[str], None] = print) -> dict[str, Any]:
    """Refine a recipe by repeatedly scraping and scoring the validation papers.

    Parameters
    ----------
    db_path : str or os.PathLike[str]
        Corpus database holding the validation papers' downloaded content.
    recipe : str
        Bundled recipe name or recipe JSON path used as the starting point.
    validation_path : str or os.PathLike[str]
        Hand-curated validation CSV in the ``pmt validate template`` layout.
    output_dir : str or os.PathLike[str], optional
        Directory for each round's recipe, scraped CSV, and score report.
    mode : {'text', 'abstract'}, optional
        Content scraped from each paper.
    iterations : int, optional
        Maximum number of recipe revisions after the baseline round.
    patience : int, optional
        Stop after this many consecutive revisions fail to improve.
    holdout : float, optional
        Fraction of validation papers withheld from the optimiser and used to
        choose between recipes. ``0`` selects on the same papers it learns from.
    seed : int, optional
        Random seed for the holdout split.
    numeric_tolerance : float, optional
        Relative tolerance for numeric values during automatic scoring.
    text_threshold : float, optional
        Minimum normalized similarity for textual values.
    max_examples : int, optional
        Maximum mismatch examples sent to the optimiser per round.
    include_unreachable : bool, optional
        Select recipes by an F1 that counts expected values absent from the
        paper text as false negatives. By default those values are excluded,
        because no recipe can recover them from the text.
    model, provider, base_url : str or None, optional
        Extraction model overrides for the text profile.
    optimiser_model, optimiser_provider, optimiser_base_url : str or None, optional
        Optimiser model overrides; the text profile is used when omitted.
    echo : Callable[[str], None], optional
        Progress reporter.

    Returns
    -------
    dict[str, Any]
        Paths to the best recipe and history, and the baseline and best scores.

    Raises
    ------
    ValueError
        If an option is invalid or no validation paper is in the corpus.
    """
    mode = mode.lower()
    if mode not in OPTIMISE_MODES:
        raise ValueError(f'mode must be one of: {", ".join(sorted(OPTIMISE_MODES))}')
    if iterations < 0 or patience < 1:
        raise ValueError('iterations must be zero or more and patience at least one.')
    if not 0 <= holdout < 1:
        raise ValueError('holdout must be at least 0 and less than 1.')

    current = load_recipe(recipe)
    fields = list(current['search fields'])
    text_config = ModelConfig.from_profile('text', name=model, provider=provider, base_url=base_url)
    optimiser_config = ModelConfig.from_profile('text', name=optimiser_model or model,
                                                provider=optimiser_provider or provider,
                                                base_url=optimiser_base_url or base_url)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    papers = _load_validation_papers(validation_path, current)
    with connect(db_path) as conn:
        matches = _match_corpus(papers, paper_rows(conn))
        unmatched = [paper for paper in papers if paper['id'] not in matches]
        for paper in unmatched:
            meta = paper['meta']
            echo(f'Skipping validation paper not found in the corpus: '
                 f'{meta["doi"] or meta["paper_id"] or meta["title"]}')
        papers = [paper for paper in papers if paper['id'] in matches]
        with tempfile.TemporaryDirectory(prefix='paperminertoolkit-optimise-') as temp_dir:
            for paper in list(papers):
                paper['corpus'] = matches[paper['id']]
                paths = _paper_asset_paths(conn, paper['corpus'], temp_dir)
                source = paths['abstract'] if mode == 'abstract' else paths['text'] or paths['pdf']
                if not source:
                    echo(f'Skipping {paper["corpus"]["paper_id"]}: no downloaded {mode} content.')
                    papers.remove(paper)
                    continue
                paper['text'] = read_document_text(source)
    if not papers:
        raise ValueError('None of the validation papers have downloaded content in the corpus.')

    order = list(papers)
    random.Random(seed).shuffle(order)
    held = order[:round(len(order) * holdout)] if len(order) > 1 else []
    held_ids = {paper['id'] for paper in held}
    train = [paper for paper in papers if paper['id'] not in held_ids]
    echo(f'Optimising on {len(train)} papers'
         + (f', selecting on {len(held)} held-out papers.' if held else '.'))

    def evaluate(candidate: Mapping[str, Any], directory: Path) -> dict[str, Any]:
        """Scrape every paper with a recipe, score it, and save the round."""
        directory.mkdir(parents=True, exist_ok=True)
        prompt = build_scrape_prompt(candidate, source='text')
        records: dict[str, list[dict[str, Any]]] = {}
        scores: dict[str, dict[str, Any]] = {}
        errors = {}
        for paper in papers:
            try:
                records[paper['id']] = [record for chunk in _text_chunks(paper['text'], text_config, prompt=prompt)
                                        for record in scrape_text(chunk, candidate, model_config=text_config)]
            except Exception as error:
                records[paper['id']] = []
                errors[paper['corpus']['paper_id']] = str(error)
            scores[paper['id']] = score_paper_records(paper['rows'], records[paper['id']], candidate,
                                                      numeric_tolerance, text_threshold, paper['text'])
        result = {
            'train': _summarise([scores[paper['id']] for paper in train], fields),
            'holdout': _summarise([scores[paper['id']] for paper in held], fields) if held else None,
            'errors': errors,
            'mismatches': [dict(item, paper=paper['corpus']['paper_id'])
                           for paper in train for item in scores[paper['id']]['mismatches']],
        }
        selection = (result['holdout'] or result['train'])['overall']
        result['score'] = selection['f1_including_unreachable' if include_unreachable else 'f1'] or 0.0
        _dump(directory / 'recipe.json', candidate)
        _dump(directory / 'score.json', {key: result[key] for key in ('score', 'train', 'holdout', 'errors')})
        _write_scraped_csv(directory / 'scraped.csv', papers, records, fields)
        if len(errors) == len(papers):
            raise RuntimeError(f'Every paper failed to scrape, so there is nothing to optimise. '
                               f'First error: {next(iter(errors.values()))}')
        if errors:
            echo(f'Warning: {len(errors)} of {len(papers)} papers failed to scrape and score as empty; '
                 f'see {directory / "score.json"}.')
        return result

    best = evaluate(current, output / 'iteration_00')
    baseline_score = best['score']
    echo(f'Baseline F1: {baseline_score:.3f}')
    overall = best['train']['overall']
    expected = overall['tp'] + overall['fn'] + overall['unreachable']
    if overall['unreachable']:
        echo(f'{overall["unreachable"]} of {expected} expected values on the optimisation papers do not appear '
             f'in the paper text and are {"counted as misses" if include_unreachable else "excluded from scoring"}.')
    history: list[dict[str, Any]] = [{'iteration': 0, 'score': baseline_score, 'accepted': True,
                                      'rationale': 'baseline'}]
    stale = 0
    for iteration in range(1, iterations + 1):
        if best['score'] >= 1.0 or stale >= patience:
            break
        attempt: dict[str, Any] = {'iteration': iteration}
        try:
            candidate, attempt['rationale'] = propose_recipe(current, best['train'], best['mismatches'], history,
                                                              model_config=optimiser_config,
                                                              max_examples=max_examples)
        except ValueError as error:
            attempt.update(score=None, accepted=False, error=str(error))
            echo(f'Iteration {iteration}: proposal rejected ({error}).')
        else:
            result = evaluate(candidate, output / f'iteration_{iteration:02d}')
            attempt['score'] = result['score']
            attempt['accepted'] = result['score'] > best['score']
            echo(f'Iteration {iteration}: F1 {result["score"]:.3f} '
                 + ('(improved, kept).' if attempt['accepted'] else '(not improved, discarded).'))
            if attempt['accepted']:
                current, best = candidate, result
        history.append(attempt)
        stale = 0 if attempt.get('accepted') else stale + 1
        _dump(output / 'history.json', history)

    _dump(output / 'history.json', history)
    _dump(output / 'best_recipe.json', current)
    echo(f'Best F1: {best["score"]:.3f} (baseline {baseline_score:.3f}). '
         f'Best recipe written to {output / "best_recipe.json"}.')
    return {'best_recipe': str(output / 'best_recipe.json'), 'history': str(output / 'history.json'),
            'baseline_score': baseline_score, 'best_score': best['score'], 'papers': len(papers),
            'holdout_papers': len(held)}
