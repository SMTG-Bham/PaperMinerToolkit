"""Plan a scrape run without calling a model or changing the corpus.

:func:`plan_scrape` selects papers exactly as :func:`scrape_papers` would,
then reports what a real run would do with each: which stages it would skip
and why, how many model requests each text input and supporting-information
file needs, and how many input tokens they carry. It reads the corpus through
a read-only connection and counts tokens offline, so a plan costs nothing and
leaves the corpus byte for byte unchanged.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Mapping
from os import PathLike
from typing import Any

from paperminertoolkit.corpus.database import connect_read_only, paper_rows
from paperminertoolkit.corpus.documents import read_document_text
from paperminertoolkit.corpus.filtering import active_filter_stack, current_filter_statuses
from paperminertoolkit.corpus.supplements import get_supplements
from paperminertoolkit.extraction.compression import compression_config, should_compress_text
from paperminertoolkit.extraction.extract import build_scrape_prompt
from paperminertoolkit.extraction.models import ModelConfig
from paperminertoolkit.extraction.recipes import load_recipe
from paperminertoolkit.extraction.scrape import (IMAGE_EXTRACTION_MODES,
                                                 MAX_SUPPLEMENT_TEXT_REQUESTS,
                                                 SCRAPE_MODES,
                                                 SCRAPE_ORDERS,
                                                 _paper_asset_paths,
                                                 _select_papers,
                                                 _text_chunks)
from paperminertoolkit.extraction.supplements import supplement_parts, supplement_text
from paperminertoolkit.extraction.tokenizer import (count_text_tokens,
                                                    estimated_token_fallbacks,
                                                    offline_token_counting,
                                                    request_token_budget,
                                                    token_count_method,
                                                    usable_input_token_limit)


def _text_plan(text: str, prompt: str, model_config: ModelConfig,
               compression: Any, prompt_tokens: int) -> dict[str, Any]:
    """Measure one text input as a real run would split it."""
    chunks = _text_chunks(text, model_config, prompt=prompt)
    chunk_tokens = [count_text_tokens(chunk, model_config=model_config) for chunk in chunks]
    return {
        'text_tokens': count_text_tokens(text, model_config=model_config),
        'requests': len(chunks),
        'input_tokens': sum(chunk_tokens) + prompt_tokens * len(chunks),
        'largest_request_tokens': max(chunk_tokens, default=0) + prompt_tokens,
        'compress': bool(should_compress_text(text, prompt, model_config, compression)),
    }


def _supplement_plans(conn: Any, paper_id: str, stages: list[str], force: bool,
                      recipe: Mapping[str, Any], model_config: ModelConfig,
                      compression: Any, totals: dict[str, int],
                      overflows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe what a real run would do with each supporting-information file."""
    plans = []
    seen = set()
    prompt = build_scrape_prompt(recipe, source='text')
    for asset in get_supplements(conn, paper_id, include_content=True):
        if asset['blob_id'] in seen:
            continue
        seen.add(asset['blob_id'])
        metadata = asset.get('metadata') or {}
        label = str(metadata.get('label') or asset.get('original_filename') or asset['blob_id'])
        plan: dict[str, Any] = {'blob_id': asset['blob_id'], 'label': label, 'stages': {}}
        try:
            parts = supplement_parts(asset)
            parse_error = ''
        except Exception as error:
            parts, parse_error = [], str(error)
        for stage in stages:
            previous = metadata.get(f'{stage}_extraction_status')
            if not force and previous in {'succeeded', 'not_applicable'}:
                plan['stages'][stage] = {'action': 'skip', 'reason': f'already {previous}'}
                totals['supplement_files_skipped'] += 1
                continue
            if parse_error:
                plan['stages'][stage] = {'action': 'fail', 'reason': parse_error}
                continue
            members = dict(metadata.get(f'{stage}_extraction_members') or {}) if not force else {}
            member_plans = []
            for part in parts:
                key = part.member or part.name
                entry: dict[str, Any] = {'member': key}
                if not force and members.get(key) == 'succeeded':
                    entry.update(action='skip', reason='already succeeded')
                elif part.kind in {'unsupported', 'zip'}:
                    entry.update(action='skip', reason='unsupported format')
                elif not (part.supports_text if stage == 'text' else part.supports_images):
                    entry.update(action='skip', reason=f'no {stage} content')
                elif stage == 'image':
                    entry.update(action='scrape', reason='image tokens are not estimated')
                    totals['image_inputs'] += 1
                else:
                    reference = f'corpus:supplement:{asset["blob_id"]}'
                    description = f'Supporting information: {label}\nFile: {part.name}\nAsset: {reference}'
                    try:
                        text = supplement_text(part)
                    except Exception as error:
                        entry.update(action='fail', reason=str(error))
                        member_plans.append(entry)
                        continue
                    if not text.strip():
                        entry.update(action='fail', reason='no extractable text')
                        member_plans.append(entry)
                        continue
                    full_prompt = f'{prompt}\n{description}'
                    measured = _text_plan(
                        text, full_prompt, model_config, compression,
                        count_text_tokens(full_prompt, model_config=model_config))
                    entry.update(measured)
                    if measured['requests'] > MAX_SUPPLEMENT_TEXT_REQUESTS:
                        entry.update(action='skip', reason=(
                            f'too large: {measured["requests"]} requests exceed the limit of '
                            f'{MAX_SUPPLEMENT_TEXT_REQUESTS}'))
                        overflows.append({'paper_id': paper_id, 'item': f'supplement {label}: {key}',
                                          'requests': measured['requests'],
                                          'text_tokens': measured['text_tokens'],
                                          'scraped': False})
                    else:
                        entry['action'] = 'scrape'
                        totals['requests'] += measured['requests']
                        totals['input_tokens'] += measured['input_tokens']
                        totals['supplement_files_to_scrape'] += 1
                        if measured['requests'] > 1:
                            overflows.append({'paper_id': paper_id,
                                              'item': f'supplement {label}: {key}',
                                              'requests': measured['requests'],
                                              'text_tokens': measured['text_tokens'],
                                              'scraped': True})
                member_plans.append(entry)
            plan['stages'][stage] = {'action': 'scrape', 'members': member_plans}
        plans.append(plan)
    return plans


def plan_scrape(db_path: str | PathLike[str] = 'papers.db',
                recipe: str = 'sse',
                mode: str = 'text',
                model: str | None = None,
                provider: str | None = None,
                base_url: str | None = None,
                force: bool = False,
                ignore_filters: bool = False,
                scrape_count: int | None = None,
                scrape_order: str = 'corpus',
                compression_scope: str = 'none',
                compression_mode: str = 'auto',
                compression_ratio: float | str = 'auto',
                compression_content_detection: bool = True,
                include_supplements: bool = True,
                image_extraction: str = 'auto',
                paper_ids: Iterable[str] | None = None,
                ) -> dict[str, Any]:
    """Report what :func:`scrape_papers` would do, without doing any of it.

    Paper selection matches a real run: active filters (unless
    ``ignore_filters`` or explicit ``paper_ids``), ``scrape_order`` and
    ``scrape_count``. Text inputs are read from the corpus and split with the
    same chunker and token budget as a real run, but token counting is
    offline (see :func:`offline_token_counting`), no model is called, and
    compression is only predicted, so token counts are before compression.
    Image inputs are listed but not extracted or measured.

    Parameters
    ----------
    db_path : str or os.PathLike[str], optional
        Existing, current-schema corpus database. It is opened read-only.
    recipe : str, optional
        Bundled recipe name or recipe JSON path.
    mode : str, optional
        Scrape mode, as for :func:`scrape_papers`.
    model, provider, base_url : str or None, optional
        Text model overrides, as for :func:`scrape_papers`.
    force : bool, optional
        Plan to redo stages that already succeeded.
    ignore_filters : bool, optional
        Plan to scrape papers that active filters exclude.
    scrape_count : int or None, optional
        Maximum number of papers to select.
    scrape_order : str, optional
        Selection order applied before ``scrape_count``.
    compression_scope, compression_mode : str, optional
        Compression settings used to predict whether text would be compressed.
    compression_ratio : float or str, optional
        Target compression ratio, or ``"auto"``.
    compression_content_detection : bool, optional
        Headroom content detection setting.
    include_supplements : bool, optional
        Whether text and image modes would also process SI.
    image_extraction : str, optional
        Image selection strategy, validated as for :func:`scrape_papers`.
    paper_ids : Iterable[str] or None, optional
        Only consider these papers; active filters do not apply to them.

    Returns
    -------
    dict[str, Any]
        ``dry_run`` (always true), ``mode``, ``recipe``, the text ``model``
        (provider, name, input token limit, usable limit), the ``tokenizer``
        used and ``tokenizer_fallbacks``, ``papers_selected``,
        ``papers_filtered_out``, ``totals`` (papers to scrape and skip by
        reason, model ``requests``, ``input_tokens``, SI files and image
        inputs), ``overflows`` (inputs needing several requests or over the SI
        request limit), and ``papers`` with each paper's stage plans.

    Raises
    ------
    ValueError
        If an option is invalid.
    RuntimeError
        If the corpus schema does not match this package.
    FileNotFoundError
        If the corpus does not exist.
    """
    mode = mode.lower()
    if mode not in SCRAPE_MODES:
        raise ValueError(f'mode must be one of: {", ".join(sorted(SCRAPE_MODES))}')
    if image_extraction.lower() not in IMAGE_EXTRACTION_MODES:
        raise ValueError(f'image_extraction must be one of: {", ".join(sorted(IMAGE_EXTRACTION_MODES))}')
    scrape_order = str(scrape_order).lower()
    if scrape_order not in SCRAPE_ORDERS:
        raise ValueError(f'scrape_order must be one of: {", ".join(sorted(SCRAPE_ORDERS))}')
    compression = compression_config(compression_scope, compression_mode, ratio=compression_ratio,
                                     content_detection=compression_content_detection)
    recipe_data = load_recipe(recipe)
    text_config = ModelConfig.from_profile('text', name=model, provider=provider, base_url=base_url)
    text_stage = 'abstract' if mode == 'abstract' else ('text' if mode in {'text', 'text-images'} else None)
    image_stage = mode in {'images', 'text-images'}
    si_stages = [] if mode == 'abstract' or not include_supplements else [
        stage for stage in ('text', 'image')
        if mode == 'text-images' or mode == ('images' if stage == 'image' else stage)]
    totals = {'papers_to_scrape': 0, 'skipped_done': 0, 'skipped_missing_source': 0,
              'requests': 0, 'input_tokens': 0, 'image_inputs': 0,
              'supplement_files_to_scrape': 0, 'supplement_files_skipped': 0}
    overflows: list[dict[str, Any]] = []
    papers_out: list[dict[str, Any]] = []
    fallbacks_before = estimated_token_fallbacks()
    with offline_token_counting(), connect_read_only(db_path) as conn:
        papers = paper_rows(conn)
        total = len(papers)
        if paper_ids is not None:
            selected = set(paper_ids)
            papers = [paper for paper in papers if paper['paper_id'] in selected]
        elif active_filter_stack(conn) and not ignore_filters:
            statuses = current_filter_statuses(conn)
            papers = [paper for paper in papers if statuses.get(paper['paper_id']) == 'included']
        filtered_out = total - len(papers) if paper_ids is None else 0
        papers = _select_papers(papers, scrape_order=scrape_order, scrape_count=scrape_count)
        prompt = build_scrape_prompt(recipe_data, source='text')
        prompt_tokens = count_text_tokens(prompt, model_config=text_config)
        text_budget = request_token_budget(prompt, text_config)
        with tempfile.TemporaryDirectory(prefix='paperminertoolkit-plan-') as temp_dir:
            for row in papers:
                paths = _paper_asset_paths(conn, row, temp_dir)
                stages: dict[str, Any] = {}
                scrapes = False
                if text_stage is not None:
                    if not force and row.get(f'{text_stage}_scrape_status') == 'succeeded':
                        stages[text_stage] = {'action': 'skip', 'reason': 'already succeeded'}
                        totals['skipped_done'] += 1
                    else:
                        source = paths['abstract'] if text_stage == 'abstract' else (
                            paths['text'] or paths['pdf'])
                        if not source:
                            stages[text_stage] = {'action': 'fail', 'reason': (
                                'no downloaded abstract' if text_stage == 'abstract'
                                else 'no downloaded text or PDF; run pmt download')}
                            totals['skipped_missing_source'] += 1
                        else:
                            text = read_document_text(source)
                            measured = _text_plan(text, prompt, text_config, compression, prompt_tokens)
                            stages[text_stage] = {'action': 'scrape', 'source': (
                                'abstract' if text_stage == 'abstract'
                                else 'text' if paths['text'] else 'pdf'), **measured}
                            totals['requests'] += measured['requests']
                            totals['input_tokens'] += measured['input_tokens']
                            scrapes = True
                            if measured['requests'] > 1:
                                overflows.append({'paper_id': row['paper_id'], 'item': text_stage,
                                                  'requests': measured['requests'],
                                                  'text_tokens': measured['text_tokens'],
                                                  'scraped': True})
                if image_stage:
                    if not force and row.get('image_scrape_status') == 'succeeded':
                        stages['image'] = {'action': 'skip', 'reason': 'already succeeded'}
                    elif not paths['pdf'] and image_extraction != 'layout':
                        stages['image'] = {'action': 'fail', 'reason': 'no downloaded PDF'}
                    else:
                        stages['image'] = {'action': 'scrape', 'reason': 'image tokens are not estimated'}
                        totals['image_inputs'] += 1
                        scrapes = True
                supplements = (_supplement_plans(conn, row['paper_id'], si_stages, force, recipe_data,
                                                 text_config, compression, totals, overflows)
                               if si_stages else [])
                if any(stage.get('action') == 'scrape' and any(
                        member.get('action') == 'scrape' for member in stage.get('members', []))
                       for plan in supplements for stage in plan['stages'].values()):
                    scrapes = True
                totals['papers_to_scrape'] += scrapes
                papers_out.append({'paper_id': row['paper_id'], 'title': row.get('title') or '',
                                   'stages': stages, 'supplements': supplements})
    fallbacks = sorted(f'{provider_name}/{model_name}' for provider_name, model_name
                       in estimated_token_fallbacks() - fallbacks_before)
    return {
        'dry_run': True,
        'mode': mode,
        'recipe': recipe,
        'model': {'provider': text_config.provider, 'name': text_config.name,
                  'input_token_limit': text_config.input_token_limit,
                  'usable_input_tokens': usable_input_token_limit(text_config),
                  'request_text_budget': text_budget},
        'tokenizer': token_count_method(text_config, offline=True),
        'tokenizer_fallbacks': fallbacks,
        'prompt_tokens': prompt_tokens,
        'papers_selected': len(papers),
        'papers_filtered_out': filtered_out,
        'totals': totals,
        'overflows': overflows,
        'papers': papers_out,
    }
