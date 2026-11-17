"""JSON Schemas of the results ``pmt`` commands print with ``--json``.

Each command declares one of these with ``agent_meta(output=...)`` and
``pmt schema`` publishes it as the command's ``output_schema``. Schemas list
the keys a result always carries as ``required`` and allow additional keys, so
new fields can be added without breaking callers; nested provider and
database records are described as plain objects. The test suite validates
every JSON result the commands print against these schemas.
"""

from __future__ import annotations

from typing import Any

STRING: dict[str, Any] = {'type': 'string'}
INTEGER: dict[str, Any] = {'type': 'integer'}
NUMBER: dict[str, Any] = {'type': 'number'}
BOOLEAN: dict[str, Any] = {'type': 'boolean'}
OBJECT: dict[str, Any] = {'type': 'object'}
ARRAY: dict[str, Any] = {'type': 'array'}
OPTIONAL_STRING: dict[str, Any] = {'type': ['string', 'null']}
OPTIONAL_NUMBER: dict[str, Any] = {'type': ['number', 'null']}


def obj(required: dict[str, Any] | None = None,
        optional: dict[str, Any] | None = None) -> dict[str, Any]:
    """Describe an object whose ``required`` keys are always present."""
    required = required or {}
    return {'type': 'object', 'required': list(required),
            'properties': {**required, **(optional or {})}}


def array(items: dict[str, Any]) -> dict[str, Any]:
    """Describe an array whose every item matches ``items``."""
    return {'type': 'array', 'items': items}


def counts(*keys: str) -> dict[str, Any]:
    """Describe an object of integer counts."""
    return obj({key: INTEGER for key in keys})


ERROR = obj({'error': obj({'type': STRING, 'message': STRING, 'exit_code': INTEGER},
                          {'hint': OPTIONAL_STRING})})

SCHEMA = obj({'schema_version': INTEGER, 'package_version': STRING, 'effects': OBJECT,
              'command': obj({'name': STRING, 'path': STRING, 'effects': array(STRING),
                              'parameters': array(OBJECT), 'commands': ARRAY},
                             {'help': OPTIONAL_STRING, 'output_schema': {'type': ['object', 'null']}})},
             {'error_schema': OBJECT})

PROVIDERS = obj({'providers': array(OBJECT), 'ok': BOOLEAN})

SEARCH = obj({'search_id': INTEGER, 'status': STRING, 'sources': ARRAY,
              'source_results': OBJECT, 'paper_ids': array(STRING), 'result_count': INTEGER,
              'papers_added': INTEGER, 'papers_updated': INTEGER, 'abstracts_stored': INTEGER})

DOWNLOAD = counts('texts', 'pdfs', 'abstracts', 'texts_skipped', 'pdfs_skipped',
                  'abstracts_skipped', 'supplements', 'supplements_skipped',
                  'supplements_failed', 'supplements_unavailable',
                  'supplement_discovery_failed')

DOWNLOAD_PLAN = obj({
    'dry_run': {'type': 'boolean', 'enum': [True]}, 'format': STRING, 'sources': array(STRING),
    'supplement_sources': array(STRING), 'papers_considered': INTEGER,
    'would_attempt': counts('texts', 'pdfs', 'abstracts', 'supplement_discovery'),
    'already_stored': counts('texts', 'pdfs', 'abstracts', 'supplement_files'),
    'no_text_source': INTEGER,
    'papers': array(obj({'paper_id': STRING, 'attempt': array(STRING), 'stored': array(STRING)})),
})
DOWNLOAD_OR_PLAN = {'anyOf': [DOWNLOAD, DOWNLOAD_PLAN]}

GATHER = obj({'search': SEARCH, 'downloads': {'anyOf': [DOWNLOAD, {'type': 'null'}]}})

DIGITIZE = obj({'requests': INTEGER, 'supplementary': INTEGER, 'not_found': INTEGER,
                'skipped': INTEGER, 'digitized': INTEGER, 'failed': INTEGER, 'panels': INTEGER,
                'series': INTEGER, 'points': INTEGER, 'output': STRING, 'errors': ARRAY})

ENRICH = counts('succeeded', 'partial', 'not_found', 'unresolved', 'failed', 'authors',
                'subjects', 'references', 'batches')

CORPUS_STATS = obj({'stats': OBJECT, 'enrichment': OBJECT})
CORPUS_SEARCHES = array(obj({'search_id': INTEGER, 'query': STRING, 'status': STRING}))
CORPUS_LIST = obj({'papers': array(obj({'paper_id': STRING})), 'total': INTEGER,
                   'limit': INTEGER, 'offset': INTEGER})
CORPUS_SHOW = obj({'paper': obj({'paper_id': STRING}), 'assets': ARRAY, 'searches': ARRAY,
                   'authors': ARRAY, 'subjects': ARRAY, 'references': ARRAY})
CORPUS_EXPORT = obj({'paper_id': STRING, 'role': STRING, 'path': STRING, 'bytes': INTEGER},
                    {'asset_id': STRING, 'original_filename': OPTIONAL_STRING})

STATUS = obj({'papers': INTEGER, 'papers_with_structured_documents': INTEGER,
              'papers_with_supplements': INTEGER, 'supplements': INTEGER, 'pipeline': OBJECT,
              'text_materials': INTEGER, 'image_materials': INTEGER})

FILTER_OVERVIEW = obj({'filters': array(OBJECT), 'expression': OPTIONAL_STRING,
                       'counts': counts('included', 'excluded', 'unavailable'),
                       'unavailable_reasons': OBJECT, 'stale_topic_filters': array(STRING)},
                      {'dry_run': BOOLEAN})

MODEL_STATUS = obj({'text': OBJECT, 'vision': OBJECT})
CONFIG_MODEL = obj({'profile': STRING, 'provider': STRING, 'model': STRING,
                    'base_url': OPTIONAL_STRING, 'capabilities': array(STRING),
                    'temperature': NUMBER, 'top_p': NUMBER, 'input_token_limit': INTEGER,
                    'api_key_set': BOOLEAN})
CONFIG_SETTING = obj({'setting': STRING, 'secret': BOOLEAN,
                      'source': {'type': 'string', 'enum': ['env', 'stdin', 'prompt']}},
                     {'value': {'type': ['string', 'number', 'null']}})
RECIPE_LIST = array(STRING)
RECIPE = obj({'record definition': OBJECT, 'search fields': OBJECT})

IMPORT_AUTHOR = obj({'found': INTEGER, 'added': INTEGER, 'updated': INTEGER,
                     'enriched': INTEGER, 'review_csv': STRING})
IMPORT_PDFS = obj({'files': INTEGER, 'added': INTEGER, 'updated': INTEGER, 'doi_found': INTEGER,
                   'enriched': INTEGER, 'metadata_errors': INTEGER, 'paper_ids': array(STRING)})

TOPICS_TRAIN = obj({'model_dir': STRING, 'report': OBJECT})
TOPICS_COMPARE = obj({'models_trained': INTEGER, 'output_dir': STRING, 'comparison_csv': STRING,
                      'models': array(OBJECT)})
TOPICS_SHOW = obj({'model_dir': STRING, 'topics': array(obj(
    {'topic_id': INTEGER, 'top_terms': array(STRING), 'representative_papers': array(OBJECT)},
    {'topic_name': OPTIONAL_STRING}))})
TOPICS_NAME = obj({'model_dir': STRING, 'topic_id': INTEGER, 'topic_name': STRING})
TOPICS_PREDICT = obj({'papers_total': INTEGER, 'papers_predicted': INTEGER,
                      'papers_without_vocabulary_terms': INTEGER, 'output_path': STRING})
TOPICS_TRENDS = obj({'output_dir': STRING, 'trends_csv': STRING, 'report_json': STRING,
                     'windows': INTEGER, 'papers_missing_or_invalid_date': INTEGER,
                     'plot_path': OPTIONAL_STRING})
TOPICS_STORE = obj({'model_id': STRING, 'name': STRING, 'papers_predicted': INTEGER,
                    'papers_without_vocabulary_terms': INTEGER,
                    'prediction_corpus_fingerprint': STRING})
TOPICS_MODELS = obj({'models': array(obj({'name': STRING, 'model_id': STRING}))})

ENTITIES = obj({'entities': INTEGER, 'documents': INTEGER, 'output': STRING},
               {'papers': INTEGER})

VALIDATE_TEMPLATE = obj({'recipe': STRING, 'outfile': STRING, 'columns': array(STRING)})
_METRIC = obj({'tp': INTEGER, 'fp': INTEGER, 'fn': INTEGER, 'pending': INTEGER,
               'complete': BOOLEAN, 'precision': OPTIONAL_NUMBER, 'recall': OPTIONAL_NUMBER,
               'f1': OPTIONAL_NUMBER})
VALIDATE_SCORE = obj({'fingerprint': STRING, 'sources': OBJECT, 'recipe': STRING, 'run': OBJECT,
                      'papers_evaluated': INTEGER, 'pending_pairs': INTEGER,
                      'missing_metadata': array(STRING),
                      'fields': {'type': 'object', 'additionalProperties': _METRIC},
                      'complete': BOOLEAN})

_STAGE_COUNTS = {f'{stage}_{outcome}': INTEGER
                 for stage in ('abstract', 'text', 'image')
                 for outcome in ('attempted', 'skipped', 'succeeded', 'failed')}
SCRAPE_RUN = obj({
    'mode': STRING, 'output_path': STRING, 'papers_selected': INTEGER, 'papers': INTEGER,
    **_STAGE_COUNTS, 'missing_source': INTEGER,
    'failed_papers': array(obj({'paper_id': STRING, 'stage': STRING, 'error': STRING})),
    'materials': INTEGER, 'layout_figures': INTEGER, 'chunked_inputs': INTEGER,
    'chunk_requests': INTEGER, 'abstract_chunked': INTEGER, 'text_chunked': INTEGER,
    'supplement_text_attempted': INTEGER, 'supplement_image_attempted': INTEGER,
    'supplement_skipped': INTEGER, 'supplement_failed': INTEGER,
    'supplement_unsupported': INTEGER, 'supplement_materials': INTEGER,
    'elapsed_seconds': NUMBER,
})
SCRAPE_PLAN = obj({
    'dry_run': {'type': 'boolean', 'enum': [True]}, 'mode': STRING, 'recipe': STRING,
    'model': obj({'provider': OPTIONAL_STRING, 'name': OPTIONAL_STRING,
                  'input_token_limit': INTEGER, 'usable_input_tokens': INTEGER,
                  'request_text_budget': INTEGER}),
    'tokenizer': STRING, 'tokenizer_fallbacks': array(STRING), 'prompt_tokens': INTEGER,
    'papers_selected': INTEGER, 'papers_filtered_out': INTEGER,
    'totals': counts('papers_to_scrape', 'skipped_done', 'skipped_missing_source', 'requests',
                     'input_tokens', 'image_inputs', 'supplement_files_to_scrape',
                     'supplement_files_skipped'),
    'overflows': array(obj({'paper_id': STRING, 'item': STRING, 'requests': INTEGER,
                            'text_tokens': INTEGER, 'scraped': BOOLEAN})),
    'papers': array(obj({'paper_id': STRING, 'stages': OBJECT, 'supplements': ARRAY},
                        {'title': STRING})),
})
SCRAPE = {'anyOf': [SCRAPE_RUN, SCRAPE_PLAN]}

SCRAPE_PROGRESS = obj({'mode': STRING, 'scope': INTEGER, 'out_of_scope': INTEGER,
                       'filtered': BOOLEAN, 'stages': OBJECT, 'supplements': OBJECT,
                       'remaining': counts('papers', 'supplement_files'),
                       'failed_papers': OBJECT, 'no_asset_papers': OBJECT})

STORE = obj({'status': {'type': 'string',
                        'enum': ['stored', 'nothing_to_store', 'declined', 'dry_run']},
             'dry_run': BOOLEAN,
             'input_path': STRING, 'output_path': STRING, 'rows_read': INTEGER,
             'rows_written': INTEGER, 'rows_total': INTEGER,
             'converted_columns': array(STRING), 'unconverted_columns': array(STRING),
             'skipped_columns': array(STRING), 'papers_marked_stored': INTEGER})

RESET = obj({'papers_reset': INTEGER, 'supplement_stages_reset': INTEGER,
             'columns': array(STRING)}, {'dry_run': BOOLEAN})
