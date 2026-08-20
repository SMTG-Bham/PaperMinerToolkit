"""Tests for persistent post-download corpus filtering."""

import json

import pytest
from click.testing import CliRunner

import paperscraper.cli as cli
import paperscraper.corpus as corpus
import paperscraper.filtering as filtering
import paperscraper.scrape as scrape


def _write_rules(path, name, pattern, fields=None, **overrides):
    """Write a configurable regular-expression filter fixture."""
    definition = {
        'name': name,
        'fields': fields or ['title'],
        'case_sensitive': False,
        'include_mode': 'any',
        'include': [{'name': f'{name}-include', 'pattern': pattern}],
        'exclude': [],
    }
    definition.update(overrides)
    path.write_text(json.dumps(definition))
    return path


def _add_text(conn, paper, content, role='text'):
    """Add a text asset to a test corpus."""
    corpus.add_asset(
        conn, paper, content, role=role, kind='text', mime_type='text/plain'
    )


def test_regex_filter_classifies_matches_vetoes_missing_content_and_evidence(tmp_path):
    """Positive matches win over missing fields while excludes remain a veto."""
    db_path = tmp_path / 'papers.db'
    papers = [
        {'paper_id': 'paper:included', 'title': 'A TARGET material'},
        {'paper_id': 'paper:unavailable', 'title': 'Unrelated title'},
        {'paper_id': 'paper:excluded', 'title': 'Unrelated title'},
        {'paper_id': 'paper:vetoed', 'title': 'Target review'},
    ]
    with corpus.connect(db_path) as conn:
        for paper in papers:
            corpus.upsert_paper(conn, paper)
        _add_text(conn, papers[2], 'No relevant terms.', role='abstract')
        _add_text(conn, papers[2], 'No relevant terms.')
        _add_text(conn, papers[3], 'Ordinary abstract.', role='abstract')
        _add_text(conn, papers[3], 'This is a review of target materials.')

    rules_path = tmp_path / 'target.json'
    _write_rules(
        rules_path,
        'target',
        r'\btarget\b',
        fields=['title', 'abstract', 'full_text'],
        exclude=[{'name': 'review', 'pattern': r'\breview\b'}],
    )
    overview = filtering.apply_regex_filter(db_path, rules_path)

    assert overview['expression'] == 'target'
    assert overview['counts'] == {'excluded': 2, 'included': 1, 'unavailable': 1}
    with corpus.connect(db_path) as conn:
        statuses = filtering.current_filter_statuses(conn)
        evidence = {
            row['paper_id']: json.loads(row['evidence_json'])
            for row in conn.execute(
                'SELECT paper_id, evidence_json FROM paper_filter_results'
            )
        }
    assert statuses == {
        'paper:included': 'included',
        'paper:unavailable': 'unavailable',
        'paper:excluded': 'excluded',
        'paper:vetoed': 'excluded',
    }
    assert evidence['paper:included']['matched_include_rules'] == ['target-include']
    assert evidence['paper:vetoed']['matched_exclude_rules'] == ['review']


def test_full_text_uses_pdf_fallback_and_ignores_references(tmp_path, monkeypatch):
    """PDF-only papers are searchable, but matches in their references are ignored."""
    db_path = tmp_path / 'papers.db'
    paper = {'paper_id': 'paper:pdf', 'title': 'PDF paper'}
    with corpus.connect(db_path) as conn:
        corpus.add_asset(
            conn, paper, b'not-a-real-pdf', role='pdf', kind='pdf', mime_type='application/pdf'
        )
    monkeypatch.setattr(
        filtering,
        'read_pdf_bytes',
        lambda _: 'The target is a result.\nReferences\nA review of target materials.',
    )
    rules_path = tmp_path / 'pdf.json'
    _write_rules(
        rules_path,
        'pdf-target',
        r'\btarget\b',
        fields=['full_text'],
        exclude=[{'name': 'review', 'pattern': r'\breview\b'}],
    )

    overview = filtering.apply_regex_filter(db_path, rules_path)

    assert overview['counts']['included'] == 1
    with corpus.connect(db_path) as conn:
        evidence = json.loads(conn.execute(
            'SELECT evidence_json FROM paper_filter_results'
        ).fetchone()[0])
    assert evidence['sources'] == {'full_text': 'corpus:pdf'}
    assert evidence['matched_exclude_rules'] == []


def test_filters_combine_left_to_right_and_named_reset_recomputes_state(tmp_path):
    """Successive operators form an explicit left-associative expression."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        for paper_id, title in [
            ('paper:1', 'alpha'),
            ('paper:2', 'beta gamma'),
            ('paper:3', 'beta'),
        ]:
            corpus.upsert_paper(conn, {'paper_id': paper_id, 'title': title})
    alpha = _write_rules(tmp_path / 'alpha.json', 'alpha', r'\balpha\b')
    beta = _write_rules(tmp_path / 'beta.json', 'beta', r'\bbeta\b')
    gamma = _write_rules(tmp_path / 'gamma.json', 'gamma', r'\bgamma\b')

    filtering.apply_regex_filter(db_path, alpha)
    filtering.apply_regex_filter(db_path, beta, join_operator='or')
    overview = filtering.apply_regex_filter(db_path, gamma, join_operator='and')

    assert overview['expression'] == '((alpha OR beta) AND gamma)'
    assert overview['counts'] == {'excluded': 2, 'included': 1, 'unavailable': 0}
    overview = filtering.reset_filters(db_path, name='beta')
    assert overview['expression'] == '(alpha AND gamma)'
    assert overview['counts']['included'] == 0
    overview = filtering.reset_filters(db_path, name='alpha')
    assert overview['expression'] == 'gamma'
    assert overview['counts']['included'] == 1
    overview = filtering.reset_filters(db_path, all_filters=True)
    assert overview['expression'] == ''
    assert overview['counts'] == {'excluded': 0, 'included': 3, 'unavailable': 0}


def test_duplicate_requires_replace_and_invalid_pattern_is_atomic(tmp_path):
    """Definitions are compiled before existing persisted state can be changed."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'paper:1', 'title': 'alpha'})
    rules_path = _write_rules(tmp_path / 'alpha.json', 'alpha', r'\balpha\b')
    filtering.apply_regex_filter(db_path, rules_path)

    with pytest.raises(ValueError, match='already active'):
        filtering.apply_regex_filter(db_path, rules_path)
    _write_rules(rules_path, 'alpha', '[')
    with pytest.raises(ValueError, match='Invalid regex'):
        filtering.apply_regex_filter(db_path, rules_path, replace=True)
    with corpus.connect(db_path) as conn:
        overview = filtering.filter_overview(conn)
    assert overview['expression'] == 'alpha'
    assert overview['counts']['included'] == 1


def test_any_all_case_and_timeout_validation():
    """Definition normalization exposes stable defaults and validates unsafe input."""
    definition = filtering.normalize_regex_definition({
        'name': 'demo',
        'include_mode': 'all',
        'include': [
            {'name': 'one', 'pattern': 'one'},
            {'name': 'two', 'pattern': 'two'},
        ],
    })
    assert definition['fields'] == ['title', 'abstract', 'full_text']
    assert definition['timeout_ms'] == filtering.DEFAULT_TIMEOUT_MS
    assert definition['case_sensitive'] is False
    with pytest.raises(ValueError, match='positive integer'):
        filtering.normalize_regex_definition({
            'name': 'bad', 'include': [{'name': 'x', 'pattern': 'x'}], 'timeout_ms': 0,
        })


def test_all_mode_case_sensitivity_and_timeout_classification():
    """All-mode requires every rule, while timed-out undecided matches are unavailable."""
    raw = {
        'name': 'all-terms',
        'fields': ['title'],
        'case_sensitive': True,
        'include_mode': 'all',
        'include': [
            {'name': 'one', 'pattern': 'one'},
            {'name': 'two', 'pattern': 'two'},
        ],
    }
    definition = filtering.normalize_regex_definition(raw)
    status, _, _ = filtering.evaluate_regex_paper(
        None,
        {'paper_id': 'paper:1', 'title': 'ONE two'},
        definition,
        filtering.compile_regex_definition(definition),
    )
    assert status == 'excluded'

    raw['case_sensitive'] = False
    definition = filtering.normalize_regex_definition(raw)
    status, _, _ = filtering.evaluate_regex_paper(
        None,
        {'paper_id': 'paper:1', 'title': 'ONE two'},
        definition,
        filtering.compile_regex_definition(definition),
    )
    assert status == 'included'

    class TimeoutExpression:
        """Represent an expression that always times out."""

        def finditer(self, text, timeout):
            """Raise a timeout instead of returning matches."""
            raise TimeoutError

    matches, timed_out = filtering._match_rule(
        TimeoutExpression(), {'title': 'text'}, timeout_seconds=0.001
    )
    assert matches == []
    assert timed_out == ['title']


def test_filter_cli_applies_reports_and_resets(tmp_path):
    """The three CLI entry points expose the persisted filter lifecycle."""
    db_path = tmp_path / 'papers.db'
    with corpus.connect(db_path) as conn:
        corpus.upsert_paper(conn, {'paper_id': 'paper:1', 'title': 'alpha'})
    rules_path = _write_rules(tmp_path / 'alpha.json', 'alpha', 'alpha')
    runner = CliRunner()

    applied = runner.invoke(cli.filter_regex, [str(db_path), str(rules_path)])
    shown = runner.invoke(cli.filter_status, [str(db_path)])
    reset = runner.invoke(cli.filter_reset, [str(db_path), '--all'])

    assert applied.exit_code == 0
    assert 'Active expression: alpha' in applied.output
    assert 'Final result: included=1, excluded=0, unavailable=0' in shown.output
    assert reset.exit_code == 0
    assert 'Active expression: none' in reset.output


def test_scrape_enforces_filter_gate_and_supports_explicit_bypass(tmp_path, monkeypatch, capsys):
    """Scraping intersects its normal selection with persisted included papers."""
    db_path = tmp_path / 'papers.db'
    papers = [
        {'paper_id': 'paper:included', 'title': 'alpha study'},
        {'paper_id': 'paper:excluded', 'title': 'beta study'},
    ]
    with corpus.connect(db_path) as conn:
        for paper in papers:
            _add_text(conn, paper, f'text for {paper["paper_id"]}')
    rules_path = _write_rules(tmp_path / 'alpha.json', 'alpha', r'\balpha\b')
    filtering.apply_regex_filter(db_path, rules_path)

    class FakeModelConfig:
        """Provide a minimal model configuration for scrape tests."""

        @classmethod
        def from_profile(cls, *args, **kwargs):
            """Return a new fake configuration for any profile."""
            return cls()

    class FakeTqdm:
        """Provide a no-op progress-bar context manager."""

        def __init__(self, *args, **kwargs):
            """Accept and ignore progress-bar options."""
            pass

        def __enter__(self):
            """Return the fake progress bar."""
            return self

        def __exit__(self, *_):
            """Propagate exceptions raised inside the context."""
            return False

        def update(self, amount):
            """Ignore a progress update."""
            pass

    calls = []
    monkeypatch.setattr(scrape, 'load_recipe', lambda _: {'search fields': {}})
    monkeypatch.setattr(scrape, 'ModelConfig', FakeModelConfig)
    monkeypatch.setattr(scrape, 'tqdm', FakeTqdm)
    monkeypatch.setattr(scrape, 'read_document_text', lambda path: path)
    monkeypatch.setattr(scrape, 'build_scrape_prompt', lambda *args, **kwargs: 'prompt')
    monkeypatch.setattr(scrape, 'maybe_compress_text', lambda text, *args: text)
    monkeypatch.setattr(scrape, '_text_chunks', lambda text, *args, **kwargs: [text])
    monkeypatch.setattr(
        scrape, 'scrape_text',
        lambda text, recipe, model_config=None: calls.append(text) or [],
    )

    scrape.scrape_papers(db_path, output_path=str(tmp_path / 'filtered.csv'))
    with corpus.connect(db_path) as conn:
        statuses = {
            row['paper_id']: row['text_scrape_status'] for row in corpus.paper_rows(conn)
        }
    assert len(calls) == 1
    assert statuses == {'paper:included': 'succeeded', 'paper:excluded': 'pending'}
    assert 'Applying corpus filters: alpha' in capsys.readouterr().out

    calls.clear()
    scrape.scrape_papers(
        db_path,
        output_path=str(tmp_path / 'unfiltered.csv'),
        ignore_filters=True,
        force=True,
    )
    assert len(calls) == 2
    assert 'Ignoring active corpus filters for this scrape: alpha' in capsys.readouterr().out


@pytest.mark.parametrize(
    ('left', 'right', 'operator', 'expected'),
    [
        ('included', 'unavailable', 'and', 'unavailable'),
        ('excluded', 'unavailable', 'and', 'excluded'),
        ('included', 'unavailable', 'or', 'included'),
        ('excluded', 'unavailable', 'or', 'unavailable'),
    ],
)
def test_three_valued_status_combination(left, right, operator, expected):
    """Combine representative three-valued filter decisions."""
    assert filtering.combine_status(left, right, operator) == expected
