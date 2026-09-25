"""Local, read-only CSV comparison and persistent human review."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import re
import secrets
import tempfile
import webbrowser
from difflib import SequenceMatcher
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from paperminertoolkit.extraction.recipes import RECIPES_PATH, canonical_match, load_recipe
from paperminertoolkit.workflows.validation_scoring import GAP_FIELDS, score_review

PAGE = Path(__file__).resolve().parents[1] / 'resources' / 'validation.html'
SCRIPT = Path(__file__).resolve().parents[1] / 'resources' / 'validation.js'
BRAND_LOGO = Path(__file__).resolve().parents[2] / 'assets' / 'Paper_Miner_Toolkit_acronym_light.svg'
REVIEW_DIR = Path.home() / '.config' / 'paperminertoolkit' / 'validation-reviews'
MAX_REQUEST_BYTES = 32 * 1024 * 1024
META = {
    'doi': ('doi',),
    'paper_id': ('identifier', 'paper id', 'paper_id'),
    'title': ('title', 'paper title'),
}


def _clean(value: object) -> str:
    """Return a usable CSV cell without treating missing markers as names."""
    result = str(value or '').strip()
    return '' if result.casefold() in {'none', 'nan', 'null'} else result


def _doi(value: object) -> str:
    """Normalize common DOI URL and prefix forms."""
    result = _clean(value).casefold()
    result = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', result)
    result = re.sub(r'^doi\s*:\s*', '', result)
    return result.strip().rstrip('/').strip()


def _name(value: object) -> str:
    """Normalize a title or recipe identity for suggestions."""
    return ' '.join(re.findall(r'[\w]+', _clean(value).casefold()))


def _structured_cell(value: str) -> str:
    """Convert Python-style list/dict cells to JSON for the browser review."""
    if not value.lstrip().startswith(('[', '{')):
        return value
    try:
        json.loads(value)
        return value
    except (ValueError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, dict)):
            return json.dumps(parsed, ensure_ascii=False)
    except (ValueError, SyntaxError, TypeError, OverflowError, RecursionError):
        pass
    return value


def _csv_rows(content: str, label: str, recipe: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse a CSV and map recipe fields while preserving source cells."""
    try:
        reader = csv.DictReader(io.StringIO(content.lstrip('\ufeff'), newline=''))
        headers = reader.fieldnames or []
        if not headers or len(headers) != len(set(headers)):
            raise ValueError(f'{label} CSV needs a header with unique column names.')
        fields = list(recipe['search fields'])
        columns: dict[str, str] = {}
        for header in headers:
            if not header or header.startswith('Unnamed:'):
                continue
            match = canonical_match(header.split(' [', 1)[0], fields, recipe)
            if match and match not in columns:
                columns[match] = header
        if not any(field in columns for field in fields):
            raise ValueError(f'{label} CSV has no fields from the selected recipe: {", ".join(fields)}.')
        missing_identity = [field for field in recipe['record definition']['identity fields']
                            if field not in columns]
        if missing_identity:
            raise ValueError(f'{label} CSV is missing recipe identity field(s): {", ".join(missing_identity)}.')
        names = {header.casefold().strip(): header for header in headers if header}
        meta_columns = {key: next((names[n] for n in aliases if n in names), None)
                        for key, aliases in META.items()}
        if not any(meta_columns.values()):
            raise ValueError(f'{label} CSV needs a DOI, paper ID, or title column.')
        rows = []
        for index, raw in enumerate(reader, 2):
            if None in raw:
                raise ValueError(f'{label} CSV row {index} has more cells than its header.')
            if not any(_clean(value) for value in raw.values()):
                continue
            metadata = {key: _clean(raw.get(column)) if column else ''
                        for key, column in meta_columns.items()}
            raw_values = {field: str(raw.get(columns[field]) or '') if field in columns else ''
                          for field in fields}
            values = {field: _structured_cell(value) for field, value in raw_values.items()}
            rows.append({'id': f'{label[0].lower()}:{index}', 'line': index,
                         'meta': metadata, 'values': values, 'raw_values': raw_values})
        return rows
    except csv.Error as error:
        raise ValueError(f'{label} CSV cannot be read: {error}') from error


def _empty_gold(row: dict[str, Any], fields: list[str],
                identity_fields: list[str]) -> bool:
    """Recognize the validation CSV's zero-material paper placeholder."""
    if any(_clean(row['values'].get(field)) for field in identity_fields):
        return False
    return all(not _clean(row['values'][field]) or row['values'][field].strip() == '[]'
               for field in fields if field not in identity_fields)


def _paper_key(meta: dict[str, str], row_id: str) -> str:
    """Choose the strongest available within-file paper grouping key."""
    return _doi(meta['doi']) or _name(meta['paper_id']) or _name(meta['title']) or row_id


def _papers(rows: list[dict[str, Any]], prefix: str,
            fields: list[str], identity_fields: list[str]) -> list[dict[str, Any]]:
    """Group CSV rows by paper and remove zero-material placeholders."""
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _paper_key(row['meta'], row['id'])
        if key not in grouped:
            grouped[key] = {'id': f'{prefix}:{len(grouped)}', 'meta': row['meta'],
                            'rows': [], 'empty': False, 'metadata_warning': ''}
        paper = grouped[key]
        for field, normalize in (('doi', _doi), ('paper_id', _name), ('title', _name)):
            first = normalize(paper['meta'][field])
            later = normalize(row['meta'][field])
            if first and later and first != later:
                paper['metadata_warning'] = (
                    'Rows grouped as one paper have conflicting '
                    f'{field.replace("_", " ")} values; check this paper manually.'
                )
        if prefix == 'v' and _empty_gold(row, fields, identity_fields):
            paper['empty'] = True
        else:
            paper['rows'].append(row)
    return list(grouped.values())


def _paper_suggestions(gold: list[dict[str, Any]], scraped: list[dict[str, Any]]) -> None:
    """Suggest unique DOI, then ID, then title matches; never hide conflicts."""
    used: set[str] = set()
    for paper in gold:
        meta = paper['meta']
        paper['suggested'] = None
        paper['match_reason'] = ''
        paper['warning'] = paper['metadata_warning']
        stages = [('DOI', 'doi', _doi), ('paper ID', 'paper_id', _name),
                  ('title', 'title', _name)]
        for label, key, normalize in stages:
            value = normalize(meta[key])
            if not value:
                continue
            matches = [candidate for candidate in scraped
                       if normalize(candidate['meta'][key]) == value]
            if len(matches) > 1:
                paper['warning'] = f'Ambiguous {label}: choose a paper manually.'
                break
            if not matches:
                continue
            candidate = matches[0]
            if candidate['id'] in used:
                paper['warning'] = f'{label} matches a paper already suggested elsewhere; choose manually.'
                break
            if meta['doi'] and candidate['meta']['doi'] and _doi(meta['doi']) != _doi(candidate['meta']['doi']):
                paper['warning'] = 'Paper metadata conflicts: DOIs differ.'
                break
            paper['suggested'] = candidate['id']
            paper['match_reason'] = label
            if key == 'doi' and meta['paper_id'] and candidate['meta']['paper_id'] and _name(meta['paper_id']) != _name(candidate['meta']['paper_id']):
                paper['warning'] = 'Paper IDs differ; DOI matched.'
            used.add(candidate['id'])
            break


def _identity(row: dict[str, Any], identity_fields: list[str]) -> str:
    """Build a comparison string from the recipe's identity fields."""
    return ' '.join(_name(row['values'].get(field, '')) for field in identity_fields).strip()


def _material_suggestions(gold: list[dict[str, Any]], scraped: list[dict[str, Any]],
                          identity_fields: list[str]) -> dict[str, str]:
    """Make one-to-one provisional identity suggestions within a paper."""
    choices = []
    for left in gold:
        a = _identity(left, identity_fields)
        if not a:
            continue
        for right in scraped:
            b = _identity(right, identity_fields)
            if not b:
                continue
            score = SequenceMatcher(None, a, b).ratio()
            if score >= 0.72:
                choices.append((score, left['id'], right['id']))
    choices.sort(reverse=True)
    result: dict[str, str] = {}
    used = set()
    for _, left, right in choices:
        if left not in result and right not in used:
            result[left] = right
            used.add(right)
    return result


def _load_recipe_text(content: str) -> dict[str, Any]:
    """Load a custom uploaded recipe using the package's schema validation."""
    from paperminertoolkit.extraction.recipes import _validate_recipe

    try:
        obj = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f'Recipe JSON is invalid: {error}') from error
    if isinstance(obj, dict) and 'record definition' not in obj and len(obj) == 1:
        obj = next(iter(obj.values()))
    return _validate_recipe(obj, 'uploaded recipe')


class ReviewApp:
    """One localhost review process with an active source selection."""

    def __init__(self, output: Path | None = None) -> None:
        """Prepare a review app with an optional decisions destination."""
        self.output = output
        self.session: dict[str, Any] | None = None
        self.review_path: Path | None = None
        self.decisions: dict[str, Any] = {}
        self.input_snapshot: dict[str, Any] | None = None
        self._resume_path: Path | None = None
        self._review_paths: dict[str, Path] = {}

    @staticmethod
    def recipes() -> list[str]:
        """List bundled recipes for the browser selector."""
        with RECIPES_PATH.open(encoding='utf-8') as handle:
            return list(json.load(handle))

    def list_reviews(self) -> list[dict[str, Any]]:
        """List locally saved reviews and whether their inputs can be restored."""
        paths = ({path for path in REVIEW_DIR.glob('*.json')
                  if not path.name.endswith('.inputs.json')}
                 if REVIEW_DIR.is_dir() else set())
        if self.output and self.output.is_file():
            paths.add(self.output)
        reviews = []
        self._review_paths = {}
        for path in paths:
            try:
                saved = json.loads(path.read_text(encoding='utf-8'))
                fingerprint = saved.get('fingerprint', '')
                if not re.fullmatch(r'[0-9a-f]{64}', fingerprint):
                    continue
                cache_name = saved.get('input_cache', '')
                cache_path = REVIEW_DIR / cache_name if cache_name and Path(cache_name).name == cache_name else None
                cache = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path and cache_path.is_file() else {}
                inputs = cache.get('inputs', {})
                resumable = (isinstance(inputs, dict)
                    and bool(inputs.get('recipe_name') or inputs.get('recipe_text'))
                    and cache.get('fingerprint') == fingerprint and all(
                    isinstance(cache.get(key), str) for key in ('validation_csv', 'scraped_csv')
                ))
                review_id = hashlib.sha256(str(path.resolve()).encode('utf-8')).hexdigest()[:24]
                self._review_paths[review_id] = path
                inputs = cache.get('inputs', {}) if isinstance(cache, dict) else {}
                reviews.append({
                    'id': review_id,
                    'fingerprint': fingerprint,
                    'recipe': saved.get('recipe', 'unknown recipe'),
                    'validation_name': inputs.get('validation_name', 'Validation CSV'),
                    'scraped_name': inputs.get('scraped_name', 'Scraped CSV'),
                    'saved_at': int(path.stat().st_mtime),
                    'resumable': resumable,
                })
            except (OSError, ValueError, TypeError, AttributeError):
                continue
        reviews.sort(key=lambda review: review['saved_at'], reverse=True)
        return reviews

    def resume_review(self, review_id: str) -> dict[str, Any]:
        """Restore a review and its local input snapshot by opaque review id."""
        self.list_reviews()
        path = self._review_paths.get(review_id)
        if path is None:
            raise ValueError('That saved review is no longer available.')
        saved = json.loads(path.read_text(encoding='utf-8'))
        cache_name = saved.get('input_cache', '')
        if not cache_name or Path(cache_name).name != cache_name:
            raise ValueError('This older review has no saved CSV copies. Select its files once to enable quick resume.')
        cache_path = REVIEW_DIR / cache_name
        snapshot = json.loads(cache_path.read_text(encoding='utf-8'))
        if snapshot.get('fingerprint') != saved.get('fingerprint'):
            raise ValueError('Saved CSV copies do not match this review.')
        inputs = snapshot.get('inputs', {})
        if not isinstance(inputs, dict) or not (inputs.get('recipe_name') or inputs.get('recipe_text')):
            raise ValueError('Saved review is missing its recipe; select its files again.')
        self._resume_path = path
        try:
            return self.load(
                snapshot['validation_csv'], snapshot['scraped_csv'],
                inputs.get('recipe_name', ''), inputs.get('recipe_text', ''),
                inputs.get('validation_name', ''), inputs.get('scraped_name', ''),
                inputs.get('validation_sha256', ''), inputs.get('scraped_sha256', ''),
            )
        except Exception:
            self._resume_path = None
            raise

    def load(self, validation_csv: str, scraped_csv: str, recipe_name: str,
             recipe_text: str = '', validation_name: str = '',
             scraped_name: str = '', validation_sha256: str = '',
             scraped_sha256: str = '') -> dict[str, Any]:
        """Read the three inputs and resume matching saved decisions."""
        validation_csv = validation_csv.lstrip('\ufeff')
        scraped_csv = scraped_csv.lstrip('\ufeff')
        if any(value and not re.fullmatch(r'[0-9a-f]{64}', value)
               for value in (validation_sha256, scraped_sha256)):
            raise ValueError('Source SHA-256 values must be 64 lowercase hexadecimal characters.')
        recipe = _load_recipe_text(recipe_text) if recipe_text else load_recipe(recipe_name)
        fields = list(recipe['search fields'])
        identity_fields = recipe['record definition']['identity fields']
        if not identity_fields:
            identity_fields = fields[:1]
        gold = _papers(_csv_rows(validation_csv, 'Validation', recipe), 'v', fields, identity_fields)
        scraped = _papers(_csv_rows(scraped_csv, 'Scraped', recipe), 's', fields, identity_fields)
        structure_fields = {}
        for field in fields:
            example = recipe['search fields'][field].get('example')
            kind = ('list' if isinstance(example, list) else
                    'dict' if isinstance(example, dict) else None)
            if kind is None:
                for paper in gold + scraped:
                    for row in paper['rows']:
                        try:
                            value = json.loads(row['values'][field])
                        except (ValueError, TypeError):
                            continue
                        if isinstance(value, (list, dict)):
                            kind = 'list' if isinstance(value, list) else 'dict'
                            break
                    if kind:
                        break
            if kind is None and field in GAP_FIELDS:
                kind = 'list'
            if kind:
                structure_fields[field] = kind
        list_fields = [field for field, kind in structure_fields.items() if kind == 'list']
        _paper_suggestions(gold, scraped)
        by_id = {paper['id']: paper for paper in scraped}
        for paper in gold:
            match = by_id.get(paper['suggested'])
            paper['material_suggestions'] = (
                _material_suggestions(paper['rows'], match['rows'], identity_fields)
                if match else {}
            )
        digest = hashlib.sha256()
        for content in (validation_csv, scraped_csv, json.dumps(recipe, sort_keys=True)):
            data = content.encode('utf-8')
            digest.update(len(data).to_bytes(8, 'big'))
            digest.update(data)
        fingerprint = digest.hexdigest()
        if self._resume_path is not None:
            self.review_path = self._resume_path
            self._resume_path = None
        else:
            self.review_path = self.output or REVIEW_DIR / f'{fingerprint}.json'
        if self.output and self._resume_path is None and self.output.is_file() and self.review_path == self.output:
            existing = json.loads(self.output.read_text(encoding='utf-8'))
            if existing.get('fingerprint') != fingerprint:
                self.review_path = self.output.with_name(
                    f'{self.output.stem}-{fingerprint}{self.output.suffix or ".json"}'
                )
        self.decisions = {}
        if self.review_path.is_file():
            saved = json.loads(self.review_path.read_text(encoding='utf-8'))
            if saved.get('fingerprint') != fingerprint:
                raise ValueError(f'Review output {self.review_path} belongs to different inputs.')
            self.decisions = saved.get('decisions', {})
        sources = {'validation': {'name': validation_name or 'selected validation CSV',
                                  'sha256': validation_sha256 or hashlib.sha256(validation_csv.encode('utf-8')).hexdigest()},
                   'scraped': {'name': scraped_name or 'selected scraped CSV',
                               'sha256': scraped_sha256 or hashlib.sha256(scraped_csv.encode('utf-8')).hexdigest()},
                   'recipe': {'name': recipe_name or 'uploaded recipe',
                              'sha256': hashlib.sha256(json.dumps(recipe, sort_keys=True).encode('utf-8')).hexdigest()}}
        self.session = {'fingerprint': fingerprint, 'fields': fields,
                        'identity_fields': identity_fields, 'recipe': recipe_name or 'custom',
                        'list_fields': list_fields, 'structure_fields': structure_fields,
                        'sources': sources,
                        'validation': gold, 'scraped': scraped,
                        'decisions': self.decisions, 'review_path': str(self.review_path)}
        self.input_snapshot = {
            'fingerprint': fingerprint,
            'validation_csv': validation_csv,
            'scraped_csv': scraped_csv,
            'inputs': {
                'recipe_name': recipe_name,
                'recipe_text': recipe_text,
                'validation_name': validation_name or 'selected validation CSV',
                'scraped_name': scraped_name or 'selected scraped CSV',
                'validation_sha256': validation_sha256,
                'scraped_sha256': scraped_sha256,
            },
        }
        self._save_input_snapshot()
        self.save(self.decisions)
        return self.session

    def _save_input_snapshot(self) -> None:
        """Cache chosen files locally once so saved reviews can be resumed."""
        if self.input_snapshot is None:
            return
        path = REVIEW_DIR / f"{self.input_snapshot['fingerprint']}.inputs.json"
        if path.is_file():
            try:
                cached = json.loads(path.read_text(encoding='utf-8'))
                if (cached.get('fingerprint') == self.input_snapshot['fingerprint']
                        and isinstance(cached.get('validation_csv'), str)
                        and isinstance(cached.get('scraped_csv'), str)):
                    return
            except (OSError, ValueError, TypeError):
                pass
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix='.inputs-', suffix='.tmp', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(self.input_snapshot, handle, ensure_ascii=False)
                handle.write('\n')
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def save(self, decisions: dict[str, Any]) -> None:
        """Atomically save decisions without writing to the input CSVs."""
        if self.session is None or self.review_path is None:
            raise ValueError('Select the review inputs first.')
        if not isinstance(decisions, dict):
            raise ValueError('Review decisions must be an object.')
        self.review_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_input_snapshot()
        payload = {'fingerprint': self.session['fingerprint'],
                   'recipe': self.session['recipe'],
                   'input_cache': f"{self.session['fingerprint']}.inputs.json",
                   'decisions': decisions}
        fd, temp = tempfile.mkstemp(prefix='.review-', suffix='.tmp', dir=self.review_path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write('\n')
            os.replace(temp, self.review_path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        self.decisions = decisions
        self.session['decisions'] = decisions


def serve(app: ReviewApp, *, open_browser: bool = True) -> None:
    """Serve a single-user review UI on a random loopback port."""
    token = secrets.token_urlsafe(24)
    html = PAGE.read_text(encoding='utf-8').replace('__TOKEN__', token)

    class Handler(BaseHTTPRequestHandler):
        def _authorized(self) -> bool:
            """Require the per-process token for every route."""
            query = parse_qs(urlsplit(self.path).query)
            return query.get('token', [''])[0] == token

        def _reply(self, status: int, body: bytes, content_type: str) -> None:
            """Write an uncached response."""
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: object) -> None:
            """Write a JSON response."""
            self._reply(status, json.dumps(obj, ensure_ascii=False).encode('utf-8'),
                        'application/json; charset=utf-8')

        def do_GET(self) -> None:
            """Serve the reviewer and its read-only session endpoints."""
            if not self._authorized():
                self._json(HTTPStatus.FORBIDDEN, {'error': 'Invalid review session token.'})
                return
            path = urlsplit(self.path).path
            if path == '/':
                self._reply(HTTPStatus.OK, html.encode('utf-8'), 'text/html; charset=utf-8')
            elif path == '/brand.svg':
                self._reply(HTTPStatus.OK, BRAND_LOGO.read_bytes(), 'image/svg+xml; charset=utf-8')
            elif path == '/review.js':
                self._reply(HTTPStatus.OK, SCRIPT.read_bytes(), 'text/javascript; charset=utf-8')
            elif path == '/api/config':
                self._json(HTTPStatus.OK, {'recipes': app.recipes(), 'session': app.session,
                                           'reviews': app.list_reviews()})
            elif path == '/api/reviews':
                self._json(HTTPStatus.OK, app.list_reviews())
            elif path == '/api/export':
                if app.session is None:
                    self._json(HTTPStatus.BAD_REQUEST, {'error': 'Select the review inputs first.'})
                else:
                    self._json(HTTPStatus.OK, {'score': score_review(app.session, app.decisions),
                                               'decisions': app.decisions})
            else:
                self._json(HTTPStatus.NOT_FOUND, {'error': 'Not found.'})

        def do_POST(self) -> None:
            """Load inputs or persist decisions."""
            if not self._authorized():
                self._json(HTTPStatus.FORBIDDEN, {'error': 'Invalid review session token.'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if size < 1 or size > MAX_REQUEST_BYTES:
                    raise ValueError('Request is empty or exceeds the 32 MB limit.')
                data = json.loads(self.rfile.read(size))
                path = urlsplit(self.path).path
                if path == '/api/load':
                    result = app.load(data['validation'], data['scraped'],
                                      data.get('recipe', ''), data.get('recipe_text', ''),
                                      data.get('validation_name', ''), data.get('scraped_name', ''),
                                      data.get('validation_sha256', ''), data.get('scraped_sha256', ''))
                    self._json(HTTPStatus.OK, result)
                elif path == '/api/resume':
                    self._json(HTTPStatus.OK, app.resume_review(data['review_id']))
                elif path == '/api/save':
                    if data.get('fingerprint') != (app.session or {}).get('fingerprint'):
                        raise ValueError('Input files changed; reload the review before saving.')
                    app.save(data['decisions'])
                    self._json(HTTPStatus.OK, {'saved': True})
                elif path == '/api/score':
                    if app.session is None or data.get('fingerprint') != app.session['fingerprint']:
                        raise ValueError('Select the review inputs first.')
                    self._json(HTTPStatus.OK, score_review(app.session, data['decisions']))
                else:
                    self._json(HTTPStatus.NOT_FOUND, {'error': 'Not found.'})
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {'error': str(error)})

        def log_message(self, format: str, *args: object) -> None:
            """Keep the CLI output focused on the local URL."""
            return

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    address = f'http://127.0.0.1:{server.server_port}/?token={token}'
    print(f'Validation review: {address}')
    print('Press Ctrl+C to stop the review server.')
    if open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
