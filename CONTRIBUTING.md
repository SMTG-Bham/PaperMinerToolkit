# Contributing to PaperMinerToolkit

Thanks for your interest in the project. Bug reports, data-source clients,
recipes, documentation, and tests are all welcome.

PaperMinerToolkit is alpha software, so interfaces still move. Open an
[issue](https://github.com/SMTG-Bham/PaperMinerToolkit/issues) before starting
anything large, so the design can be agreed before the work is done.

## Reporting issues

Include the PaperMinerToolkit version
(`python -c "import paperminertoolkit; print(paperminertoolkit.__version__)"`),
the Python version, the command or Python snippet that failed, and the full
traceback. For extraction problems, the recipe name and the model
configuration (`pmt config status`) usually decide the diagnosis.

## Development setup

PaperMinerToolkit requires Python 3.11 or newer. With conda:

```bash
conda env create -f build_tools/environment.yml
conda activate paperminertoolkit
pip install -e ".[test]"
```

Or with any Python 3.11+ interpreter:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

`environment.yml` supplies the interpreter only. Every runtime and test
dependency is declared once in `pyproject.toml`; add new ones there and re-run
`pip install -e ".[test]"`. See [build_tools/README.md](build_tools/README.md),
which also covers installing on ASU Sol's Intel Gaudi nodes.

## Branching and pull requests

`dev` is the integration branch and `main` tracks released state. Branch off
`dev`, open the pull request against `dev`, and keep each one to a single
change with its tests. Write a short summary line for each commit; the
`test:` and `refactor:` prefixes in the history mark focused test and
restructuring work.

## Checks before opening a pull request

These are the same steps CI runs, in order:

```bash
pytest
```

```bash
ruff check paperminertoolkit tests
```

```bash
python -m compileall paperminertoolkit examples
```

```bash
make -C docs html
```

The test job runs on Python 3.11 and 3.14 (the ends of the supported range)
and uploads coverage to Codecov; the docs job builds Sphinx on 3.13.

## Tests

`pyproject.toml` deselects the `network` and `slow` markers by default, so a
bare `pytest` run is offline and fast. Run the excluded tests explicitly when
you touch what they cover:

```bash
pytest -m network
```

```bash
pytest -m slow
```

Network tests call live external services and need the corresponding
credentials configured. To reproduce the CI coverage report locally:

```bash
pytest --cov=paperminertoolkit --cov-report=term-missing
```

Default-suite tests must not touch the network. Every source client takes an
injected HTTP session, so use the `FakeResponse` and session doubles in
[tests/doubles.py](tests/doubles.py) rather than writing new ones. The autouse
fixture in [tests/conftest.py](tests/conftest.py) resets shared provider
limiters and silences pacing sleeps between tests.

Test files are held to the same docstring and annotation rules as the package.

## Code style

- Start each module with `from __future__ import annotations`.
- Every module, class, function, and method needs a docstring in NumPy style.
  `ruff` (`D`, numpy convention) and
  [tests/test_docstrings.py](tests/test_docstrings.py) both enforce this, and
  Google-style headings such as `Args:` or `Returns:` are rejected.
- Annotate every parameter and return value. `ruff` (`ANN`) and the same test
  module enforce it; `Any` is allowed only where a payload genuinely is
  dynamic.
- Single quotes and four-space indentation, matching the surrounding code.
- Comments explain why a decision was made, not what the line does.

## Adding a data source

The registry in
[paperminertoolkit/providers/registry.py](paperminertoolkit/providers/registry.py)
is the single place a source is declared. To add one:

1. Write the client module under `paperminertoolkit/providers/`, taking an
   injected `requests` session so it can be tested offline.
2. Declare a `Source` in the registry with its capabilities, the credential it
   needs, and the corpus column that identifies a paper to it. Place it in the
   order tuple for each capability it declares — a test asserts every order
   covers exactly the sources carrying that capability.
3. Add its dotted path to `PUBLIC_MODULES` in
   [tests/test_package_layout.py](tests/test_package_layout.py).
4. Add a `tests/test_<source>.py` built on the shared doubles.
5. Add `docs/reference/modules/providers/<source>.rst` as an `automodule` stub
   and list it in that directory's `index.md` toctree.

Recipes live in `paperminertoolkit/resources/recipes.json`, which ships as
package data; see [docs/workflow/recipes.md](docs/workflow/recipes.md).

## Documentation

```bash
python -m pip install -e '.[docs]'
make -C docs html
```

Open `docs/_build/html/index.html`. Narrative guides live in `docs/workflow/`,
runnable notebooks in `docs/examples/`, and the API reference is generated from
docstrings under `docs/reference/`. A new public module needs an `.rst` stub
and a toctree entry alongside the existing ones. Keep `docs/requirements.txt`
aligned with the `docs` optional dependency in `pyproject.toml` — Read the Docs
installs the package with `--no-deps` and uses that file.

## Releases

Maintainers bump `__version__` in
[paperminertoolkit/_version.py](paperminertoolkit/_version.py) and publish a
GitHub release. That triggers `.github/workflows/publish.yml`, which builds the
wheel and source distribution, validates them with `twine check`, and uploads
to PyPI through trusted publishing.

## License

By contributing you agree that your contributions are licensed under the
[MIT License](LICENSE) that covers the project.
