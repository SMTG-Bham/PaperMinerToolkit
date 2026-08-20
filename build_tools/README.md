# Installation guide

This guide will help you install and set up the project on your local machine.

Create the environment (this installs the Python interpreter):

```bash
conda env create -f build_tools/environment.yml
```

```bash
conda activate paperscraper
```

Then install the package and its dependencies from the repository root. All
dependencies are declared in `pyproject.toml`, so this single command pulls in
the full runtime and test stack:

```bash
pip install -e ".[test]"
```

Verify the install:

```bash
ruff check paperscraper tests
pytest
```

## Running on ASU Sol

The steps above cover a local machine. To run on Sol's Intel Gaudi nodes, see
[`examples/sol_gaudi/README.md`](../examples/sol_gaudi/README.md) — the install differs there (the
CUDA build of torch has to be kept out) and the model server is scheduled
through SLURM.

## Adding a dependency

Add it to `dependencies` (or `optional-dependencies.test`) in `pyproject.toml`
and re-run `pip install -e ".[test]"`. Do not add it to `environment.yml` —
that file intentionally pins only the interpreter, so there is a single source
of truth for the dependency set.
