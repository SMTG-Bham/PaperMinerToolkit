# Installation

PaperMiner requires Python 3.11 or newer. Create an isolated environment before installing it.

## Development installation

Clone the repository, activate your environment, and install the package in editable mode:

```bash
git clone https://github.com/SMTG-Bham/PaperScraper.git
cd PaperScraper
python -m pip install -e .
```

To run the test suite as well:

```bash
python -m pip install -e '.[test]'
```

For documentation work:

```bash
python -m pip install -e '.[test,docs]'
make -C docs html
```

The generated site is written to `docs/_build/html/index.html`.

For a documentation-only environment, use the same lean installation as CI and Read the Docs:

```bash
python -m pip install -r docs/requirements.txt
python -m pip install --no-deps -e .
make -C docs html
```

Sphinx mocks runtime-only scientific and model libraries while reading docstrings. Use the normal editable installation when running PaperMiner itself.

## Verify the installation

Every command should be available after installation:

```bash
ps_status --help
ps_search --help
ps_scrape --help
ps_topics_train --help
```

## HPC environments

Install PaperMiner in a login or build job rather than repeatedly installing it in compute jobs. Local vLLM deployments may need accelerator-specific PyTorch and vLLM builds, so install PaperMiner without dependency resolution when those packages are already managed by the cluster environment:

```bash
python -m pip install -e . --no-deps
```

See {doc}`../user-guide/hpc` for the supplied Intel Gaudi scripts.
