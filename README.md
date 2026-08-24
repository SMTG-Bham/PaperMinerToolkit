# PaperMiner

[![Tests](https://github.com/SMTG-Bham/PaperMiner/actions/workflows/tests.yml/badge.svg?branch=dev)](https://github.com/SMTG-Bham/PaperMiner/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/SMTG-Bham/PaperMiner/branch/main/graph/badge.svg)](https://codecov.io/gh/SMTG-Bham/PaperMiner)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/Paper_Miner_banner_dark.svg">
    <img src="assets/Paper_Miner_banner_light.svg" alt="PaperMiner banner" width="640">
  </picture>
</p>

PaperMiner builds scientific-paper corpora and extracts structured, recipe-defined data with configurable text and vision models. It searches Elsevier/Scopus, CORE, OpenAlex, PubMed, arXiv, medRxiv, bioRxiv, and chemRxiv; supplements paper metadata from Crossref, OpenAlex, PubMed, arXiv, medRxiv, bioRxiv, and chemRxiv, and imports an author's works from Crossref; downloads abstracts, full text, and PDFs; supports persistent regex and LDA topic filters; and stores source content and pipeline state in SQLite.

The complete user guide, CLI reference, Python API, HPC instructions, and rendered notebooks live in the [documentation source](docs/index.md). The repository is ready for Read the Docs; a hosted link will be added after the project is imported.

## Installation

PaperMiner requires Python 3.11 or newer:

```bash
git clone https://github.com/SMTG-Bham/PaperMiner.git
cd PaperMiner
python -m pip install -e .
```

## Quickstart

Configure a text model and any search/download credentials you need, then run a small workflow:

```bash
ps_model_config text --provider openai --model YOUR_TEXT_MODEL

ps_search "lithium solid electrolyte" papers.db \
  --source openalex --count 25
ps_enrich papers.db
ps_download papers.db --format abstract

ps_scrape papers.db sse \
  --mode abstract \
  --output temp_scraped_materials.csv

ps_store papers.db \
  temp_scraped_materials.csv \
  materials.csv \
  sse \
  --assume-yes
```

Use `ps_corpus_stats papers.db` to inspect stored content and `ps_status papers.db` to inspect pipeline progress.

## Documentation

Install and build the Sphinx site locally:

```bash
python -m pip install -e '.[docs]'
make -C docs html
```

Open `docs/_build/html/index.html`. Notebook templates for OpenAI, Anthropic, local Qwen/vLLM, LDA model selection, temporal trends, and hybrid filtering are under `docs/examples/`.

## Testing

```bash
python -m pip install -e '.[test]'
ruff check paperscraper tests
pytest
```

PaperMiner is currently alpha software. Keep the corpus database, recipe, model configuration, intermediate extraction CSV, and final results together so a workflow can be reviewed and reproduced.
