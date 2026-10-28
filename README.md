<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/SMTG-Bham/PaperMinerToolkit/main/assets/Paper_Miner_Toolkit_banner_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/SMTG-Bham/PaperMinerToolkit/main/assets/Paper_Miner_Toolkit_banner_light.svg">
    <img src="https://raw.githubusercontent.com/SMTG-Bham/PaperMinerToolkit/main/assets/Paper_Miner_Toolkit_banner_light.svg" alt="PaperMinerToolkit banner" width="640">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/SMTG-Bham/PaperMinerToolkit/actions/workflows/tests.yml"><img src="https://github.com/SMTG-Bham/PaperMinerToolkit/actions/workflows/tests.yml/badge.svg?branch=main" alt="Tests"></a>
  <a href="https://codecov.io/gh/SMTG-Bham/PaperMinerToolkit"><img src="https://codecov.io/gh/SMTG-Bham/PaperMinerToolkit/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://pypi.org/project/paperminertoolkit/"><img src="https://img.shields.io/pypi/v/paperminertoolkit?cacheSeconds=300&amp;logo=pypi&amp;logoColor=white" alt="PyPI version"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&amp;logoColor=white" alt="Python 3.11 or newer"></a>
  <a href="https://paperminertoolkit.readthedocs.io/"><img src="https://img.shields.io/badge/Docs-Read%20the%20Docs-8CA1AF?logo=readthedocs&amp;logoColor=white" alt="Documentation"></a>
</p>

# PaperMinerToolkit

PaperMinerToolkit builds scientific-paper corpora and extracts structured, recipe-defined data with configurable text and vision models. It searches Elsevier/Scopus, CORE, OpenAlex, PubMed, arXiv, medRxiv, bioRxiv, and chemRxiv; supplements paper metadata from Crossref, OpenAlex, PubMed, arXiv, medRxiv, bioRxiv, and chemRxiv, and imports an author's works from Crossref; downloads abstracts, full text, and PDFs; supports persistent regex and LDA topic filters; and stores source content and pipeline state in SQLite.

The complete user guide, CLI reference, Python API, HPC instructions, and rendered notebooks live in the [documentation source](https://paperminertoolkit.readthedocs.io/). The repository is ready for Read the Docs; a hosted link will be added after the project is imported.

## Installation

PaperMinerToolkit requires Python 3.11 or newer:

```bash
pip install paperminertoolkit
```

For local entity extraction with fine-tuned BERT checkpoints, install
`pip install 'paperminertoolkit[bert]'`.

## Quickstart

Probe a public source, then gather a small corpus of abstracts. These steps need no model or API key:

```bash
pmt probe --source arxiv

pmt gather "lithium solid electrolyte" papers.db \
  --source arxiv --count 5 --format abstract --download-source arxiv
pmt corpus stats papers.db
```

`pmt gather` combines search and download, merging duplicate records and downloading only papers matched by that search. Repeat `--source` to search several providers and `--download-source` to select content providers separately. Configure only the [provider credentials](https://paperminertoolkit.readthedocs.io/en/latest/workflow/configuration.html) you need. Use `--json` on `probe` or `gather` for scriptable summaries.

To extract structured records from those abstracts, configure a text model:

```bash
pmt config model text --provider openai --model YOUR_TEXT_MODEL

pmt scrape papers.db sse \
  --mode abstract \
  --output temp_scraped_materials.csv

pmt store papers.db \
  temp_scraped_materials.csv \
  materials.csv \
  sse \
  --assume-yes
```

Use `pmt status papers.db` to inspect pipeline progress. `pmt search` and `pmt download` remain available for separate discovery and content retrieval stages.

## BERT entity extraction

Annotate stored abstracts with a fine-tuned token-classification checkpoint:

```bash
pmt entities corpus papers.db entities.jsonl \
  --model ./models/matscibert-ner --field abstract
```

Replace the example path with your trained checkpoint directory or Hugging Face
model ID. Use `pmt entities text paper.txt entities.jsonl --model CHECKPOINT` for
a UTF-8 file. Results include entity labels, confidence scores, exact text,
character offsets, and source metadata; long documents use overlapping windows.

The published [MatSciBERT](https://huggingface.co/m3rg-iitd/matscibert) base model
requires entity fine-tuning. The published
[polyBERT](https://huggingface.co/HAYDERphd/polyBERT) produces polymer PSMILES
fingerprints and is not a prose entity tagger. Supply a compatible fine-tuned
token classifier; training is not part of this workflow. See the
[entity extraction guide](https://paperminertoolkit.readthedocs.io/en/latest/workflow/entities.html).

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
ruff check paperminertoolkit tests
pytest
```

PaperMinerToolkit is currently alpha software. Keep the corpus database, recipe, model configuration, intermediate extraction CSV, and final results together so a workflow can be reviewed and reproduced.
