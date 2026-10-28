# Entity extraction with fine-tuned BERTs

Use `pmt entities` to annotate materials, properties, processes, or other named
entities with a locally executed Hugging Face token-classification checkpoint.
The labels come from that checkpoint's training and `id2label` configuration.
The output is JSONL containing the original text, source information, and entity
spans with confidence scores and character offsets.

## Install and choose a checkpoint

```bash
python -m pip install 'paperminertoolkit[bert]'
```

Choose a model already fine-tuned for token classification, either a Hugging
Face model ID or a local directory saved with its model, configuration, and
tokenizer files. A fast tokenizer is required to preserve character offsets
and process long inputs. BERT and other architectures supported by
`AutoModelForTokenClassification` can be used when their checkpoint and
tokenizer meet these requirements.

The two example base models have different purposes:

| Model | Published capability | Entity extraction requirement |
| --- | --- | --- |
| [MatSciBERT](https://huggingface.co/m3rg-iitd/matscibert) | A materials-language model published for masked language modelling. | Use a derivative fine-tuned on the entity labels needed for your materials text. |
| [polyBERT](https://huggingface.co/HAYDERphd/polyBERT) | Encodes polymer PSMILES strings as numerical fingerprints. | The published encoder is not a prose entity tagger; use a token classifier trained for the input and labels you need. |

PaperMinerToolkit performs inference with an existing checkpoint; it does not
train one. It rejects incompatible base-model architectures and incomplete
token-classification weights instead of using a newly initialized prediction
head. Select a checkpoint appropriate to your corpus and inspect its labels
and extraction quality before using the annotations in an analysis.

## Extract corpus entities

In the examples below, replace `./models/matscibert-ner` with your fine-tuned
checkpoint directory or Hugging Face model ID:

```bash
pmt download papers.db --format abstract

pmt entities corpus papers.db entities.jsonl \
  --model ./models/matscibert-ner \
  --field abstract
```

`abstract` is the default field. Repeat `--field` to annotate titles or stored
full text as separate documents:

```bash
pmt entities corpus papers.db entities.jsonl \
  --model ./models/matscibert-ner \
  --field title --field abstract --field text \
  --min-score 0.8 \
  --label MAT --label PROPERTY
```

`MAT` and `PROPERTY` are illustrative labels: use the exact entity labels
produced by your checkpoint, with any BIO prefix removed by aggregation.
Omit `--label` to retain all labels. `--min-score` retains scores greater than
or equal to the specified value.

The command visits all corpus papers, independently of active regex or topic
filters. Missing and blank fields are skipped. Abstract and full-text fields
use the latest stored asset. This workflow does not download missing text or
change recipe-scraping statuses.

## Extract a text file

```bash
pmt entities text paper.txt paper_entities.jsonl \
  --model ./models/matscibert-ner \
  --local-files-only
```

The input must be UTF-8 text. The complete file, including its original
newlines, is treated as one document. PDF and XML documents must first be
converted to text or imported into the corpus through the existing workflows.

## Long documents and local execution

Long documents are processed in overlapping token windows, so text beyond the
first window is included. Compatible BIO predictions are aggregated into
spans, and overlapping window predictions are reconciled by the Transformers
token-classification pipeline.

| Option | Default | Purpose |
| --- | --- | --- |
| `--device` | `cpu` | Select a PyTorch device such as `cuda:0`. |
| `--batch-size` | `8` | Limit token windows per inference batch. |
| `--stride` | `64` | Set token overlap between adjacent windows. |
| `--max-length` | Model/tokenizer limit | Reduce the window length, including special tokens. |
| `--revision` | Checkpoint default | Select a Hugging Face branch, tag, or commit. |
| `--cache-dir` | Hugging Face default | Choose the model download cache. |
| `--local-files-only` | Disabled | Require a local or previously cached checkpoint. |

The stride must be smaller than the number of content tokens in a window.
Explicit window lengths cannot exceed the model's supported limit. Reduce
`--batch-size` if accelerator memory is limited. Use `--revision` with a commit
hash when fixing a Hub checkpoint version for a reproducible run.

## Read the output

Each JSONL row represents one source document or corpus field, even if it has
no detected entities. A row includes `text`, `text_sha256`, `source`, `model`,
`revision`, `extraction` settings, and `entities`. Corpus rows additionally
identify the paper, DOI, field, and source blob with `paper_id`, `doi`, `field`,
and `blob_id`, and record the absolute corpus database path as `corpus`.

Each entity has this shape:

```json
{"label": "MAT", "text": "LiFePO4", "score": 0.98, "start": 0, "end": 7}
```

This illustrative span uses Python character offsets: `start` is inclusive
and `end` is exclusive, so `row['text'][start:end]` equals the entity's `text`.
Offsets are relative to the original field or file, not a token window.
Scores are the classifier's confidence values, not an accuracy guarantee.

Existing output files are preserved unless you pass `--overwrite`. A failed
run does not publish a partial JSONL file. Input files and corpus databases
cannot be used as their own output.

Entity spans are annotations of text. They are not recipe records accepted by
`pmt store`, and this workflow does not generate polymer fingerprints. Use
{doc}`scraping` for recipe-defined structured records.

## Python API

```python
from paperminertoolkit.extraction.entities import (
    EntityExtractionConfig,
    TransformerEntityExtractor,
)
from paperminertoolkit.workflows.entities import extract_corpus_entities

config = EntityExtractionConfig(
    model='./models/matscibert-ner',
    device='cpu',
    local_files_only=True,
)
extractor = TransformerEntityExtractor(config)
entities = extractor.extract('LiFePO4 was synthesized by a solid-state reaction.')
# One entity list per input, including when passing a single string.
print(entities[0])

summary = extract_corpus_entities(
    'papers.db', 'entities.jsonl', config,
    text_fields=('abstract',), min_score=0.8,
)
```

Model files are loaded lazily on the first nonblank input and reused by that
extractor. See the {doc}`../reference/cli/entities`,
{doc}`../reference/modules/extraction/entities`, and
{doc}`../reference/modules/workflows/entities` reference pages for details.
