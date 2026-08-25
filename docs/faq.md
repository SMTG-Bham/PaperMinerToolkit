# Troubleshooting

## A command is missing

Confirm that the intended environment is active and reinstall the editable package:

```bash
python -m pip install -e .
command -v pm
pm scrape --help
```

## Search returns fewer papers than expected

- Query several providers because their coverage differs.
- Increase `--count`; it is a maximum, not a guaranteed result count.
- Prefer ORCID over author-name matching for author imports.
- Inspect the author review CSV before concluding that works were omitted.
- Remember that provider metadata may lack DOIs, abstracts, or affiliations.

## Content is not downloaded

Run `pm corpus stats` and inspect configured credentials with the relevant key commands. Access is source- and publisher-dependent; metadata discovery does not imply PDF availability. Repeat `--source` to test individual PDF sources. Use `--force` only when refreshing an asset that is already stored.

## A filter reports unavailable papers

The requested abstract, text, or PDF may not exist, PDF extraction may have failed, or a regex may have timed out. Narrow expensive patterns, increase the timeout deliberately, or download the missing content. `pm filter status` reports the most frequent unavailable reasons.

## Topic terms are not meaningful

- Train on abstracts or full text rather than titles alone.
- Remove corpus-wide generic words with `--stopwords-file`.
- Keep bigrams enabled so domain phrases survive.
- Compare several topic counts and random seeds.
- Inspect representative papers, not just the top terms.
- Treat warnings about small corpora or limited vocabulary as model-quality warnings, not cosmetic messages.

## A topic filter is stale

The paper text fingerprint no longer matches the stored prediction. Refresh it with the same model name:

```bash
pm topics store topic_model papers.db --name MODEL_NAME
```

## Text is split into many chunks

Check the configured model input limit and the server's true context size. Reduce source length, enable an appropriate compression mode, or use a longer-context model. Increasing the configured number without corresponding model capacity causes failed requests rather than better extraction.

## A scrape rerun does nothing

Successful stages are skipped. Pass `--force` to rescrape, or use `pm reset` when deliberately resetting wider pipeline state. Inspect `pm status` before resetting anything.

## Documentation does not build

```bash
python -m pip install -e '.[docs]'
make -C docs clean html
```

The documentation build treats warnings as errors. Fix broken cross-references, import failures, and invalid notebook metadata instead of suppressing them.
