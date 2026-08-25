# LDA topic analysis

PaperMiner uses a fixed-vocabulary scikit-learn latent Dirichlet allocation model to discover recurring word distributions. Topic names remain manual: inspect the terms and representative papers, then assign names that make sense for the corpus.

## Prepare the corpus

Titles and abstracts are the default input. Download abstracts and inspect coverage before training:

```bash
pm_download papers.db --format abstract
pm_corpus_stats papers.db
```

Use a corpus-specific stopword file to remove terms that occur everywhere:

```text
# domain_stopwords.txt
lithium
battery
study
performance
```

Bigram features such as `solid_electrolyte` are enabled by default, preserving phrases when their individual words are stopwords. Use `--ngram-max 1` only for a deliberate unigram model.

## Compare candidate models

```bash
pm_topics_compare papers.db topic_comparison \
  --topics 6 --topics 8 --topics 10 \
  --seed 0 --seed 1 --seed 2 \
  --field abstract \
  --stopwords-file domain_stopwords.txt
```

The comparison prepares the streaming vocabulary and sparse batches once, then reuses them. `model_comparison.csv` reports perplexity, log likelihood, topic diversity, dominant-topic balance, training time, and cross-seed stability. Metrics narrow the candidates; coherent terms and representative papers decide the final model.

## Train, inspect, and name

```bash
pm_topics_train papers.db topic_model \
  --topics 8 \
  --field abstract \
  --stopwords-file domain_stopwords.txt \
  --batch-size 1000 \
  --iterations 10

pm_topics_show topic_model --representatives 5
pm_topics_name topic_model 0 "sulfide solid electrolytes"
```

Disk-backed streaming is the default and avoids constructing the complete document-term matrix in memory. Put temporary sparse batches on high-capacity scratch with `--cache-dir`. For a small corpus, `--in-memory` supports conventional batch LDA.

Training warns about small corpora, short inputs, missing text, and weak retained vocabularies. The model directory contains the model, vectorizer, immutable model ID, configuration and corpus fingerprints, topic terms, representative papers, and per-paper probabilities.

## Predict and store

Apply the saved model without retraining its topics:

```bash
pm_topics_predict topic_model new_papers.db new_paper_topics.csv
```

Inputs containing no fitted vocabulary terms are reported as `no_vocabulary_terms`, rather than receiving uniform probabilities.

External CSV predictions are useful for analysis. To make a model available to persistent corpus filters, perform a fresh transactional prediction into the corpus:

```bash
pm_topics_store topic_model papers.db --name sse-lda-v1
pm_topics_models papers.db
```

## Trends

Always aggregate one fixed model across time; do not fit separate topics in each period.

```bash
# Annual windows
pm_topics_trends topic_model annual_trends \
  --bin-size 1 --step-size 1 --plot

# Non-overlapping five-year blocks
pm_topics_trends topic_model five_year_trends \
  --bin-size 5 --step-size 5 --plot

# Rolling five-year windows advanced annually
pm_topics_trends topic_model rolling_trends \
  --bin-size 5 --step-size 1 --plot-file rolling_topics.pdf
```

Relative plot filenames are written inside the output directory. PNG is the default; the filename extension selects another Matplotlib format. Trend data include mean and summed probability, dominant-paper count and share, coverage, and partial-window state. Use `--predictions` to analyse a CSV created by `pm_topics_predict`.

Avoid interpreting sharp movements in bins containing very few papers. Compare probability prevalence with the paper-count panel and inspect representative papers before assigning a scientific explanation.

## Filter with topics

After storing a model, use `pm_filter_topic` alone or compose it with regex filters. See {doc}`filtering` for the definition schema, stale-score protection, and hybrid stack semantics.
