# Workflow

PaperMiner keeps literature discovery, downloaded content, filter decisions, extraction state, and model results in one SQLite corpus. A typical run follows the stages below; each stage is resumable and already stored content is skipped by default.

```{mermaid}
flowchart LR
    A[Search or import] --> B[(Corpus database)]
    B --> S[Supplement metadata]
    S --> C[Download content]
    C --> D[Filter papers]
    D --> E[Scrape with recipe]
    E --> F[Store results]
```

For a first run, follow the quickstart. The remaining pages explain each production stage, model and credential configuration, and cluster execution in more detail.

```{toctree}
:maxdepth: 1

getting-started/quickstart
getting-started/configuration
user-guide/corpus
user-guide/filtering
user-guide/recipes
user-guide/scraping
user-guide/topics
user-guide/hpc
```
