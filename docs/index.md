# Build literature datasets with PaperScraper

<p class="hero-tagline">Build scientific-paper corpora and extract structured materials data with configurable language and vision models.</p>

PaperScraper searches Crossref, Elsevier/Scopus, CORE, and OpenAlex; downloads abstracts, full text, and PDFs; filters papers; and extracts recipe-defined records. Metadata, source documents, processing state, filters, and topic-model predictions live together in a portable SQLite corpus.

Start with {doc}`getting-started/installation`, then follow the {doc}`workflow` for a small end-to-end scrape, including LDA model selection, training, trends, and topic filtering. Runnable workflows are collected under {doc}`examples/index`.

```{mermaid}
flowchart LR
    A[Search or import] --> B[(Corpus database)]
    B --> S[Supplement metadata]
    S --> C[Download content]
    C --> D[Filter papers]
    D --> E[Scrape with recipe]
    E --> F[Store results]
    B --> G[Train LDA model]
    G --> H[Trends and topic filters]
```

```{toctree}
:hidden:
:maxdepth: 2

Installation <getting-started/installation>
Workflow <workflow>
Examples <examples/index>
FAQ <user-guide/troubleshooting>
API <reference/index>
```
