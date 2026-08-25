<p class="docs-hero-banner">
  <img class="only-light" src="_static/Paper_Miner_Toolkit_banner_light.svg" alt="PaperMinerToolkit" width="800">
  <img class="only-dark" src="_static/Paper_Miner_Toolkit_banner_dark.svg" alt="PaperMinerToolkit" width="800">
</p>

# Build literature datasets with PaperMinerToolkit

<p class="hero-tagline">Build scientific-paper corpora and extract structured, recipe-defined data with configurable language and vision models.</p>

PaperMinerToolkit searches Elsevier/Scopus, CORE, OpenAlex, PubMed, arXiv, medRxiv, bioRxiv, and chemRxiv, and supplements what it finds from Crossref; downloads abstracts, full text, and PDFs; filters papers; and extracts recipe-defined records. Metadata, source documents, processing state, filters, and topic-model predictions live together in a portable SQLite corpus.

Start with {doc}`installation`, then follow the {doc}`workflow/index` for a small end-to-end scrape, including LDA model selection, training, trends, and topic filtering. Runnable workflows are collected under {doc}`examples/index`.

## From discovery to dataset

The SQLite corpus is the central hub of every PaperMinerToolkit workflow. Search results and imported records enter the corpus, where PaperMinerToolkit can enrich their metadata and add available abstracts, full text, and PDFs. The same corpus then supports topic modelling, trend analysis, paper filtering, and recipe-defined LLM extraction. Each stage records its outputs and processing state, making the workflow traceable, resumable, and easy to refine.

```{mermaid}
:alt: PaperMinerToolkit workflow arranged around a central SQLite corpus. Search enters from above, enrichment and downloading form return loops below it, LDA topic analysis descends on the left, and LLM extraction descends on the right.
:align: center

%%{init: {"flowchart": {"curve": "basis", "nodeSpacing": 32, "rankSpacing": 42}}}%%
flowchart TB
    A("1 · Search<br/>or import")
    C("2 · Enrich<br/>metadata<br/>(optional)")
    D("3 · Download<br/>content")
    B[("SQLite corpus")]
    E("4 · LDA topic<br/>model<br/>(optional)")
    F("5 · Topics<br/>and trends")
    L("6 · Regex or<br/>topic filtering<br/>(optional)")
    G("7 · LLM recipe<br/>scrape")
    I("8 · Convert units<br/>and store records")

    A --> B
    C --> B
    B --> C
    D --> B
    B --> D
    B --> E
    E --> F
    B --> L
    L --> G
    G --> I
    E --> L

    classDef search fill:#bfc0c1,stroke:#7f8081,stroke-width:2px,color:#202124,font-weight:600
    classDef enrich fill:#e7c08f,stroke:#b47b38,stroke-width:2px,color:#202124,font-weight:600
    classDef download fill:#bfd5ca,stroke:#729986,stroke-width:2px,color:#202124,font-weight:600
    classDef llm fill:#dfbbb4,stroke:#a97065,stroke-width:2px,color:#202124,font-weight:600
    classDef lda fill:#b7cbd5,stroke:#728e9d,stroke-width:2px,color:#202124,font-weight:600
    classDef corpus fill:#e4e4e4,stroke:#8b8c8d,stroke-width:4px,color:#202124,font-size:24px,font-weight:bold
    class A search
    class C enrich
    class D download
    class L,G,I llm
    class E,F lda
    class B corpus

    linkStyle 0 stroke:#7f8081,stroke-width:3px
    linkStyle 1,2 stroke:#b47b38,stroke-width:3px
    linkStyle 3,4 stroke:#729986,stroke-width:3px
    linkStyle 5,6,10 stroke:#728e9d,stroke-width:3px
    linkStyle 7,8,9 stroke:#a97065,stroke-width:3px
```

```{toctree}
:hidden:
:maxdepth: 1

Installation <installation>
Workflow <workflow/index>
Examples <examples/index>
FAQ <faq>
API <reference/index>
```
