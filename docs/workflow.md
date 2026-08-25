# Workflow

PaperMiner keeps literature discovery, downloaded content, filter decisions, extraction state, and model results in one SQLite corpus. A typical run follows the stages below; each stage is resumable and already stored content is skipped by default.

```{mermaid}
:alt: PaperMiner workflow arranged around a central SQLite corpus. Search enters from above, enrichment and downloading form return loops below it, LDA topic analysis descends on the left, and LLM extraction descends on the right.
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
