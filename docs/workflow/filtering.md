# Filtering papers

Filters are applied after content downloading. They persist named decisions and their definitions in the corpus without deleting papers or changing scrape status. When a filter stack is active, scraping processes only the final `included` set unless `--ignore-filters` is supplied.

## Regex definitions

A regex definition contains positive rules and optional veto rules:

```json
{
  "name": "band-gap-materials",
  "description": "Papers concerning semiconductor band gaps",
  "fields": ["title", "abstract", "full_text"],
  "case_sensitive": false,
  "include_mode": "any",
  "timeout_ms": 500,
  "include": [
    {"name": "band-gap", "pattern": "\\bband[ -]?gaps?\\b"},
    {"name": "electronic-gap", "pattern": "\\belectronic\\s+gaps?\\b"}
  ],
  "exclude": [
    {"name": "background-only", "pattern": "\\breview of unrelated systems\\b"}
  ]
}
```

Apply a first filter and compose subsequent filters explicitly:

```bash
pmt filter regex papers.db band_gap.json
pmt filter regex papers.db experimental.json --join or
pmt filter regex papers.db oxide.json --join and
```

Operators are evaluated from left to right, so this stack becomes `((band-gap-materials OR experimental) AND oxide)`. `include_mode` controls whether any or all positive rules must match; every matching exclusion vetoes that filter. Repeat `--field` or pass `--timeout-ms` to override those settings for an application.

Stored text is searched before falling back to PDF extraction, and reference sections are ignored. A paper is `unavailable` when selected content is missing or unreadable, or when matching times out without another field proving a positive match.

Example title filters are available in `examples/filters/`.

## Topic definitions

First store predictions from a named model in the corpus:

```bash
pmt topics store topic_model papers.db --name sse-lda-v1
```

Then create a topic filter:

```json
{
  "name": "solid-electrolyte-topics",
  "description": "Topic-model relevance filter",
  "model": "sse-lda-v1",
  "include_mode": "any",
  "include": [
    {
      "name": "electrolyte-topic",
      "topic_id": 2,
      "min_probability": 0.35,
      "require_dominant": false
    }
  ],
  "exclude": [
    {
      "name": "photovoltaics-veto",
      "topic_id": 7,
      "require_dominant": true
    }
  ]
}
```

```bash
pmt filter topic papers.db solid_electrolyte_topics.json
```

Rules can require a minimum probability, dominance, or both. Exclusion rules take precedence. If source text changes after prediction, the topic filter becomes stale and scraping fails closed until the predictions are refreshed.

## Hybrid filtering and maintenance

Regex and topic filters share one stack:

```bash
pmt filter regex papers.db broad_materials.json
pmt filter topic papers.db focused_topics.json --join and
pmt filter status papers.db
```

Replacing an existing named filter requires `--replace`. Remove one filter or the complete stack with:

```bash
pmt filter reset papers.db --name focused-topics
pmt filter reset papers.db --all
```

Use `pmt scrape ... --ignore-filters` only for a deliberate one-run bypass. With no active filters, all eligible papers retain the original processing behavior.
