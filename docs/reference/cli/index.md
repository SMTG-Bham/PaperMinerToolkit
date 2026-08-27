# CLI

PaperMinerToolkit provides one `pmt` executable. Commands that operate on the same part of the application are collected into groups, so help is available at every level:

```bash
pmt --help
pmt filter --help
pmt filter regex --help
```

## Top-level commands

| Command | Purpose |
| --- | --- |
| `pmt search` | Find papers and add them to a corpus. |
| `pmt enrich` | Supplement stored bibliographic metadata. |
| `pmt download` | Retrieve abstracts, full text, and PDFs. |
| `pmt scrape` | Extract recipe-defined records with language or vision models. |
| `pmt store` | Validate, convert, and store extracted records. |
| `pmt status` | Inspect pipeline progress. |
| `pmt reset` | Reset pipeline statuses for deliberate reruns. |

## Command groups

| Group | Purpose |
| --- | --- |
| `pmt corpus` | Inspect corpus contents and storage. |
| `pmt filter` | Apply, inspect, and reset regex or topic filters. |
| `pmt topics` | Train, compare, inspect, name, apply, and store LDA models. |
| `pmt import` | Import local PDFs or an author's works. |
| `pmt config` | Configure model profiles and provider credentials. |
| `pmt recipe` | Inspect recipes and render their LLM prompts. |

Exact arguments and options are generated from the installed commands on the pages below. For task-oriented instructions, use the {doc}`../../workflow/index` and the {doc}`../../examples/index`.

```{toctree}
:maxdepth: 1

core
corpus
filtering
topics
imports
configuration
recipes
```
