# CLI

PaperMiner provides one `pm` executable. Commands that operate on the same part of the application are collected into groups, so help is available at every level:

```bash
pm --help
pm filter --help
pm filter regex --help
```

## Top-level commands

| Command | Purpose |
| --- | --- |
| `pm search` | Find papers and add them to a corpus. |
| `pm enrich` | Supplement stored bibliographic metadata. |
| `pm download` | Retrieve abstracts, full text, and PDFs. |
| `pm scrape` | Extract recipe-defined records with language or vision models. |
| `pm store` | Validate, convert, and store extracted records. |
| `pm status` | Inspect pipeline progress. |
| `pm reset` | Reset pipeline statuses for deliberate reruns. |

## Command groups

| Group | Purpose |
| --- | --- |
| `pm corpus` | Inspect corpus contents and storage. |
| `pm filter` | Apply, inspect, and reset regex or topic filters. |
| `pm topics` | Train, compare, inspect, name, apply, and store LDA models. |
| `pm import` | Import local PDFs or an author's works. |
| `pm config` | Configure model profiles and provider credentials. |

Exact arguments and options are generated from the installed commands on the pages below. For task-oriented instructions, use the {doc}`../../workflow/index` and the {doc}`../../examples/index`.

```{toctree}
:maxdepth: 1

core
corpus
filtering
topics
imports
configuration
```
