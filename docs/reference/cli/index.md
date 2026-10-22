# CLI

PaperMinerToolkit provides one `pmt` executable. Commands that operate on the same part of the application are collected into groups, so help is available at every level:

```bash
pmt --help
pmt filter --help
pmt filter regex --help
pmt --version
pmt schema corpus export
```

`python -m paperminertoolkit` exposes the same command tree. `pmt schema`
describes the installed commands as JSON, including options, defaults, and
choices. Selected commands accept a command-local `--json` flag; see the
{doc}`agent CLI guide <../../workflow/agents>` for payloads, exit codes, and an
unattended workflow.

## Top-level commands

| Command | Purpose |
| --- | --- |
| `pmt schema` | Describe all commands or one command subtree as JSON. |
| `pmt probe` | Check provider configuration and availability. |
| `pmt gather` | Search and retrieve content for matching papers in one run. |
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
| `pmt entities` | Annotate corpus or file text using fine-tuned BERT token classifiers. |
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
entities
imports
configuration
recipes
```
