---
orphan: true
---

# Agent CLI implementation plan — issue #91

The existing nested `pmt` CLI already runs the main paper-mining workflows.
This work makes it discoverable and usable by an agent that invokes subprocesses,
parses stdout, checks exit codes, and resumes work from a persistent corpus.

## Stage 1: discover commands and consume workflow results

- Add `pmt --version`, `python -m paperminertoolkit`, and `pmt schema` with a
  versioned JSON description generated from the actual Click command tree.
- Extend the existing command-local `--json` convention to search, download,
  enrichment, corpus statistics/history, pipeline/filter status, and model
  configuration status. Keep progress and errors on stderr and preserve current
  human-readable defaults and existing gather/probe payloads.
- Add recipe listing and inspection so agents can choose a recipe before making
  model calls. Configuration output must omit API keys.
- Test discovery, clean JSON, invalid input, and partial search results offline.

## Stage 2: inspect papers and retrieve stored assets

- Add bounded, deterministic corpus listing, individual-paper inspection, and
  export of a selected stored abstract, text, or PDF to an explicit output path.
- Use paper IDs from search/gather/list results. Expose metadata, pipeline state,
  asset metadata, and provenance without loading binary content into JSON.
- Open these new inspection commands read-only; fail clearly for missing or
  incompatible databases, missing papers/assets, and existing export targets.
- Test pagination, asset fidelity, error paths, and absence of database mutations.

## Stage 3: document and verify an agent workflow

- Publish a linked agent guide covering discovery, environment configuration,
  bounded acquisition, inspection/export, extraction, and resuming failed work.
- Specify JSON shapes, stdout/stderr behavior, exit codes, side effects, and the
  commands that can prompt. Provide a standard-library subprocess example that
  parses partial results, uses explicit paths, and closes stdin.
- Update the README and generated CLI-reference navigation.
- Run the offline suite, Ruff, source compilation, Sphinx, and package checks.

## Scope and compatibility

JSON is opt-in on the documented commands, not a global output wrapper. Errors
retain Click's stderr diagnostics and exit codes (0 success, 1 operational
failure, 2 usage error); an early failure may have no JSON payload. Existing
acquisition summaries and flags remain compatible. Download/enrichment summaries
must not imply every paper succeeded merely because the command completed.
Network-dependent tests and paid model/provider calls are outside offline
validation. Each implementation stage is committed separately with `#91` in
its commit title.

## Completion and verification

All three stages were completed on 22 October 2026. The implementation keeps
the existing gather/probe payloads and human output, adds generated command
discovery and JSON summaries, and supplies read-only corpus inspection with
explicit asset export. The {doc}`../workflow/agents` guide and
`examples/agent_cli.py` describe the complete agent workflow.

Validation: 1,718 offline tests passed (32 network/slow tests deselected), Ruff
and source compilation passed, and Sphinx built with warnings treated as errors.
The wheel and source distribution built successfully and passed strict Twine
metadata checks. Tests of the local validation server required localhost socket
access outside the restricted sandbox; they passed with that access enabled.
