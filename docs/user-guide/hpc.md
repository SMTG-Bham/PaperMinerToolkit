# Running on HPC

Search and download stages are network and CPU workloads; model serving and scraping are the accelerator-intensive stages. Build the corpus away from scarce accelerator queues, then submit only inference work to those resources.

## General batch workflow

1. Install PaperMiner in a persistent environment.
2. Store caches and downloaded model weights on project scratch rather than a small home filesystem.
3. Build and inspect the corpus from a login, interactive, or high-throughput CPU job.
4. Start the model server inside the accelerator allocation.
5. Wait for its `/v1/models` endpoint before starting PaperMiner.
6. Configure the local model endpoint, scrape, and store results.
7. Preserve the corpus, final CSV, model configuration, recipe, and scheduler logs.

Environment variables are preferable to interactive settings in batch jobs. Never print secrets in scheduler logs.

## Intel Gaudi scripts on ASU Sol

The `examples/sol_gaudi/` directory contains:

- `install.sbatch` to build and verify an accelerator-compatible environment.
- `fetch_corpus.sh` to search and download outside the Gaudi queue.
- `fetch_corpus.sbatch` to build a larger corpus in an HTC batch job.
- `scrape_gaudi.sbatch` to serve a model and scrape in one allocation.
- `serve_gaudi.sbatch` to keep a server warm for several runs.

Run the scripts from their own directory so relative database, CSV, and log paths remain predictable:

```bash
cd examples/sol_gaudi
sbatch install.sbatch
```

The supplied workflow defaults to biodegradable polymers. Override `PS_RECIPE`, the query, model, context size, and scheduler resources for other projects.

## Context and memory sizing

The model server's advertised context limit, its actual memory allocation, and PaperMiner's configured input limit must agree. A large context can consume substantial accelerator memory before inference begins. Start conservatively, inspect server logs and PaperMiner's chunk plan, then increase only when the server is stable.

Use a bounded LDA `--batch-size` and a scratch `--cache-dir` for corpora containing tens of thousands of papers. Streaming training is specifically designed to keep memory use proportional to the batch rather than the corpus.

For the full cluster-specific command sequence and troubleshooting notes, see the scripts and their adjacent `examples/sol_gaudi/README.md` in the repository.
