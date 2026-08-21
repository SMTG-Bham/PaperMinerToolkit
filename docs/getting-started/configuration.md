# Credentials and model configuration

PaperScraper keeps search/download credentials separate from the text and vision model profiles. Secrets can be supplied through environment variables, which is best for batch jobs, or saved interactively with the `ps_*_key` commands.

## Search and download services

| Service | Environment variable | Interactive command | Used for |
| --- | --- | --- | --- |
| Elsevier | `ELSEVIER_API_KEY` | `ps_elsevier_key` | Scopus search, full text, and eligible PDFs |
| CORE | `CORE_API_KEY` | `ps_core_key` | Search, abstracts, and PDFs |
| OpenAlex | `OPENALEX_API_KEY` | `ps_openalex_key` | Higher API budget for search, abstracts, and OA locations |
| Unpaywall | `UNPAYWALL_EMAIL` | `ps_unpaywall_email` | Open-access PDF discovery |

OpenAlex works without a key, but authenticated use has a substantially larger credit budget. Crossref author imports require a contact email on each command.

## Hosted model providers

Save provider keys interactively or export `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`:

```bash
ps_openai_key
ps_anthropic_key
```

Configure text and vision independently:

```bash
ps_model_config text --provider openai --model YOUR_TEXT_MODEL
ps_model_config vision --provider openai --model YOUR_VISION_MODEL
ps_model_status
```

Use model identifiers available to your provider account. PaperScraper infers ordinary capabilities from the provider and model name; `--capability` is an override for unusual or locally served models.

## Local OpenAI-compatible servers

Point both profiles at the local endpoint when a model supports text and images:

```bash
ps_model_config text \
  --provider local \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --base-url http://127.0.0.1:8000/v1

ps_model_config vision \
  --provider local \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --base-url http://127.0.0.1:8000/v1
```

Requests default to `temperature=0` and `top_p=1` for repeatable extraction. Environment variables prefixed with `PAPERSCRAPER_MODEL_` configure the text profile; `PAPERSCRAPER_VISION_MODEL_` configures vision.

:::{warning}
Never put real keys in notebooks, recipe files, shell scripts, or committed configuration. Prefer your scheduler's secret mechanism or environment variables for unattended jobs.
:::
