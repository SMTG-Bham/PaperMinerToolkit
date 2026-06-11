# PaperScraper

PaperScraper searches Elsevier/Scopus, downloads paper content, and extracts structured materials data from paper text with configurable language models.

## Model configuration

Run `ps_model_config` to set the default model provider, model name, optional base URL, API key, and declared capabilities.

Supported providers:

- `openai`: OpenAI Responses API.
- `anthropic` or `claude`: Anthropic Messages API.
- `openai-compatible`, `local`, or `hpc`: OpenAI-compatible chat endpoint, such as vLLM or another local/HPC inference server.

For a one-off scrape, override the configured model from the CLI:

`ps_scrape papers papers.csv sse --provider local --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-72B-Instruct`

## Downloading papers

`ps_elsevier papers.csv papers --format text` downloads Elsevier full text as text.

`ps_elsevier papers.csv papers --format pdf` downloads PDF files when the Elsevier API and your entitlement allow it.

`ps_elsevier papers.csv papers --format both` downloads both formats.

## Scraping content

`ps_scrape papers papers.csv sse --content text` extracts structured data from text. This is the default.

`ps_scrape papers papers.csv sse --content images --image-dir paper_images` extracts embedded images from PDFs without sending them to a text model.

`ps_scrape papers papers.csv sse --content both` extracts PDF images and structured text data in one pass.

## BlueBEAR example

The examples folder contains a two-paper SLURM workflow for BlueBEAR using `Qwen/Qwen3.6-35B-A3B-FP8`:

`sbatch examples/bluebear_qwen_two_papers.sbatch`

Set `ELSEVIER_API_KEY` and `PAPERSCRAPER_MODEL_BASE_URL` in the job environment before submitting.
