# PaperScraper

PaperScraper searches Elsevier/Scopus, downloads paper content, and extracts structured materials data from papers with configurable text and vision models.

## Model Profiles

Configure separate model profiles for text and vision analysis:

`ps_model_config text --provider local --model Qwen/Qwen3-30B-A3B-FP8 --base-url http://127.0.0.1:8000/v1 --temperature 0 --top-p 1`

`ps_model_config vision --provider local --model Qwen/Qwen2.5-VL-7B-Instruct --base-url http://127.0.0.1:8001/v1`

Capabilities are inferred automatically from the profile and model name. Use `--capability` only as an override for unusual models. Model requests default to `temperature=0` and `top_p=1` for deterministic extraction.

Inspect configured profiles:

`ps_model_status`

Environment variables still work for batch jobs. `PAPERSCRAPER_MODEL_*` applies to the text profile, while `PAPERSCRAPER_VISION_MODEL_*` applies to the vision profile.

## Workflow

Search:

`ps_search "Li2NH AIMD solid electrolyte" papers.csv`

Import externally downloaded PDFs. This scans each PDF for a DOI and uses Crossref to fill metadata when possible:

`ps_import_pdfs papers external_papers.csv`

Skip Crossref lookup for offline runs while still trying to scrape the DOI from the PDF text:

`ps_import_pdfs papers external_papers.csv --no-crossref`

Download Elsevier content:

`ps_elsevier papers.csv papers --format text`

`ps_elsevier papers.csv papers --format pdf`

`ps_elsevier papers.csv papers --format both`

Scrape text only:

`ps_scrape papers papers.csv sse --mode text`

Analyze images with the vision profile:

`ps_scrape papers papers.csv sse --mode images --vision-provider local --vision-model Qwen/Qwen2.5-VL-7B-Instruct --vision-base-url http://127.0.0.1:8001/v1`

Analyze images with paper text as additional context:

`ps_scrape papers papers.csv sse --mode text-images --image-context paper-text`

Cleanup after successful scraping:

`ps_scrape papers papers.csv sse --mode text --delete-papers-after`

Store results:

`ps_store papers.csv temp_scraped_materials.csv materials.csv sse --assume-yes`

Check pipeline status:

`ps_status papers.csv`

## BlueBEAR Example

The examples folder contains a two-paper SLURM workflow for BlueBEAR using `Qwen/Qwen3-30B-A3B-FP8` for text scraping:

`sbatch examples/bluebear_qwen_two_papers.sbatch`

Set `ELSEVIER_API_KEY` in the job environment before submitting. The script can start a local vLLM server, or you can set `PAPERSCRAPER_START_MODEL_SERVER=0` and provide `PAPERSCRAPER_MODEL_BASE_URL` yourself.
