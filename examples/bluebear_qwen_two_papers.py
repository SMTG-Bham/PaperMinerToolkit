"""End-to-end two-paper PaperScraper example for BlueBEAR.

This script searches for two papers, downloads available Elsevier content,
scrapes structured data with Qwen on an OpenAI-compatible endpoint, and stores
the results.
"""

import argparse
import os

import pandas as pd

from paperscraper.download import elsevier_downloader
from paperscraper.models import ModelConfig
from paperscraper.scrape import scrape_papers
from paperscraper.search import document_search
from paperscraper.store import store_results


def search_two_papers(query: str, papers_path: str):
    papers = document_search(query, count=2, get_all=False)
    papers = papers.head(2).copy()
    papers["status"] = "retrieved"
    papers.to_csv(papers_path)
    print(f"Saved {len(papers)} papers to {papers_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Li2NH AIMD solid electrolyte")
    parser.add_argument("--papers-path", default="examples/bluebear_papers.csv")
    parser.add_argument("--papers-dir", default="examples/bluebear_papers")
    parser.add_argument("--recipe", default="sse")
    parser.add_argument("--download-format", choices=["text", "pdf", "both"], default="both")
    parser.add_argument("--content", choices=["text", "images", "both"], default="both")
    parser.add_argument("--image-dir", default="examples/bluebear_images")
    parser.add_argument("--temp-results", default="temp_scraped_materials.csv")
    parser.add_argument("--out-file", default="examples/bluebear_materials.csv")
    parser.add_argument("--base-url", default=os.environ.get("PAPERSCRAPER_MODEL_BASE_URL"))
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    args = parser.parse_args()

    if not args.base_url:
        raise ValueError(
            "Set PAPERSCRAPER_MODEL_BASE_URL or pass --base-url for the BlueBEAR model endpoint."
        )

    os.makedirs(os.path.dirname(args.papers_path), exist_ok=True)
    model_config = ModelConfig.from_settings(
        name=args.model,
        provider='hpc',
        base_url=args.base_url,
        capabilities=['text'],
    )
    search_two_papers(args.query, args.papers_path)
    elsevier_downloader(args.papers_path, args.papers_dir, download_format=args.download_format)
    scrape_papers(
        args.papers_dir,
        args.papers_path,
        args.recipe,
        content_mode=args.content,
        image_dir=args.image_dir,
        model=args.model,
        provider='hpc',
        base_url=args.base_url,
    )

    if os.path.isfile(args.temp_results):
        store_results(
            args.papers_path,
            args.temp_results,
            args.out_file,
            True,
            args.recipe,
            assume_yes=True,
            model_config=model_config,
        )
    else:
        print("No temporary scrape results were produced; skipping store step.")


if __name__ == "__main__":
    main()
