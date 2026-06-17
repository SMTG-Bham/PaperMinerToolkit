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


def _full_text_papers(papers: pd.DataFrame) -> pd.DataFrame:
    if papers.empty or "link" not in papers.columns:
        return pd.DataFrame()
    return papers[papers["link"].astype(str).str.contains("full-text", na=False)].copy()


def search_two_papers(query: str, papers_path: str):
    queries = [
        query,
        "Li2NH lithium nitride hydride solid electrolyte",
        "lithium nitride hydride solid electrolyte",
        "lithium solid electrolyte AIMD",
        "lithium solid electrolyte",
    ]
    selected = []
    seen = set()
    for search_query in queries:
        papers = document_search(search_query, count=25, get_all=False)
        candidates = _full_text_papers(papers)
        for _, paper in candidates.iterrows():
            identifier = paper.get("dc:identifier") or paper.get("prism:doi") or paper.get("title")
            if identifier in seen:
                continue
            seen.add(identifier)
            selected.append(paper)
            if len(selected) == 2:
                break
        if len(selected) == 2:
            break

    if len(selected) < 2:
        raise RuntimeError(
            f"Could only find {len(selected)} Elsevier full-text papers. "
            "Try a broader --query or verify Elsevier API access/entitlements."
        )

    papers = pd.DataFrame(selected).reset_index(drop=True)
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
    parser.add_argument("--mode", choices=["text", "images", "text-images"], default="text")
    parser.add_argument("--image-dir", default="examples/bluebear_images")
    parser.add_argument("--temp-results", default="temp_scraped_materials.csv")
    parser.add_argument("--out-file", default="examples/bluebear_materials.csv")
    parser.add_argument("--base-url", default=os.environ.get("PAPERSCRAPER_MODEL_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8")
    args = parser.parse_args()

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
        mode=args.mode,
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
