"""Run a bounded CLI search and inspect its corpus without importing toolkit APIs.

Run with ``--help`` for usage. No network requests occur without an explicit
query and corpus path. The example invokes the active Python environment and
defaults to arXiv; it does not configure or call a language model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def run_json(*arguments: str, timeout: float) -> tuple[Any, int]:
    """Parse stdout before interpreting a command's return code.

    Progress and errors are forwarded to stderr. A nonzero result with valid
    JSON is returned to the caller so a partial search can still be inspected.
    """
    result = subprocess.run(
        [sys.executable, "-m", "paperminertoolkit", *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if not result.stdout.strip():
        raise RuntimeError(
            f"Command {arguments[0]!r} exited {result.returncode} without JSON."
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Command {arguments[0]!r} did not return valid JSON "
            f"(exit {result.returncode})."
        ) from error
    return payload, result.returncode


def main(argv: list[str] | None = None) -> int:
    """Search explicitly selected providers and print results plus corpus state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Search expression; pass it as one argument.")
    parser.add_argument("--corpus", required=True, type=Path, help="SQLite output path.")
    parser.add_argument(
        "--source", action="append", help="Provider to search; repeat as needed (default: arxiv)."
    )
    parser.add_argument("--count", type=int, default=5, help="Results per source (1–1000).")
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="Timeout in seconds per command."
    )
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    if not 1 <= args.count <= 1000:
        parser.error("--count must be between 1 and 1000")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.query.strip():
        parser.error("query must not be blank")

    source_arguments = [
        item for source in args.source or ["arxiv"] for item in ("--source", source)
    ]
    report: dict[str, Any] = {}
    exit_code = 0
    try:
        corpus = args.corpus.expanduser().resolve()
        corpus.parent.mkdir(parents=True, exist_ok=True)
        search, search_code = run_json(
            "search", args.query, str(corpus), *source_arguments,
            "--count", str(args.count), "--store-abstract", "--json",
            timeout=args.timeout,
        )
        report["search"] = search
        exit_code = search_code
        if search_code not in (0, 1):
            raise RuntimeError(f"Search exited with unexpected code {search_code}.")
        if search_code == 1:
            print("Search incomplete; preserving its JSON and inspecting saved papers.", file=sys.stderr)

        papers, list_code = run_json(
            "corpus", "list", str(corpus), "--limit", "20", "--offset", "0",
            "--search-id", str(search["search_id"]), "--json",
            timeout=args.timeout,
        )
        report["corpus"] = papers
        if list_code:
            raise RuntimeError(f"Corpus listing exited with code {list_code}.")

        if search["paper_ids"]:
            paper, show_code = run_json(
                "corpus", "show", str(corpus), search["paper_ids"][0], "--json",
                timeout=args.timeout,
            )
            report["first_match"] = paper
            if show_code:
                raise RuntimeError(f"Paper inspection exited with code {show_code}.")
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        exit_code = 1
    if report:
        print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
