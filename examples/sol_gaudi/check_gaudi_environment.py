"""Reject packages that shadow the activated Habana vLLM environment.

Run this with the server environment's Python before loading the Gaudi model.
Locating packages without importing them also diagnoses mismatched Torch builds.
"""

from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import pathlib
import sys


def main() -> None:
    """Report package origins and exit if required packages are missing or shadowed."""
    prefix = pathlib.Path(sys.prefix).resolve()
    shadowed = []
    for name in ("torch", "torchvision", "habana_frameworks"):
        # find_spec locates a package without importing it, so a mismatched pair is
        # reported here rather than raising on the way in.
        spec = importlib.util.find_spec(name)
        if spec is None:
            sys.exit(f"ERROR: {name} is not importable in {prefix}.")
        if spec.origin and spec.origin != "namespace":
            locations = [pathlib.Path(spec.origin).resolve().parent]
        else:
            # habana_frameworks is a namespace package: no top-level __init__.py, so
            # no origin, only the directories the package spans.
            locations = [pathlib.Path(p).resolve() for p in (spec.submodule_search_locations or [])]
        if not locations:
            sys.exit(f"ERROR: {name} is not importable in {prefix}.")
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = "?"
        shown = ", ".join(str(location) for location in locations)
        print(f"{name} {version} from {shown}")
        if any(prefix not in location.parents for location in locations):
            shadowed.append(name)

    if shadowed:
        sys.exit(
            f"ERROR: shadowed by another install: {', '.join(shadowed)}\n"
            f"       These resolve outside {prefix}, so the Habana build is not the\n"
            "       one that loads and vLLM will fail during startup. Fix it with\n"
            "         python -m pip uninstall -y torch torchvision torchaudio\n"
            "       run with no environment active, or move the offending directory\n"
            "       aside. Never install into the shared vLLM environment itself."
        )


if __name__ == '__main__':
    main()
