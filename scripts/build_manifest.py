#!/usr/bin/env python3
"""Create the package-level SHA-256 manifest in deterministic path order."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST_SHA256.txt"
EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".nox",
}


def is_runtime_artifact(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return any(part in EXCLUDED_PARTS for part in relative.parts) or path.suffix == ".pyc"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and path != OUTPUT and not is_runtime_artifact(path)
    )
    content = "".join(
        f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in files
    )
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"Wrote {OUTPUT.name} with {len(files)} entries")


if __name__ == "__main__":
    main()
