#!/usr/bin/env python3
"""Create the package-level SHA-256 manifest in deterministic path order."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST_SHA256.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and path != OUTPUT
    )
    content = "".join(
        f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in files
    )
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"Wrote {OUTPUT.name} with {len(files)} entries")


if __name__ == "__main__":
    main()

