"""
Builds mturk_pref/manifest.json from a local chunks/ directory.

Expected local structure:
chunks/
  <file_id>/
    chunk_01_of_06.mp3
    chunk_02_of_06.mp3
    ...

The manifest maps each file_id -> total chunks + URL per chunk.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any

CHUNK_RE = re.compile(r"chunk_(\d+)_of_(\d+)\.mp3$", re.IGNORECASE)


def build_manifest(chunks_root: Path, base_url: str) -> Dict[str, Any]:
    files: Dict[str, Any] = {}

    if not chunks_root.exists() or not chunks_root.is_dir():
        raise SystemExit(f"chunks_root not found: {chunks_root}")

    base_url = base_url.rstrip("/")

    for file_dir in sorted(chunks_root.iterdir()):
        if not file_dir.is_dir():
            continue
        file_id = file_dir.name
        found = []
        for p in sorted(file_dir.iterdir()):
            if not p.is_file():
                continue
            m = CHUNK_RE.match(p.name)
            if not m:
                continue
            idx = int(m.group(1))
            total = int(m.group(2))
            found.append((idx, total, p.name))

        if not found:
            continue

        totals = {t for _, t, _ in found}
        if len(totals) != 1:
            raise SystemExit(f"Inconsistent totals in {file_dir}: {sorted(totals)}")
        total = next(iter(totals))

        idxs = sorted(i for i, _, _ in found)
        if idxs != list(range(1, total + 1)):
            raise SystemExit(f"Missing chunk indices in {file_dir}. Have {idxs}, expected 1..{total}")

        chunks = {}
        for idx, _, name in found:
            chunks[str(idx)] = f"{base_url}/{file_id}/{name}"

        files[file_id] = {"total": total, "chunks": chunks}

    if len(files) < 2:
        raise SystemExit("Need at least 2 file_ids with chunks to proceed.")

    return {"files": files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-root", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = build_manifest(Path(args.chunks_root), args.base_url)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {out_path} (files={len(manifest['files'])})")


if __name__ == "__main__":
    main()

