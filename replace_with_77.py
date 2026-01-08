#!/usr/bin/env python3
"""
replace_with_77.py

Replace a set of existing "sample*" items (folders/files) with items from a source folder,
renaming the source items to match the existing sample names.

Behavior:
- Reads target names from --dst-dir by listing items whose names start with "sample"
- Reads source items from --src by listing non-hidden immediate children
- Requires counts match exactly (source items == target names)
- By default does a dry-run unless --execute is provided
- If --replace is set, existing destination items are moved into a backup folder first
- Supports moving or copying the source items

Usage:
  python3 replace_with_77.py \
    --src "/Users/ayaanb/voicegain_confidence/77" \
    --dst-dir "/path/to/folder/with/sample014_ac..." \
    --mode move \
    --replace \
    --execute

Safe preview:
  python3 replace_with_77.py --src ".../77" --dst-dir "..." --mode move --replace
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


def list_source_items(src: Path) -> List[Path]:
    items: List[Path] = []
    for name in os.listdir(src):
        if name.startswith("."):
            continue
        p = src / name
        items.append(p)
    items.sort(key=lambda p: p.name)
    return items


def list_target_names(dst_dir: Path) -> List[str]:
    names: List[str] = []
    for name in os.listdir(dst_dir):
        if name.startswith("."):
            continue
        if not name.startswith("sample"):
            continue
        names.append(name)
    names.sort()
    seen = set()
    out: List[str] = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def move_or_copy(src: Path, dst: Path, mode: str) -> None:
    if src.is_dir():
        if mode == "copy":
            shutil.copytree(src, dst)
        else:
            shutil.move(str(src), str(dst))
        return
    if src.is_file():
        if mode == "copy":
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))
        return
    raise SystemExit(f"Source item is neither file nor directory: {src}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst-dir", required=True, type=Path)
    ap.add_argument("--mode", choices=["move", "copy"], default="move")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    dst_dir = args.dst_dir.expanduser().resolve()

    if not src.exists() or not src.is_dir():
        raise SystemExit(f"--src must be a directory: {src}")
    if not dst_dir.exists() or not dst_dir.is_dir():
        raise SystemExit(f"--dst-dir must be a directory: {dst_dir}")

    source_items = list_source_items(src)
    target_names = list_target_names(dst_dir)

    if not target_names:
        raise SystemExit(f"No target names found in {dst_dir} (expected items starting with 'sample').")

    if len(source_items) != len(target_names):
        raise SystemExit(
            f"Count mismatch:\n"
            f"  source items in {src}: {len(source_items)}\n"
            f"  target names in {dst_dir}: {len(target_names)}\n"
            f"Make them equal before running."
        )

    if args.shuffle:
        import random
        rng = random.Random(args.seed)
        rng.shuffle(source_items)

    plan: List[Tuple[Path, Path]] = []
    for src_item, tgt_name in zip(source_items, target_names):
        ext = src_item.suffix if src_item.is_file() else ""
        dst_path = dst_dir / f"{tgt_name}{ext}"
        plan.append((src_item, dst_path))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = dst_dir / f"_backup_before_replace_{stamp}"

    mapping_path = dst_dir / f"_replace_mapping_{stamp}.tsv"
    lines = ["src\tassigned_dst"]
    for s, d in plan:
        lines.append(f"{s}\t{d}")
    mapping_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"source: {src}")
    print(f"dest:   {dst_dir}")
    print(f"mode:   {args.mode}")
    print(f"items:  {len(plan)}")
    print(f"plan:   {mapping_path}")

    if not args.execute:
        print("dry-run only (add --execute to perform changes)")
        return 0

    if args.replace:
        backup_dir.mkdir(parents=True, exist_ok=True)
        for name in target_names:
            p = dst_dir / name
            if p.exists():
                shutil.move(str(p), str(backup_dir / name))
        print(f"moved existing targets into: {backup_dir}")
    else:
        for _, d in plan:
            if d.exists():
                raise SystemExit(f"Destination already exists (use --replace): {d}")

    for s, d in plan:
        if d.exists():
            raise SystemExit(f"Refusing to overwrite existing destination: {d}")
        move_or_copy(s, d, args.mode)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
