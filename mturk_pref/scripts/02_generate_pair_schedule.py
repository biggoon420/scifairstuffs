"""
Generates a global schedule of file pairs for pairwise comparisons.

Output CSV:
pair_index,a_file,b_file

Design:
- Starts with a ring to guarantee the graph is connected across all 250 files.
- Then fills remaining pairs by favoring low-appearance files and low-repeat pairs.

If you pay $1 for 15 comparisons, pick total_comparisons divisible by 15
(e.g., 2010 for 134 workers).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_file_ids(manifest_path: Path) -> List[str]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = data.get("files", {})
    ids = sorted(files.keys())
    if len(ids) < 2:
        raise SystemExit("Manifest must contain at least 2 file_ids.")
    return ids


def gen_pairs(file_ids: List[str], total: int, seed: int) -> List[Tuple[str, str]]:
    rnd = random.Random(seed)
    n = len(file_ids)

    appear = defaultdict(int)
    pair_use = defaultdict(int)

    pairs: List[Tuple[str, str]] = []

    ring = file_ids[:]
    rnd.shuffle(ring)
    for i in range(n):
        a = ring[i]
        b = ring[(i + 1) % n]
        if a == b:
            continue
        if rnd.random() < 0.5:
            a, b = b, a
        pairs.append((a, b))
        appear[a] += 1
        appear[b] += 1
        key = tuple(sorted((a, b)))
        pair_use[key] += 1

    def pick_a() -> str:
        m = min(appear[f] for f in file_ids)
        cands = [f for f in file_ids if appear[f] == m]
        return rnd.choice(cands)

    def pick_b(a: str) -> str:
        cands = [f for f in file_ids if f != a]
        rnd.shuffle(cands)

        best = None
        best_score = None
        for b in cands:
            key = tuple(sorted((a, b)))
            score = (pair_use[key] * 100000) + (appear[b] * 10) + rnd.random()
            if best_score is None or score < best_score:
                best_score = score
                best = b
        return best

    while len(pairs) < total:
        a = pick_a()
        b = pick_b(a)
        if rnd.random() < 0.5:
            a, b = b, a
        pairs.append((a, b))
        appear[a] += 1
        appear[b] += 1
        key = tuple(sorted((a, b)))
        pair_use[key] += 1

    return pairs[:total]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    file_ids = load_file_ids(Path(args.manifest))
    pairs = gen_pairs(file_ids, args.total, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_index", "a_file", "b_file"])
        for i, (a, b) in enumerate(pairs):
            w.writerow([i, a, b])

    print(f"Wrote schedule: {out_path} (pairs={len(pairs)}, files={len(file_ids)})")


if __name__ == "__main__":
    main()

