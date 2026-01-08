"""
Download all session CSVs from the MTurk Pref FastAPI app and re-save them locally as:
  PREF_DATAS/CSV_1.csv, PREF_DATAS/CSV_2.csv, ...

It uses the admin endpoint:
  GET /api/sessions?token=...&limit=...

Then downloads each CSV via:
  GET /api/download?name=...&token=...

It also writes:
  PREF_DATAS/index.csv
mapping new_name -> original server filename + mtime/size.

Usage:
  python mturk_pref/scripts/05_pull_all_sessions.py \
    --base "https://scifairstuffs.onrender.com" \
    --token "YOUR_ADMIN_TOKEN" \
    --outdir "PREF_DATAS" \
    --prefix "CSV" \
    --newest-first
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _http_get_json(url: str, timeout: int) -> Dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return json.loads(raw.decode("utf-8", errors="replace"))


def _http_download(url: str, dst: Path, timeout: int) -> None:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)


def _join(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--outdir", default="PREF_DATAS")
    ap.add_argument("--prefix", default="CSV")
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--limit", type=int, default=200000)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--newest-first", action="store_true")
    args = ap.parse_args()

    base = args.base.strip().rstrip("/")
    token = args.token.strip()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sessions_url = _join(base, "api/sessions") + "?" + urllib.parse.urlencode(
        {"token": token, "limit": str(int(args.limit))}
    )
    j = _http_get_json(sessions_url, timeout=int(args.timeout))

    if not isinstance(j, dict) or not j.get("ok"):
        raise SystemExit(f"/api/sessions failed: {str(j)[:400]}")

    files = j.get("files", [])
    if not isinstance(files, list) or not files:
        print("No CSVs found on server.")
        return

    def key_fn(x: Dict[str, Any]) -> Tuple[float, str]:
        return (float(x.get("mtime", 0.0)), str(x.get("name", "")))

    files_sorted = sorted(files, key=key_fn, reverse=bool(args.newest_first))

    index_path = outdir / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "new_name", "original_name", "size_bytes", "mtime_epoch"])

        n = len(files_sorted)
        t0 = time.time()
        ok = 0

        for i, meta in enumerate(files_sorted, start=1):
            original_name = str(meta.get("name", "")).strip()
            if not original_name:
                continue

            if args.pad and int(args.pad) > 0:
                new_name = f"{args.prefix}_{i:0{int(args.pad)}d}.csv"
            else:
                new_name = f"{args.prefix}_{i}.csv"

            dst = outdir / new_name

            dl_url = _join(base, "api/download") + "?" + urllib.parse.urlencode(
                {"name": original_name, "token": token}
            )

            _http_download(dl_url, dst, timeout=int(args.timeout))

            size_b = int(meta.get("size", 0) or 0)
            mtime = float(meta.get("mtime", 0.0) or 0.0)

            w.writerow([i, new_name, original_name, size_b, mtime])
            ok += 1

            if i % 25 == 0 or i == n:
                dt = max(0.001, time.time() - t0)
                rate = ok / dt
                print(f"Downloaded {ok}/{n} -> {outdir.as_posix()} ({rate:.2f} files/sec)")

    print(f"Done. CSVs in: {outdir.as_posix()}")
    print(f"Index file: {index_path.as_posix()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
