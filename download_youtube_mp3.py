#!/usr/bin/env python3
"""
Download YouTube URLs listed in a text file and extract audio as MP3 via yt-dlp + ffmpeg.

Usage:
  python3 download_youtube_mp3.py urls.txt mp3_out
  python3 download_youtube_mp3.py urls.txt mp3_out --min-seconds 120 --max-seconds 300

Requires:
  yt-dlp on PATH
  ffmpeg on PATH
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _require(cmd: str) -> None:
    if shutil.which(cmd) is None:
        raise SystemExit(f"Missing dependency: '{cmd}' not found on PATH.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("urls_file", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--min-seconds", type=int, default=120)
    p.add_argument("--max-seconds", type=int, default=300)
    p.add_argument("--concurrent-fragments", type=int, default=4)
    args = p.parse_args()

    _require("yt-dlp")
    _require("ffmpeg")

    if not args.urls_file.exists():
        raise SystemExit(f"urls_file not found: {args.urls_file}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    match_filter = f"duration >= {args.min_seconds} & duration <= {args.max_seconds}"
    output_template = str(args.out_dir / "%(title).200s [%(id)s].%(ext)s")

    cmd = [
        "yt-dlp",
        "-a",
        str(args.urls_file),
        "--no-playlist",
        "--match-filter",
        match_filter,
        "-N",
        str(args.concurrent_fragments),
        "-x",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "-o",
        output_template,
    ]

    proc = subprocess.run(cmd)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()

