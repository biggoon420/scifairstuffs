"""
Audio Preference Trainer (Chunk-first, Session-based)

Workflow
1) Choose a source folder (and optionally create a random subset inside it).
2) Start Training -> prompts for session name + number of comparisons.
3) App ensures 15s VAD-based chunk MP3s exist (min 3, max 12 per source file).
4) Labels are collected as A/B chunk comparisons (never same file vs itself).
5) Skip replaces ONLY the chunk on that side (same file), logs winner=0, and advances progress.

Session CSV format (saved live):
a_file,a_chunk_idx,a_chunk_total,b_file,b_chunk_idx,b_chunk_total,winner,replacement

winner: 1 (A wins), 2 (B wins), 0 (skipped)
replacement: "0" normally; "A<k>" if A-side chunk was replaced to chunk k; "B<k>" if B-side chunk replaced to chunk k

sessions_meta.csv (updated on every session finish):
session_name,comparisons,alignment
"""

import os
import re
import csv
import math
import time
import json
import random
import hashlib
import threading
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import simpledialog
from tkinter import font as tkfont

import torch
from pydub import AudioSegment
from PIL import Image, ImageTk, ImageSequence


sd.default.device = None
sd.default.reset()

vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False
)

(get_speech_timestamps,
 read_audio,
 save_audio,
 VADIterator,
 collect_chunks) = vad_utils


AUDIO_EXT = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg")
CHUNK_SR = 16000
CHUNK_SEC = 15
CHUNK_MS = CHUNK_SEC * 1000
MIN_CHUNKS = 3
MAX_CHUNKS = 12
CHUNKS_DIR = "chunks"
PREF_DIR = os.path.join("preferences", "sessions")
SESSIONS_META = os.path.join("preferences", "sessions_meta.csv")
QUEUE_DEBUG_TXT = os.path.join("preferences", "queue_debug.txt")
CHUNK_NAME_RE = re.compile(r"chunk_(\d+)_of_(\d+)\.mp3$", re.IGNORECASE)
PAIR_QUEUE = []        # list[Pair]
QUEUE_VISIBLE = False  # debug toggle
QUEUE_SIZE = 5         # how many upcoming pairs to precompute




def get_macos_output_volume() -> float:
    try:
        out = subprocess.check_output(["osascript", "-e", "output volume of (get volume settings)"])
        return int(out.strip()) / 100.0
    except Exception:
        return 1.0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_audio_files(folder: str) -> List[str]:
    out = []
    try:
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            if os.path.isfile(p) and f.lower().endswith(AUDIO_EXT):
                out.append(p)
    except Exception:
        return []
    out.sort()
    return out
    
def fill_pair_queue(stats, chunks_root, files, queue, target_size):
    while len(queue) < target_size:
        queue.append(select_next_pair(stats, chunks_root, files))


def short_hash_file(path: str, n: int = 8) -> str:
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            head = f.read(65536)
            if head:
                h.update(head)
            try:
                f.seek(-65536, os.SEEK_END)
                tail = f.read(65536)
                if tail:
                    h.update(tail)
            except Exception:
                pass
    except Exception:
        h.update(path.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:n]


def file_id_for_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem).strip("_")
    return f"{stem}_{short_hash_file(path, 8)}"


def num_chunks_for_duration(duration_sec: float) -> int:
    d = max(1.0, float(duration_sec))
    if d <= 120:
        n = 3
    elif d <= 360:
        n = 3 + int((d - 120) / 80)
    elif d <= 900:
        n = 6 + int((d - 360) / 110)
    else:
        n = 10 + int((d - 900) / 300)
    return max(MIN_CHUNKS, min(MAX_CHUNKS, n))


def audiosegment_to_float32_mono(seg: AudioSegment, sr: int) -> np.ndarray:
    seg = seg.set_channels(1).set_frame_rate(sr)
    arr = np.array(seg.get_array_of_samples())
    if seg.sample_width == 1:
        div = 128.0
    elif seg.sample_width == 2:
        div = 32768.0
    elif seg.sample_width == 4:
        div = 2147483648.0
    else:
        div = float(1 << (8 * seg.sample_width - 1))
    y = arr.astype(np.float32) / div
    if y.ndim > 1:
        y = y.reshape(-1)
    return y


def _candidate_windows(samples: np.ndarray, sr: int, speech_ts: List[Dict[str, int]]) -> List[Tuple[float, int]]:
    n = int(samples.shape[0])
    win = int(CHUNK_SEC * sr)
    step = max(int(2.5 * sr), 1)
    if n <= win:
        return [(1.0, 0)]
    ts = [(int(t["start"]), int(t["end"])) for t in speech_ts] if speech_ts else []
    out = []
    for s0 in range(0, n - win + 1, step):
        s1 = s0 + win
        voiced = 0
        for a, b in ts:
            if b <= s0:
                continue
            if a >= s1:
                break
            inter = min(b, s1) - max(a, s0)
            if inter > 0:
                voiced += inter
        voiced_sec = voiced / float(sr)
        chunk = samples[s0:s1]
        energy = float(np.mean(np.abs(chunk))) if chunk.size else 0.0
        min_voiced_sec = 6.0  # require at least 6s of detected speech in the 15s window
        if voiced_sec < min_voiced_sec:
            continue

        score = voiced_sec + 0.20 * energy * CHUNK_SEC
        out.append((score, s0))

    out.sort(key=lambda x: x[0], reverse=True)
    return out


def _greedy_nonoverlap(starts: List[int], sr: int, k: int) -> List[int]:
    chosen = []
    min_sep = int(0.55 * CHUNK_SEC * sr)
    for s in starts:
        ok = True
        for c in chosen:
            if abs(s - c) < min_sep:
                ok = False
                break
        if ok:
            chosen.append(s)
            if len(chosen) >= k:
                break
    return chosen


def _fill_evenly(n_samples: int, sr: int, already: List[int], k: int) -> List[int]:
    win = int(CHUNK_SEC * sr)
    if n_samples <= win:
        return already + [0] * (k - len(already))
    span = n_samples - win
    if k <= 1:
        targets = [0]
    else:
        targets = [int(round(span * i / (k - 1))) for i in range(k)]
    used = set(already)
    for t in targets:
        if len(already) >= k:
            break
        if t not in used:
            already.append(t)
            used.add(t)
    while len(already) < k:
        t = random.randint(0, span)
        if t not in used:
            already.append(t)
            used.add(t)
    return already


def ensure_chunks_for_file(path: str, chunks_root: str) -> Tuple[str, int]:
    fid = file_id_for_path(path)
    out_dir = os.path.join(chunks_root, fid)
    ensure_dir(out_dir)

    existing = []
    for f in os.listdir(out_dir):
        m = CHUNK_NAME_RE.match(f)
        if m:
            existing.append((int(m.group(1)), int(m.group(2)), os.path.join(out_dir, f)))
    if existing:
        existing.sort(key=lambda x: x[0])
        total = existing[0][1]
        if all(t == total for _, t, _ in existing) and len(existing) == total:
            return fid, total

    try:
        for f in os.listdir(out_dir):
            if f.lower().endswith(".mp3"):
                try:
                    os.remove(os.path.join(out_dir, f))
                except Exception:
                    pass
    except Exception:
        pass

    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(CHUNK_SR)
    duration_sec = float(len(audio)) / 1000.0
    k = num_chunks_for_duration(duration_sec)

    samples = audiosegment_to_float32_mono(audio, CHUNK_SR)
    audio_t = torch.tensor(samples, dtype=torch.float32)

    try:
        speech_ts = get_speech_timestamps(audio_t, vad_model, sampling_rate=CHUNK_SR)
    except Exception:
        speech_ts = []

    cands = _candidate_windows(samples, CHUNK_SR, speech_ts)
    starts_sorted = [s for _, s in cands]
    chosen = _greedy_nonoverlap(starts_sorted, CHUNK_SR, k)
    chosen = _fill_evenly(samples.shape[0], CHUNK_SR, chosen, k)
    chosen = chosen[:k]
    chosen_ms = [int(round(s * 1000.0 / CHUNK_SR)) for s in chosen]

    for i, start_ms in enumerate(chosen_ms, start=1):
        seg = audio[start_ms:start_ms + CHUNK_MS]
        if len(seg) < CHUNK_MS:
            seg = seg + AudioSegment.silent(duration=(CHUNK_MS - len(seg)), frame_rate=CHUNK_SR)
        out_path = os.path.join(out_dir, f"chunk_{i:02d}_of_{k:02d}.mp3")
        seg.export(out_path, format="mp3", bitrate="80k")

    return fid, k


def chunk_paths_for_file_id(chunks_root: str, file_id: str) -> List[Tuple[int, int, str]]:
    out_dir = os.path.join(chunks_root, file_id)
    if not os.path.isdir(out_dir):
        return []
    found = []
    for f in os.listdir(out_dir):
        m = CHUNK_NAME_RE.match(f)
        if m:
            idx = int(m.group(1))
            total = int(m.group(2))
            found.append((idx, total, os.path.join(out_dir, f)))
    found.sort(key=lambda x: x[0])
    return found


def decode_mp3_to_float32(path: str, sr: int = CHUNK_SR) -> Tuple[np.ndarray, int]:
    seg = AudioSegment.from_file(path)
    y = audiosegment_to_float32_mono(seg, sr)
    return y, sr


@dataclass
class Pair:
    a_file: str
    a_chunk_idx: int
    a_chunk_total: int
    a_path: str
    b_file: str
    b_chunk_idx: int
    b_chunk_total: int
    b_path: str


class Stats:
    def __init__(self):
        self.file_weighted_appear: Dict[str, float] = {}
        self.file_decisive: Dict[str, int] = {}
        self.chunk_used: Dict[Tuple[str, int], int] = {}
        self.chunk_skipped: Dict[Tuple[str, int], int] = {}
        self.pair_count: Dict[Tuple[str, str], int] = {}
        self.elo: Dict[str, float] = {}

    def _inc_float(self, d: Dict[str, float], k: str, v: float) -> None:
        d[k] = d.get(k, 0.0) + float(v)

    def _inc_int(self, d: Dict, k, v: int = 1) -> None:
        d[k] = int(d.get(k, 0)) + int(v)

    def ensure_file(self, fid: str) -> None:
        if fid not in self.elo:
            self.elo[fid] = 1000.0

    def update_from_row(self, row: Dict[str, str], session_weight: float) -> None:
        a = row["a_file"]
        b = row["b_file"]
        a_idx = int(row["a_chunk_idx"])
        b_idx = int(row["b_chunk_idx"])
        winner = int(row["winner"])
        repl = row["replacement"]

        self.ensure_file(a)
        self.ensure_file(b)

        self._inc_float(self.file_weighted_appear, a, session_weight)
        self._inc_float(self.file_weighted_appear, b, session_weight)

        self._inc_int(self.chunk_used, (a, a_idx), 1)
        self._inc_int(self.chunk_used, (b, b_idx), 1)

        key = tuple(sorted((a, b)))
        self._inc_int(self.pair_count, key, 1)

        if winner in (1, 2):
            self._inc_int(self.file_decisive, a, 1)
            self._inc_int(self.file_decisive, b, 1)
        if winner == 0 and repl:
            if isinstance(repl, str) and repl.startswith("A"):
                self._inc_int(self.chunk_skipped, (a, a_idx), 1)
            elif isinstance(repl, str) and repl.startswith("B"):
                self._inc_int(self.chunk_skipped, (b, b_idx), 1)

    def elo_update(self, a: str, b: str, winner: int, k_base: float) -> None:
        ra = self.elo.get(a, 1000.0)
        rb = self.elo.get(b, 1000.0)
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea
        sa = 1.0 if winner == 1 else 0.0
        sb = 1.0 - sa
        self.elo[a] = ra + k_base * (sa - ea)
        self.elo[b] = rb + k_base * (sb - eb)


def read_session_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    try:
        with open(path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if not row:
                    continue
                rows.append(row)
    except Exception:
        return []
    return rows


def list_session_files() -> List[str]:
    ensure_dir(PREF_DIR)
    out = []
    try:
        for f in os.listdir(PREF_DIR):
            if f.lower().endswith(".csv"):
                out.append(os.path.join(PREF_DIR, f))
    except Exception:
        return []
    out.sort()
    return out


def load_historical_stats(selected_files: List[str]) -> Stats:
    s = Stats()
    selected = set(selected_files)

    session_files = list_session_files()
    for sp in session_files:
        rows = read_session_csv(sp)
        decisive = 0
        for row in rows:
            try:
                w = int(row["winner"])
            except Exception:
                continue
            if w in (1, 2):
                decisive += 1
        weight = 1.0 / math.sqrt(max(decisive, 1))
        for row in rows:
            try:
                a = row["a_file"]
                b = row["b_file"]
            except Exception:
                continue
            if a not in selected or b not in selected:
                continue
            s.update_from_row(row, weight)

    global_rows = []
    for sp in session_files:
        rows = read_session_csv(sp)
        decisive = 0
        for row in rows:
            try:
                w = int(row["winner"])
            except Exception:
                continue
            if w in (1, 2):
                decisive += 1
        weight = 1.0 / math.sqrt(max(decisive, 1))
        for row in rows:
            try:
                a = row["a_file"]
                b = row["b_file"]
                w = int(row["winner"])
            except Exception:
                continue
            if a not in selected or b not in selected:
                continue
            if w not in (1, 2):
                continue
            global_rows.append((a, b, w, weight))

    random.shuffle(global_rows)
    for a, b, w, weight in global_rows:
        s.ensure_file(a)
        s.ensure_file(b)
        s.elo_update(a, b, w, k_base=32.0 * weight)

    return s


def choose_chunk_for_file(stats: Stats, chunks_root: str, fid: str, avoid_idx: Optional[int] = None) -> Tuple[int, int, str]:
    chunks = chunk_paths_for_file_id(chunks_root, fid)
    if not chunks:
        return 1, 1, ""
    scored = []
    for idx, total, p in chunks:
        if avoid_idx is not None and idx == avoid_idx and total > 1:
            continue
        used = stats.chunk_used.get((fid, idx), 0)
        skipped = stats.chunk_skipped.get((fid, idx), 0)
        score = used + 3 * skipped + random.random() * 0.25
        scored.append((score, idx, total, p))
    scored.sort(key=lambda x: x[0])
    _, idx, total, p = scored[0]
    return idx, total, p


def select_next_pair(stats: Stats, chunks_root: str, files: List[str], force: Optional[Tuple[str, int]] = None) -> Pair:
    target_min = 15.0
    target_ideal = 30.0

    def need(fid: str) -> float:
        a = stats.file_weighted_appear.get(fid, 0.0)
        cov = 2.0 * max(0.0, target_min - a) + 0.6 * max(0.0, target_ideal - a)
        unc = 0.8 / float(1 + stats.file_decisive.get(fid, 0))
        return cov + unc + random.random() * 0.10

    if force is not None:
        a_file, a_idx_fixed = force
        a_total, a_path = 1, ""
        chunks = chunk_paths_for_file_id(chunks_root, a_file)
        for idx, total, p in chunks:
            if idx == a_idx_fixed:
                a_total, a_path = total, p
                break
        if not a_path:
            a_idx, a_total, a_path = choose_chunk_for_file(stats, chunks_root, a_file, avoid_idx=None)
        else:
            a_idx = a_idx_fixed
    else:
        files_sorted = sorted(files, key=lambda f: need(f), reverse=True)
        top = files_sorted[: max(5, min(30, len(files_sorted)))]
        weights = [max(1e-6, need(f)) for f in top]
        a_file = random.choices(top, weights=weights, k=1)[0]
        a_idx, a_total, a_path = choose_chunk_for_file(stats, chunks_root, a_file, avoid_idx=None)

    a_rating = stats.elo.get(a_file, 1000.0)

    candidates = [f for f in files if f != a_file]
    random.shuffle(candidates)
    candidates = candidates[: max(20, min(120, len(candidates)))]

    best = None
    best_score = -1e9

    for b_file in candidates:
        b_rating = stats.elo.get(b_file, 1000.0)
        diff = abs(b_rating - a_rating)
        sim = math.exp(-diff / 200.0)
        cov = need(b_file)
        pair_key = tuple(sorted((a_file, b_file)))
        rep = stats.pair_count.get(pair_key, 0)
        rep_pen = 0.25 * float(rep)
        score = 1.2 * sim + 0.6 * cov - rep_pen + random.random() * 0.05
        if score > best_score:
            best_score = score
            best = b_file

    if best is None:
        best = random.choice([f for f in files if f != a_file])

    b_file = best
    b_idx, b_total, b_path = choose_chunk_for_file(stats, chunks_root, b_file, avoid_idx=None)

    return Pair(
        a_file=a_file, a_chunk_idx=a_idx, a_chunk_total=a_total, a_path=a_path,
        b_file=b_file, b_chunk_idx=b_idx, b_chunk_total=b_total, b_path=b_path
    )


def update_sessions_meta() -> None:
    ensure_dir(os.path.dirname(SESSIONS_META))
    session_files = list_session_files()

    all_rows_by_session: Dict[str, List[Dict[str, str]]] = {}
    decisive_by_session: Dict[str, int] = {}
    file_set = set()

    for sp in session_files:
        name = os.path.splitext(os.path.basename(sp))[0]
        rows = read_session_csv(sp)
        all_rows_by_session[name] = rows
        decisive = 0
        for r in rows:
            try:
                w = int(r.get("winner", "0"))
            except Exception:
                continue
            if w in (1, 2):
                decisive += 1
            a = r.get("a_file", "")
            b = r.get("b_file", "")
            if a:
                file_set.add(a)
            if b:
                file_set.add(b)
        decisive_by_session[name] = decisive

    global_elo: Dict[str, float] = {f: 1000.0 for f in file_set}
    all_updates: List[Tuple[str, str, int, float]] = []
    for name, rows in all_rows_by_session.items():
        w_sess = 1.0 / math.sqrt(max(decisive_by_session.get(name, 0), 1))
        for r in rows:
            try:
                a = r["a_file"]
                b = r["b_file"]
                w = int(r["winner"])
            except Exception:
                continue
            if w not in (1, 2):
                continue
            all_updates.append((a, b, w, w_sess))

    random.shuffle(all_updates)
    for a, b, w, w_sess in all_updates:
        ra = global_elo.get(a, 1000.0)
        rb = global_elo.get(b, 1000.0)
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea
        sa = 1.0 if w == 1 else 0.0
        sb = 1.0 - sa
        k = 32.0 * w_sess
        global_elo[a] = ra + k * (sa - ea)
        global_elo[b] = rb + k * (sb - eb)

    lines = []
    lines.append(["session_name", "comparisons", "alignment"])
    for name, rows in sorted(all_rows_by_session.items(), key=lambda x: x[0].lower()):
        comparisons = len(rows)
        decisive_rows = [r for r in rows if str(r.get("winner", "")).strip() in ("1", "2")]
        if not decisive_rows:
            alignment = 0.0
        else:
            agree = 0.0
            for r in decisive_rows:
                a = r.get("a_file", "")
                b = r.get("b_file", "")
                try:
                    w = int(r.get("winner", "0"))
                except Exception:
                    continue
                ga = global_elo.get(a, 1000.0)
                gb = global_elo.get(b, 1000.0)
                if abs(ga - gb) < 1e-9:
                    agree += 0.5
                else:
                    global_pref = 1 if ga > gb else 2
                    if w == global_pref:
                        agree += 1.0
            alignment = float(agree) / float(len(decisive_rows))
        lines.append([name, str(comparisons), f"{alignment:.6f}"])

    with open(SESSIONS_META, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(lines)


class RoundedButton(tk.Canvas):
    def __init__(
        self, parent, text, command=None,
        radius=10, padding_x=20, padding_y=10,
        bg_color="#E36D5A", fg_color="black",
        font=("Arial", 12, "bold"),
        border_color="#F3D4CF",
        border_width=4
    ):
        self.text = text
        self.command = command
        self.radius = radius
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.font = font
        self.border_color = border_color
        self.border_width = border_width

        fnt = tkfont.Font(font=font)
        text_w = fnt.measure(text)
        text_h = fnt.metrics("linespace")

        width = text_w + 2 * padding_x
        height = text_h + 2 * padding_y

        super().__init__(
            parent,
            width=width,
            height=height,
            highlightthickness=0,
            bd=0
        )

        self._draw_button()
        self.bind("<Button-1>", lambda e: self.command() if self.command else None)
        self.bind("<Enter>", lambda e: self._hover(True))
        self.bind("<Leave>", lambda e: self._hover(False))

    def _round_rect(self, x1, y1, x2, y2, r, **kwargs):
        pts = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1, x2, y1 + r,
            x2, y2 - r,
            x2, y2, x2 - r, y2,
            x1 + r, y2,
            x1, y2, x1, y2 - r,
            x1, y1 + r,
            x1, y1
        ]
        return self.create_polygon(pts, smooth=True, **kwargs)

    def _draw_button(self):
        w = int(self.cget("width"))
        h = int(self.cget("height"))
        r = self.radius

        self.rect = self._round_rect(
            1, 1, w - 1, h - 1, r,
            fill=self.bg_color,
            outline=self.border_color,
            width=self.border_width
        )
        self.txt = self.create_text(
            w // 2, h // 2,
            text=self.text,
            fill=self.fg_color,
            font=self.font
        )

    def _hover(self, state):
        if state:
            self.itemconfigure(self.rect, fill=self._lighten(self.bg_color, 1.08))
        else:
            self.itemconfigure(self.rect, fill=self.bg_color)

    def _lighten(self, color, factor):
        c = color.lstrip("#")
        r = int(c[0:2], 16)
        g = int(c[2:4], 16)
        b = int(c[4:6], 16)
        r = min(int(r * factor), 255)
        g = min(int(g * factor), 255)
        b = min(int(b * factor), 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    def set_fill(self, color):
        self.bg_color = color
        self.itemconfigure(self.rect, fill=color)


class PreferenceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Audio Preference Trainer")

        self.COL_SALMON = "#EA8C7B"
        self.COL_PEACH = "#F6B29A"
        self.COL_BLUSH = "#F3D4CF"
        self.COL_PERI = "#9DA8C8"
        self.COL_DARK = "#2F2F2F"

        self.source_folder: Optional[str] = None
        self.folder: Optional[str] = None
        self.chunks_root = CHUNKS_DIR

        self.session_name: Optional[str] = None
        self.session_target: int = 0
        self.session_done: int = 0
        self.session_fp: Optional[str] = None
        self.session_fh = None
        self.session_writer = None

        self.training_files: List[str] = []
        self.stats: Optional[Stats] = None
        self.current_pair: Optional[Pair] = None

        self.base_volume = 0.6
        self.playing = False
        self.play_start = None
        self.play_duration = None

        self.audio_cache: Dict[str, Tuple[np.ndarray, int]] = {}

        self.debug_open = True
        self.debug_log: List[str] = []

        try:
            self.root.state("zoomed")
        except Exception:
            self.root.geometry("1200x800")

        self.canvas = tk.Canvas(root, highlightthickness=0, bd=0, bg=self.COL_BLUSH)
        self.canvas.pack(fill="both", expand=True)

        self.text_items = {}
        self._create_text_box("title", "Audio Preference Trainer", ("Arial", 36, "bold"), self.COL_DARK)
        self._create_text_box("subtitle", "Pairwise 15-second VAD chunk comparisons", ("Arial", 16), self.COL_DARK)
        self._create_text_box("folder", "No source folder selected", ("Arial", 14), self.COL_DARK)
        self._create_text_box("session", "Session: (none)", ("Arial", 14, "bold"), self.COL_DARK)
        self._create_text_box("pair", "Pair: (none)", ("Arial", 14), self.COL_DARK)
        self._create_text_box("progress", "Progress: 0/0", ("Arial", 16, "bold"), self.COL_DARK)

        self.num_entry = tk.Entry(root, width=6, justify="center", font=("Arial", 12))

        self.choose_btn = RoundedButton(
            root, "Choose Folder",
            command=self._wrap(self.choose_source_folder),
            radius=10, bg_color=self.COL_PERI, fg_color="white",
            font=("Arial", 12, "bold"), border_color=self.COL_BLUSH, border_width=4
        )

        self.make_subset_btn = RoundedButton(
            root, "Create Subset",
            command=self._wrap(self.create_subset),
            radius=10, bg_color=self.COL_PERI, fg_color="white",
            font=("Arial", 12, "bold"), border_color=self.COL_BLUSH, border_width=4
        )

        self.start_btn = RoundedButton(
            root, "Start Training",
            command=self._wrap(self.start_training),
            radius=10, padding_x=40, padding_y=12,
            bg_color=self.COL_SALMON, fg_color="black",
            font=("Arial", 18, "bold"), border_color=self.COL_BLUSH, border_width=4
        )

        self.playA = RoundedButton(
            root, "▶ Play A",
            command=self._wrap(lambda: self.play_side(0)),
            radius=10, bg_color=self.COL_PERI, fg_color="white",
            font=("Arial", 14, "bold"), border_color=self.COL_BLUSH, border_width=4
        )

        self.playB = RoundedButton(
            root, "▶ Play B",
            command=self._wrap(lambda: self.play_side(1)),
            radius=10, bg_color=self.COL_PERI, fg_color="white",
            font=("Arial", 14, "bold"), border_color=self.COL_BLUSH, border_width=4
        )

        self.skipA = RoundedButton(
            root, "Skip A Chunk",
            command=self._wrap(lambda: self.skip_chunk(0)),
            radius=10, bg_color=self.COL_PEACH, fg_color="black",
            font=("Arial", 12, "bold"), border_color=self.COL_BLUSH, border_width=4
        )

        self.skipB = RoundedButton(
            root, "Skip B Chunk",
            command=self._wrap(lambda: self.skip_chunk(1)),
            radius=10, bg_color=self.COL_PEACH, fg_color="black",
            font=("Arial", 12, "bold"), border_color=self.COL_BLUSH, border_width=4
        )

        self.voteA = RoundedButton(
            root, "A IS BETTER",
            command=self._wrap(lambda: self.vote(1)),
            radius=10, padding_x=50, padding_y=15,
            bg_color=self.COL_SALMON, fg_color="black",
            font=("Arial", 20, "bold"), border_color=self.COL_BLUSH, border_width=4
        )

        self.voteB = RoundedButton(
            root, "B IS BETTER",
            command=self._wrap(lambda: self.vote(2)),
            radius=10, padding_x=50, padding_y=15,
            bg_color=self.COL_SALMON, fg_color="black",
            font=("Arial", 20, "bold"), border_color=self.COL_BLUSH, border_width=4
        )

        self.waveform_canvas = tk.Canvas(root, height=120, bg=self.COL_PERI, highlightthickness=0, bd=0)

        self.debug_panel = tk.Frame(root, bg="#1b1b29")
        self.debug_label = tk.Label(
            self.debug_panel, text="",
            fg="#A8FFB0", bg="#1b1b29",
            font=("Consolas", 11),
            justify="left"
        )
        self.debug_label.pack(padx=10, pady=5)
        self.debug_panel_window = None
        self.win_items = {}

        self._relayout()
        self.root.bind("<Configure>", lambda e: self._relayout())
        self.root.bind("1", lambda e: self.vote(1))
        self.root.bind("2", lambda e: self.vote(2))
        self.root.bind("q", lambda e: self.play_side(0))
        self.root.bind("w", lambda e: self.play_side(1))
        self.root.bind("a", lambda e: self.skip_chunk(0))
        self.root.bind("s", lambda e: self.skip_chunk(1))
        self.root.bind("d", lambda e: self.toggle_queue_debug())
        self.root.bind("<Command-Shift-f>", lambda e: self.finish_early())

        self.root.after(40, self._update_playhead)
        self.queue_used = 0


        self.debug("Keys: 1=A wins, 2=B wins, q=play A, w=play B, a=skip A chunk, s=skip B chunk, d=view debug ⌘⇧F=finish")

    def toggle_queue_debug(self):
        global QUEUE_VISIBLE
        QUEUE_VISIBLE = not QUEUE_VISIBLE

        if not QUEUE_VISIBLE:
            self.debug("Queue hidden")
            return

        lines = [
            "--- PAIR QUEUE ---",
            f"Consumed: {self.queue_used}",
            f"In queue: {len(PAIR_QUEUE)}",
            ""
        ]

        for i, p in enumerate(PAIR_QUEUE):
            prefix = ">> NEXT <<" if i == 0 else f"{i+1:02d}"
            lines.append(
                f"{prefix}: "
                f"A={p.a_file} ({p.a_chunk_idx}/{p.a_chunk_total}) "
                f"vs "
                f"B={p.b_file} ({p.b_chunk_idx}/{p.b_chunk_total})"
            )

        self.debug("\n".join(lines))




    def _wrap(self, f):
        def inner(*args, **kwargs):
            self.stop_audio()
            return f(*args, **kwargs)
        return inner

    def debug(self, msg: str) -> None:
        entry = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.debug_log.append(entry)
        self.debug_log = self.debug_log[-18:]
        self.debug_label.config(text="\n".join(self.debug_log))
    

    def _create_text_box(self, name, text, font, fill):
        text_id = self.canvas.create_text(0, 0, text=text, font=font, fill=fill, anchor="n")
        rect_id = self.canvas.create_polygon(0, 0, 0, 0, fill=self.COL_BLUSH, outline=self.COL_BLUSH, width=4)
        self.canvas.tag_lower(rect_id, text_id)
        self.text_items[name] = {"text": text_id, "rect": rect_id}

    def _round_rect(self, x1, y1, x2, y2, r, **kwargs):
        pts = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1, x2, y1 + r,
            x2, y2 - r,
            x2, y2, x2 - r, y2,
            x1 + r, y2,
            x1, y2, x1, y2 - r,
            x1, y1 + r,
            x1, y1
        ]
        return self.canvas.create_polygon(pts, smooth=True, **kwargs)

    def _move_text_box(self, name, x, y, pad_x=16, pad_y=8, radius=14):
        item = self.text_items[name]
        text_id = item["text"]
        rect_id = item["rect"]
        self.canvas.coords(text_id, x, y)
        bbox = self.canvas.bbox(text_id)
        if not bbox:
            return
        x1, y1, x2, y2 = bbox
        x1 -= pad_x
        x2 += pad_x
        y1 -= pad_y
        y2 += pad_y
        self.canvas.delete(rect_id)
        rect_id_new = self._round_rect(x1, y1, x2, y2, radius, fill=self.COL_BLUSH, outline=self.COL_BLUSH, width=4)
        self.canvas.tag_lower(rect_id_new, text_id)
        item["rect"] = rect_id_new

    def _set_text(self, name, text):
        item = self.text_items[name]
        self.canvas.itemconfigure(item["text"], text=text)

    def _place_widget(self, name, widget, x, y, anchor="n"):
        if name in self.win_items:
            self.canvas.coords(self.win_items[name], x, y)
        else:
            self.win_items[name] = self.canvas.create_window(x, y, window=widget, anchor=anchor)

    def _relayout(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            return
        cx = w // 2
        y = 40

        self._move_text_box("title", cx, y)
        y += 60
        self._move_text_box("subtitle", cx, y)
        y += 50

        self._move_text_box("folder", cx - 22, y)
        self._place_widget("choose_btn", self.choose_btn, cx + 220, y - 8, anchor="n")
        y += 50

        self._move_text_box("session", cx, y)
        y += 45

        self._place_widget("num_entry", self.num_entry, cx - 10, y - 8, anchor="n")
        self._place_widget("make_subset_btn", self.make_subset_btn, cx + 140, y - 8, anchor="n")
        y += 60

        self._place_widget("start_btn", self.start_btn, cx, y, anchor="n")
        y += 85

        self._move_text_box("pair", cx, y)
        y += 55

        self._place_widget("playA", self.playA, cx - 220, y, anchor="n")
        self._place_widget("playB", self.playB, cx + 220, y, anchor="n")
        y += 60

        self._place_widget("skipA", self.skipA, cx - 220, y, anchor="n")
        self._place_widget("skipB", self.skipB, cx + 220, y, anchor="n")
        y += 85

        self._place_widget("voteA", self.voteA, cx - 220, y, anchor="n")
        self._place_widget("voteB", self.voteB, cx + 220, y, anchor="n")
        y += 115

        self._move_text_box("progress", cx, y)
        y += 40

        wf_width = int(w * 0.9)
        self.waveform_canvas.config(width=wf_width)
        self._place_widget("waveform", self.waveform_canvas, cx, y + 80, anchor="center")

        if self.debug_panel_window is None:
            self.debug_panel_window = self.canvas.create_window(cx, h - 160, window=self.debug_panel, anchor="n")
        else:
            self.canvas.coords(self.debug_panel_window, cx, h - 160)

    def reset_audio_device(self):
        try:
            sd.stop()
            sd.default.device = None
        except Exception:
            pass

    def stop_audio(self):
        try:
            sd.stop()
        except Exception:
            pass
        self.playing = False
        self.play_start = None
        self.play_duration = None

    def draw_waveform(self, clip: np.ndarray):
        self.waveform_canvas.delete("all")
        w = self.waveform_canvas.winfo_width()
        h = self.waveform_canvas.winfo_height()
        if not w or not h or clip is None or clip.size == 0:
            return
        mid = h // 2
        amp = h * 0.4
        num_pts = min(w, int(clip.size))
        idx = np.linspace(0, clip.size - 1, num_pts).astype(int)
        vals = clip[idx]
        m = float(np.max(np.abs(vals))) if vals.size else 0.0
        if m > 0:
            vals = vals / m
        coords = []
        for i, v in enumerate(vals):
            coords.extend([i, mid - v * amp])
        self.waveform_canvas.create_line(coords, fill="white", width=2)

    def _update_playhead(self):
        self.root.after(40, self._update_playhead)

    def choose_source_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.source_folder = folder
            self.folder = folder
            self._set_text("folder", f"Source: {folder}")
            self.debug(f"Selected folder: {folder}")

    def create_subset(self):
        if not self.source_folder:
            messagebox.showerror("Error", "Pick a folder first.")
            return
        try:
            n = int(self.num_entry.get())
        except Exception:
            messagebox.showerror("Error", "Invalid number.")
            return
        files = list_audio_files(self.source_folder)
        if n < 2:
            messagebox.showerror("Error", "N must be ≥ 2.")
            return
        if n > len(files):
            messagebox.showerror("Error", f"Only {len(files)} available.")
            return
        subset = os.path.join(self.source_folder, "training_subset")
        if os.path.exists(subset):
            try:
                for f in os.listdir(subset):
                    try:
                        os.remove(os.path.join(subset, f))
                    except Exception:
                        pass
            except Exception:
                pass
        ensure_dir(subset)
        chosen = random.sample(files, n)
        for f in chosen:
            dst = os.path.join(subset, os.path.basename(f))
            try:
                with open(f, "rb") as rf, open(dst, "wb") as wf:
                    wf.write(rf.read())
            except Exception:
                pass
        self.folder = subset
        self._set_text("folder", f"Subset: {subset}")
        self.debug(f"Created subset: {n} files")

    def _prompt_session(self) -> Optional[Tuple[str, int]]:
        name = simpledialog.askstring("Session", "Session name (any text):", parent=self.root)
        if not name:
            return None
        name = name.strip()
        if not name:
            return None
        num = simpledialog.askinteger("Session", "How many comparisons?", parent=self.root, minvalue=1, maxvalue=100000)
        if not num:
            return None
        return name, int(num)

    def _ensure_chunks_for_training_files(self, file_paths: List[str]) -> List[str]:
        ensure_dir(self.chunks_root)
        fids = []
        for p in file_paths:
            fid, total = ensure_chunks_for_file(p, self.chunks_root)
            if total < MIN_CHUNKS:
                self.debug(f"Warning: {fid} has only {total} chunks")
            fids.append(fid)
        return fids

    def start_training(self):
        if not self.folder:
            messagebox.showerror("Error", "No folder selected.")
            return

        file_paths = list_audio_files(self.folder)
        if len(file_paths) < 2:
            messagebox.showerror("Error", "Need at least 2 audio files.")
            return

        sess = self._prompt_session()
        if sess is None:
            return
        session_name, num = sess

        self.session_name = session_name
        self.session_target = num
        self.session_done = 0

        ensure_dir(PREF_DIR)
        ensure_dir(os.path.dirname(SESSIONS_META))

        session_fp = os.path.join(PREF_DIR, f"{self.session_name}.csv")
        if os.path.exists(session_fp):
            ok = messagebox.askyesno("Session exists", "That session file already exists. Append to it?")
            if not ok:
                return

        self.session_fp = session_fp
        is_new = not os.path.exists(session_fp)

        try:
            self.session_fh = open(session_fp, "a", newline="")
            self.session_writer = csv.writer(self.session_fh)
            if is_new:
                self.session_writer.writerow([
                    "a_file", "a_chunk_idx", "a_chunk_total",
                    "b_file", "b_chunk_idx", "b_chunk_total",
                    "winner", "replacement"
                ])
                self.session_fh.flush()
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open session file:\n{e}")
            return

        self._set_text("session", f"Session: {self.session_name}")
        self._set_text("progress", f"Progress: {self.session_done}/{self.session_target}")
        self._set_text("pair", "Pair: preparing chunks...")
        self.debug(f"Session started: {self.session_name} ({self.session_target} comparisons)")

        def work():
            try:
                fids = self._ensure_chunks_for_training_files(file_paths)
                fids = [f for f in fids if chunk_paths_for_file_id(self.chunks_root, f)]
                fids = sorted(list(set(fids)))
                if len(fids) < 2:
                    self.root.after(0, lambda: messagebox.showerror("Error", "After chunking, need at least 2 valid files."))
                    return
                self.training_files = fids
                self.stats = load_historical_stats(self.training_files)
                self.current_pair = select_next_pair(self.stats, self.chunks_root, self.training_files, force=None)
                self.root.after(0, self._refresh_pair_ui)
                self.debug("Ready to label")
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Setup failed:\n{e}"))
            PAIR_QUEUE.clear()
            fill_pair_queue(self.stats, self.chunks_root, self.training_files, PAIR_QUEUE, QUEUE_SIZE)
            self.write_queue_to_txt()



        self.root.after(0, work)
    def write_queue_to_txt(self):
        ensure_dir("preferences")

        lines = []
        lines.append("PAIR QUEUE DEBUG")
        lines.append("=" * 40)
        lines.append(f"Session: {self.session_name}")
        lines.append(f"Completed: {self.session_done}/{self.session_target}")
        lines.append(f"Queue used: {getattr(self, 'queue_used', 0)}")
        lines.append(f"Queued items: {len(PAIR_QUEUE)}")
        lines.append("")

        if not PAIR_QUEUE:
            lines.append("(queue empty)")
        else:
            for i, p in enumerate(PAIR_QUEUE):
                prefix = ">> NEXT <<" if i == 0 else f"{i+1:02d}"
                lines.append(
                    f"{prefix}  "
                    f"A={p.a_file} ({p.a_chunk_idx}/{p.a_chunk_total})  "
                    f"vs  "
                    f"B={p.b_file} ({p.b_chunk_idx}/{p.b_chunk_total})"
                )

        try:
            with open(QUEUE_DEBUG_TXT, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            self.debug(f"Failed to write queue txt: {e}")


    def _refresh_pair_ui(self):
        if not self.current_pair:
            self._set_text("pair", "Pair: (none)")
            return
        p = self.current_pair
        self._set_text(
            "pair",
            f"Pair: A={p.a_file} ({p.a_chunk_idx}/{p.a_chunk_total})  vs  B={p.b_file} ({p.b_chunk_idx}/{p.b_chunk_total})"
        )
        self._set_text("progress", f"Progress: {self.session_done}/{self.session_target}")
        self.waveform_canvas.delete("all")

    def _load_chunk_audio(self, path: str) -> Tuple[np.ndarray, int]:
        if path in self.audio_cache:
            return self.audio_cache[path]
        y, sr = decode_mp3_to_float32(path, CHUNK_SR)
        self.audio_cache[path] = (y, sr)
        return y, sr

    def play_side(self, which: int):
        if not self.current_pair:
            return
        p = self.current_pair
        path = p.a_path if which == 0 else p.b_path
        if not path or not os.path.exists(path):
            self.debug("Missing chunk file")
            return

        y, sr = self._load_chunk_audio(path)
        if y is None or y.size == 0:
            return

        self.draw_waveform(y)
        self.reset_audio_device()

        system_vol = get_macos_output_volume()
        effective_vol = float(self.base_volume) * float(system_vol)
        safe_clip = np.clip(y * effective_vol, -1.0, 1.0)
        try:
            sd.play(safe_clip.astype(np.float32), sr)
            self.playing = True
            self.play_start = time.time()
            self.play_duration = float(len(safe_clip)) / float(sr)
            self.debug(f"Playing {'A' if which == 0 else 'B'}")
        except Exception as e:
            self.debug(f"Playback error: {e}")

    def _append_row(self, a_file: str, a_idx: int, a_total: int, b_file: str, b_idx: int, b_total: int, winner: int, replacement: str):
        if not self.session_writer or not self.session_fh:
            return
        self.session_writer.writerow([a_file, str(a_idx), str(a_total), b_file, str(b_idx), str(b_total), str(winner), str(replacement)])
        self.session_fh.flush()

    def _advance(self, force: Optional[Tuple[str, int]] = None):
        self.session_done += 1

        if self.session_done >= self.session_target:
            self.finish()
            return

        if not self.stats or not self.training_files:
            return

        # Refill queue if needed
        if len(PAIR_QUEUE) < QUEUE_SIZE:
            fill_pair_queue(
                self.stats,
                self.chunks_root,
                self.training_files,
                PAIR_QUEUE,
                QUEUE_SIZE
            )
        self.write_queue_to_txt()


    # FORCE overrides queue (skip case)
        if force is not None:
            self.current_pair = select_next_pair(
                self.stats,
                self.chunks_root,
                self.training_files,
                force=force
            )
        else:
            # ALWAYS pop from queue
            self.current_pair = PAIR_QUEUE.pop(0)
            self.queue_used += 1

        self._refresh_pair_ui()

    def vote(self, winner: int):
        if winner not in (1, 2):
            return
        if not self.current_pair or not self.stats:
            return
        p = self.current_pair

        self._append_row(
            p.a_file, p.a_chunk_idx, p.a_chunk_total,
            p.b_file, p.b_chunk_idx, p.b_chunk_total,
            winner, "0"
        )

        sess_weight = 1.0
        row = {
            "a_file": p.a_file,
            "a_chunk_idx": str(p.a_chunk_idx),
            "b_file": p.b_file,
            "b_chunk_idx": str(p.b_chunk_idx),
            "winner": str(winner),
            "replacement": "0"
        }
        self.stats.update_from_row(row, sess_weight)
        self.stats.ensure_file(p.a_file)
        self.stats.ensure_file(p.b_file)
        self.stats.elo_update(p.a_file, p.b_file, winner, k_base=18.0)

        self.debug(f"Voted: {'A' if winner == 1 else 'B'}")
        self._advance(force=None)

    def skip_chunk(self, which: int):
        if not self.current_pair or not self.stats:
            return
        p = self.current_pair

        if which == 0:
            old_idx = p.a_chunk_idx
            new_idx, new_total, new_path = choose_chunk_for_file(self.stats, self.chunks_root, p.a_file, avoid_idx=old_idx)
            replacement = f"A{new_idx}"
            self._append_row(
                p.a_file, p.a_chunk_idx, p.a_chunk_total,
                p.b_file, p.b_chunk_idx, p.b_chunk_total,
                0, replacement
            )
            row = {
                "a_file": p.a_file,
                "a_chunk_idx": str(p.a_chunk_idx),
                "b_file": p.b_file,
                "b_chunk_idx": str(p.b_chunk_idx),
                "winner": "0",
                "replacement": replacement
            }
            self.stats.update_from_row(row, 1.0)
            self.debug(f"Skipped A chunk {old_idx} -> {new_idx}")
            self._advance(force=(p.a_file, new_idx))
        else:
            old_idx = p.b_chunk_idx
            new_idx, new_total, new_path = choose_chunk_for_file(self.stats, self.chunks_root, p.b_file, avoid_idx=old_idx)
            replacement = f"B{new_idx}"
            self._append_row(
                p.a_file, p.a_chunk_idx, p.a_chunk_total,
                p.b_file, p.b_chunk_idx, p.b_chunk_total,
                0, replacement
            )
            row = {
                "a_file": p.a_file,
                "a_chunk_idx": str(p.a_chunk_idx),
                "b_file": p.b_file,
                "b_chunk_idx": str(p.b_chunk_idx),
                "winner": "0",
                "replacement": replacement
            }
            self.stats.update_from_row(row, 1.0)
            self.debug(f"Skipped B chunk {old_idx} -> {new_idx}")
            self._advance(force=(p.b_file, new_idx))
            self.write_queue_to_txt()


    def finish_early(self):
        if not self.session_writer or self.session_done <= 0:
            messagebox.showinfo("Nothing to save", "No comparisons made yet.")
            return
        ok = messagebox.askyesno("Finish", "Finish session now?")
        if ok:
            self.finish()

    def finish(self):
        try:
            if self.session_fh:
                self.session_fh.flush()
        except Exception:
            pass
        try:
            if self.session_fh:
                self.session_fh.close()
        except Exception:
            pass
        self.session_fh = None
        self.session_writer = None

        try:
            update_sessions_meta()
        except Exception as e:
            self.debug(f"sessions_meta update failed: {e}")

        messagebox.showinfo("Done", f"Saved session: {self.session_name}\nRows: {self.session_done}")
        self.debug("FINISHED")

        self.session_name = None
        self.session_target = 0
        self.session_done = 0
        self.session_fp = None
        self.current_pair = None
        self.training_files = []
        self.stats = None
        self.audio_cache = {}
        self._set_text("session", "Session: (none)")
        self._set_text("pair", "Pair: (none)")
        self._set_text("progress", "Progress: 0/0")
        self.waveform_canvas.delete("all")


if __name__ == "__main__":
    root = tk.Tk()
    app = PreferenceApp(root)
    root.mainloop()
