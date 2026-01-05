"""
mturk_pref/app.py

FastAPI webapp for A/B audio preference labeling.

Outputs:
  preferences/sessions/<session_id>.csv

CSV schema:
  a_file,a_chunk_idx,a_chunk_total,b_file,b_chunk_idx,b_chunk_total,winner,replacement

winner:
  1 = A better
  2 = B better
  0 = replaced/skipped (no vote)

replacement:
  "0" for normal votes
  reason string for replacement/skips
"""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
import time
import uuid
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel


class Settings(BaseModel):
    chunks_local_root: str = "chunks"
    chunks_base_url: str
    manifest_path: str
    pair_schedule_path: str
    db_path: str
    output_sessions_dir: str
    sessions_meta_path: str
    total_comparisons: Optional[int] = None
    comparisons_per_worker: int = 15
    reclaim_minutes: int = 120


@dataclass
class ComparisonState:
    a_file: str
    b_file: str
    a_chunk_idx: int
    b_chunk_idx: int
    a_seen: List[int]
    b_seen: List[int]
    a_exhausted: bool
    b_exhausted: bool


class SkipReq(BaseModel):
    session_id: str
    side: str


class VoteReq(BaseModel):
    session_id: str
    winner: int


def _now() -> float:
    return time.time()


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _read_text_best_effort(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace")


def _strip_illegal_json_control_chars(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if o < 32 and ch not in ("\n", "\r", "\t"):
            continue
        out.append(ch)
    return "".join(out)


def _read_json(p: Path) -> Any:
    txt = _read_text_best_effort(p)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        cleaned = _strip_illegal_json_control_chars(txt)
        if cleaned != txt:
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e2:
                raise RuntimeError(
                    f"Invalid JSON in {p.as_posix()} at line {e2.lineno} col {e2.colno}: {e2.msg}"
                ) from e2
        e = json.JSONDecodeError("Invalid JSON", txt, 0)
        raise RuntimeError(f"Invalid JSON in {p.as_posix()}: could not parse") from e


CONFIG_SOURCE_PATH = ""


def load_settings() -> Settings:
    env = os.getenv("MTURK_PREF_CONFIG", "").strip()
    candidates: List[Path] = []
    if env:
        candidates.append(Path(env))
    candidates += [
        Path("mturk_pref/config.json"),
        Path("mturk_pref/settings.json"),
        Path("config.json"),
        Path("settings.json"),
    ]
    last_err: Optional[Exception] = None
    for c in candidates:
        if c.exists():
            try:
                data = _read_json(c)
                if not isinstance(data, dict):
                    raise RuntimeError(f"Config file {c.as_posix()} must be a JSON object.")
                global CONFIG_SOURCE_PATH
                CONFIG_SOURCE_PATH = c.as_posix()
                return Settings(**data)
            except Exception as e:
                last_err = e
    if last_err:
        raise RuntimeError(str(last_err))
    raise RuntimeError("No config found. Create mturk_pref/config.json (or settings.json) or set MTURK_PREF_CONFIG.")


def _dir_writable(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        t = p / f".__w_{uuid.uuid4().hex}"
        t.write_text("x", encoding="utf-8")
        t.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _data_root() -> Path:
    env = os.getenv("MTURK_PREF_DATA_ROOT", "").strip()
    if env:
        root = Path(env)
        if _dir_writable(root):
            return root
    cand = Path("/var/data")
    if _dir_writable(cand):
        return cand
    cand2 = Path(".mturk_pref_data")
    _dir_writable(cand2)
    return cand2


def _resolve_storage_paths(settings: Settings) -> Settings:
    root = _data_root()

    def _fix(path_str: str) -> str:
        s = (path_str or "").strip()
        if not s:
            return s
        p = Path(s)
        if p.is_absolute():
            if str(p).startswith("/var/data") and not _dir_writable(Path("/var/data")):
                rel = str(p)[len("/var/data") :].lstrip("/")
                return str((root / rel).resolve())
            return str(p)
        return str((root / s).resolve())

    settings.db_path = _fix(settings.db_path)
    settings.output_sessions_dir = _fix(settings.output_sessions_dir)
    settings.sessions_meta_path = _fix(settings.sessions_meta_path)
    return settings


def _join_url(base: str, rel: str) -> str:
    rel = rel.strip()
    if rel.startswith("http://") or rel.startswith("https://"):
        return rel
    base_clean = base.rstrip("/")
    rel_clean = rel.lstrip("/")
    if base_clean.endswith("/chunks") and rel_clean.startswith("chunks/"):
        rel_clean = rel_clean[len("chunks/") :]
    return base_clean + "/" + rel_clean


_chunk_re = re.compile(r"chunk[_\- ]?(\d+)[_\- ]+of[_\- ]+(\d+)", re.IGNORECASE)


def _pair_to_files(pair: Any) -> Tuple[str, str]:
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        return str(pair[0]), str(pair[1])
    if isinstance(pair, dict):
        for ak, bk in (("a_file", "b_file"), ("a", "b"), ("left", "right")):
            if ak in pair and bk in pair:
                return str(pair[ak]), str(pair[bk])
    raise RuntimeError(f"Unrecognized pair row shape: {type(pair)} {pair}")


def _manifest_files_dict(manifest: Any) -> Dict[str, Any]:
    if isinstance(manifest, dict) and "files" in manifest and isinstance(manifest["files"], dict):
        return manifest["files"]
    if isinstance(manifest, dict):
        return manifest
    raise RuntimeError("Manifest is not a dict-like object")


def _chunks_list_for_file(file_id: str, manifest: Any) -> List[str]:
    files = _manifest_files_dict(manifest)
    if file_id not in files:
        raise RuntimeError(f"file_id not found in manifest: {file_id}")
    fobj = files[file_id]
    if not isinstance(fobj, dict):
        raise RuntimeError(f"manifest entry for {file_id} is not a dict")
    chunks = None
    for k in ("chunks", "chunk_paths", "chunk_urls"):
        if k in fobj:
            chunks = fobj[k]
            break
    if chunks is None:
        raise RuntimeError(f"manifest entry for {file_id} has no chunks field")
    if isinstance(chunks, list):
        return [str(x) for x in chunks]
    if isinstance(chunks, dict):

        def _key_sort(x: Any) -> Tuple[int, str]:
            s = str(x)
            m = re.findall(r"\d+", s)
            return (int(m[-1]) if m else 10**9, s)

        keys = sorted(chunks.keys(), key=_key_sort)
        return [str(chunks[k]) for k in keys]
    raise RuntimeError(f"chunks field for {file_id} must be list or dict, got {type(chunks)}")


def _public_chunk_url(rel_or_url: str, chunks_base_url: str) -> str:
    s = str(rel_or_url)
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return chunks_base_url.rstrip("/") + "/" + s.lstrip("/")


def _chunk_sort_key(u: str) -> Tuple[int, str]:
    m = _chunk_re.search(u)
    if m:
        return (int(m.group(1)), u)
    m2 = re.search(r"(\d+)", Path(u).name)
    if m2:
        return (int(m2.group(1)), u)
    return (10**9, u)


def normalize_manifest(data: Any, chunks_base_url: str) -> Dict[str, List[str]]:
    files = data.get("files") if isinstance(data, dict) else None
    out: Dict[str, List[str]] = {}

    def norm_list(lst: List[Any]) -> List[str]:
        urls = [_join_url(chunks_base_url, str(x)) for x in lst]
        urls = [u for u in urls if u]
        return sorted(urls, key=_chunk_sort_key)

    if isinstance(files, dict):
        for file_id, meta in files.items():
            if isinstance(meta, dict):
                ch = meta.get("chunks") or meta.get("chunk_urls") or meta.get("urls") or meta.get("paths")
                if isinstance(ch, list):
                    out[str(file_id)] = norm_list(ch)
                elif isinstance(ch, dict):
                    vals = [str(v) for _, v in sorted(ch.items(), key=lambda kv: str(kv[0]))]
                    out[str(file_id)] = sorted([_join_url(chunks_base_url, v) for v in vals if v], key=_chunk_sort_key)
            elif isinstance(meta, list):
                out[str(file_id)] = norm_list(meta)

    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            fid = item.get("file_id") or item.get("id") or item.get("name")
            if not fid:
                continue
            ch = item.get("chunks") or item.get("chunk_urls") or item.get("urls") or item.get("paths")
            if isinstance(ch, list):
                out[str(fid)] = norm_list(ch)
            elif isinstance(ch, dict):
                vals = [str(v) for _, v in sorted(ch.items(), key=lambda kv: str(kv[0]))]
                out[str(fid)] = sorted([_join_url(chunks_base_url, v) for v in vals if v], key=_chunk_sort_key)

    out = {k: v for k, v in out.items() if isinstance(v, list) and len(v) > 0}

    if not out:
        raise RuntimeError("Manifest parsed to 0 files with chunks. Fix manifest.json format.")
    return out


def _db_connect(db_path: str) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _db_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {str(r[1]) for r in cur.fetchall()}


def _db_init(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_ts REAL NOT NULL,
            updated_ts REAL NOT NULL,
            slot INTEGER,
            required_votes INTEGER NOT NULL,
            votes_done INTEGER NOT NULL,
            presented INTEGER NOT NULL,
            local_index INTEGER NOT NULL,
            state_json TEXT NOT NULL,
            csv_path TEXT NOT NULL,
            worker_id TEXT,
            assignment_id TEXT,
            hit_id TEXT,
            turk_submit_to TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            session_id TEXT,
            kind TEXT NOT NULL,
            detail TEXT NOT NULL
        )
        """
    )
    conn.commit()
    cols = _db_columns(conn, "sessions")
    if "turk_submit_to" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN turk_submit_to TEXT")
        conn.commit()


def _meta_get(conn: sqlite3.Connection, k: str, default: str) -> str:
    cur = conn.execute("SELECT v FROM meta WHERE k=?", (k,))
    row = cur.fetchone()
    if not row:
        return default
    return str(row[0])


def _meta_set(conn: sqlite3.Connection, k: str, v: str) -> None:
    conn.execute("INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))
    conn.commit()


def _log_event(kind: str, session_id: str, detail: str) -> None:
    try:
        DB.execute(
            "INSERT INTO events(ts, session_id, kind, detail) VALUES(?,?,?,?)",
            (_now(), str(session_id or ""), str(kind or ""), str(detail or "")[:2000]),
        )
        DB.commit()
    except Exception:
        pass


def _session_csv_path(settings: Settings, session_id: str) -> Path:
    root = Path(settings.output_sessions_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", session_id)[:180]
    return root / f"{safe}.csv"


def _write_csv_header_if_missing(p: Path) -> None:
    if p.exists() and p.stat().st_size > 0:
        return
    _ensure_parent(p)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a_file", "a_chunk_idx", "a_chunk_total", "b_file", "b_chunk_idx", "b_chunk_total", "winner", "replacement"])


def _append_row(
    p: Path,
    a_file: str,
    a_idx_1based: int,
    a_total: int,
    b_file: str,
    b_idx_1based: int,
    b_total: int,
    winner: int,
    replacement: str,
) -> None:
    _write_csv_header_if_missing(p)
    with p.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([a_file, a_idx_1based, a_total, b_file, b_idx_1based, b_total, winner, replacement])


def _session_id_from_params(
    assignment_id: Optional[str],
    worker_id: Optional[str],
    hit_id: Optional[str],
    sid: Optional[str],
) -> str:
    if sid:
        return sid
    if assignment_id and assignment_id != "ASSIGNMENT_ID_NOT_AVAILABLE":
        w = worker_id or "noworker"
        h = hit_id or "nohit"
        return f"{h}_{assignment_id}_{w}"
    return f"local_{uuid.uuid4().hex}"


def _state_to_json(st: ComparisonState) -> Dict[str, Any]:
    return {
        "a_file": st.a_file,
        "b_file": st.b_file,
        "a_chunk_idx": st.a_chunk_idx,
        "b_chunk_idx": st.b_chunk_idx,
        "a_seen": st.a_seen,
        "b_seen": st.b_seen,
        "a_exhausted": st.a_exhausted,
        "b_exhausted": st.b_exhausted,
    }


def _state_from_json(d: Dict[str, Any]) -> ComparisonState:
    return ComparisonState(
        a_file=str(d["a_file"]),
        b_file=str(d["b_file"]),
        a_chunk_idx=int(d.get("a_chunk_idx", 0)),
        b_chunk_idx=int(d.get("b_chunk_idx", 0)),
        a_seen=list(d.get("a_seen", [])),
        b_seen=list(d.get("b_seen", [])),
        a_exhausted=bool(d.get("a_exhausted", False)),
        b_exhausted=bool(d.get("b_exhausted", False)),
    )


def _new_state(a_file: str, b_file: str) -> ComparisonState:
    return ComparisonState(
        a_file=a_file,
        b_file=b_file,
        a_chunk_idx=0,
        b_chunk_idx=0,
        a_seen=[],
        b_seen=[],
        a_exhausted=False,
        b_exhausted=False,
    )


def _reclaim_stale(conn: sqlite3.Connection, settings: Settings) -> None:
    cutoff = _now() - float(settings.reclaim_minutes) * 60.0
    cur = conn.execute(
        "SELECT session_id, csv_path FROM sessions WHERE votes_done < required_votes AND updated_ts < ?",
        (cutoff,),
    )
    rows = cur.fetchall()
    for sid, csvp in rows:
        try:
            Path(str(csvp)).unlink(missing_ok=True)
        except Exception:
            pass
        conn.execute("DELETE FROM sessions WHERE session_id=?", (sid,))
    conn.commit()


def _allocate_slot(conn: sqlite3.Connection, settings: Settings, total_pairs: int) -> int:
    _reclaim_stale(conn, settings)
    total = int(settings.total_comparisons) if settings.total_comparisons is not None else int(total_pairs)
    per = int(settings.comparisons_per_worker)
    n_slots = max(1, (total + per - 1) // per)
    used = set()
    cur = conn.execute("SELECT slot FROM sessions WHERE slot IS NOT NULL")
    for (s,) in cur.fetchall():
        if s is not None:
            used.add(int(s))
    for s in range(n_slots):
        if s not in used:
            return s
    overflow = int(_meta_get(conn, "overflow_slot_counter", str(n_slots)))
    _meta_set(conn, "overflow_slot_counter", str(overflow + 1))
    return overflow


def load_pairs(pair_csv: Path) -> List[Tuple[str, str]]:
    with pair_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise RuntimeError("pair_schedule.csv is empty.")
    a_key = "a_file" if "a_file" in rows[0] else ("A" if "A" in rows[0] else None)
    b_key = "b_file" if "b_file" in rows[0] else ("B" if "B" in rows[0] else None)
    if not a_key or not b_key:
        raise RuntimeError(f"pair_schedule.csv needs columns a_file,b_file (found {list(rows[0].keys())}).")
    pairs: List[Tuple[str, str]] = []
    for row in rows:
        a = str(row[a_key]).strip()
        b = str(row[b_key]).strip()
        if a and b:
            pairs.append((a, b))
    if not pairs:
        raise RuntimeError("pair_schedule.csv had no usable rows.")
    return pairs


def filter_pairs(pairs: List[Tuple[str, str]], manifest: Dict[str, List[str]]) -> Tuple[List[Tuple[str, str]], int]:
    good: List[Tuple[str, str]] = []
    bad = 0
    for a, b in pairs:
        if a in manifest and b in manifest and manifest.get(a) and manifest.get(b):
            good.append((a, b))
        else:
            bad += 1
    if not good:
        raise RuntimeError("After filtering, there are 0 valid pairs. pair_schedule.csv and manifest.json do not match.")
    return good, bad


def _pick_pair(
    pairs: List[Tuple[str, str]],
    slot: int,
    local_index: int,
    settings: Settings,
    conn: sqlite3.Connection,
) -> Tuple[Tuple[str, str], int, int]:
    per = int(settings.comparisons_per_worker)
    start = slot * per
    end = start + per
    if start < len(pairs) and (start + local_index) < min(end, len(pairs)):
        return pairs[start + local_index], local_index, 0
    overflow = int(_meta_get(conn, "overflow_idx", "0"))
    pair = pairs[overflow % len(pairs)]
    _meta_set(conn, "overflow_idx", str(overflow + 1))
    return pair, local_index, 1


def _state_valid(st: ComparisonState, manifest: Dict[str, List[str]]) -> bool:
    return (
        st.a_file in manifest
        and st.b_file in manifest
        and isinstance(manifest.get(st.a_file), list)
        and isinstance(manifest.get(st.b_file), list)
        and len(manifest[st.a_file]) > 0
        and len(manifest[st.b_file]) > 0
    )


def _payload(
    settings: Settings,
    manifest: Dict[str, List[str]],
    pairs_total: int,
    session_id: str,
    assignment_id: str,
    worker_id: str,
    hit_id: str,
    turk_submit_to: str,
    required_votes: int,
    votes_done: int,
    presented: int,
    st: ComparisonState,
    msg: str,
    finished: bool,
) -> Dict[str, Any]:
    a_list = manifest[st.a_file]
    b_list = manifest[st.b_file]
    a_total = len(a_list)
    b_total = len(b_list)
    a_idx = max(0, min(st.a_chunk_idx, a_total - 1))
    b_idx = max(0, min(st.b_chunk_idx, b_total - 1))
    total_comparisons = int(settings.total_comparisons) if settings.total_comparisons is not None else int(pairs_total)
    return {
        "session_id": session_id,
        "assignment_id": assignment_id,
        "worker_id": worker_id,
        "hit_id": hit_id,
        "turk_submit_to": turk_submit_to,
        "total_comparisons": total_comparisons,
        "required_votes": required_votes,
        "votes_done": votes_done,
        "presented": presented,
        "finished": finished,
        "message": msg,
        "a": {
            "file_id": st.a_file,
            "chunk_idx": a_idx,
            "chunk_idx_1based": a_idx + 1,
            "chunk_total": a_total,
            "url": a_list[a_idx],
            "exhausted": st.a_exhausted,
        },
        "b": {
            "file_id": st.b_file,
            "chunk_idx": b_idx,
            "chunk_idx_1based": b_idx + 1,
            "chunk_total": b_total,
            "url": b_list[b_idx],
            "exhausted": st.b_exhausted,
        },
    }


def _html() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Audio Preference Trainer</title>
  <style>
    :root{
      --bg:#f6d8d8;
      --panel:#e8a896;
      --panel2:#e7a392;
      --card:#ffffff;
      --shadow: 0 10px 30px rgba(0,0,0,.10);
      --shadow2: 0 8px 18px rgba(0,0,0,.10);
      --btn:#c7b6c9;
      --btnText:#1f1b24;
      --btnStrong:#f4b35a;
      --btnStrongText:#1c1308;
      --muted:#5f5163;
    }
    html,body{height:100%;}
    body{
      margin:0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color:#1d1420;
    }
    .wrap{
      max-width: 1100px;
      margin: 42px auto;
      padding: 0 18px;
    }
    .shell{
      background: var(--card);
      border-radius: 26px;
      box-shadow: var(--shadow);
      padding: 34px 34px 26px 34px;
    }
    h1{
      margin:0 0 6px 0;
      font-size: 46px;
      letter-spacing:-0.02em;
      line-height:1.0;
    }
    .sub{
      margin:0 0 22px 0;
      font-size: 18px;
      color: var(--muted);
    }
    .toprow{
      display:flex;
      justify-content:space-between;
      align-items:flex-start;
      gap: 12px;
    }
    .err{
      font-size: 16px;
      color:#6a1b1b;
      margin-top: 8px;
      min-height: 22px;
      white-space: pre-wrap;
    }
    .progress{
      font-size: 14px;
      background: rgba(0,0,0,.06);
      padding: 10px 14px;
      border-radius: 999px;
      color: rgba(0,0,0,.70);
      margin-top: 6px;
      user-select:none;
    }
    .grid{
      display:grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-top: 18px;
    }
    .panel{
      background: var(--panel);
      border-radius: 18px;
      box-shadow: var(--shadow2);
      padding: 18px;
      position:relative;
      overflow:hidden;
    }
    .panel h2{
      margin:0 0 10px 0;
      font-size: 26px;
      letter-spacing:-0.01em;
    }
    .pill{
      display:inline-block;
      font-size: 13px;
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,.35);
      color: rgba(0,0,0,.70);
      margin-top: 4px;
      user-select:none;
    }
    .playerbar{
      margin-top: 12px;
      background: rgba(255,255,255,.28);
      border-radius: 14px;
      padding: 14px;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap: 12px;
    }
    .playbtn{
      width:100%;
      border:0;
      background: rgba(0,0,0,.12);
      color: rgba(255,255,255,.92);
      padding: 14px 16px;
      border-radius: 14px;
      font-size: 16px;
      font-weight: 700;
      cursor:pointer;
      user-select:none;
    }
    .playbtn:hover{filter:brightness(1.04);}
    .row{
      display:flex;
      gap: 12px;
      margin-top: 12px;
    }
    .btn{
      border:0;
      background: var(--btn);
      color: var(--btnText);
      padding: 14px 18px;
      border-radius: 16px;
      font-size: 16px;
      font-weight: 800;
      cursor:pointer;
      user-select:none;
      width: 100%;
    }
    .btn:hover{filter:brightness(1.03);}
    .btn:disabled{opacity:.55; cursor:not-allowed;}
    .voteRow{
      display:grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-top: 18px;
    }
    .vote{
      border:0;
      border-radius: 18px;
      padding: 18px 20px;
      font-size: 20px;
      font-weight: 900;
      cursor:pointer;
      user-select:none;
      box-shadow: var(--shadow2);
    }
    .voteA{ background: var(--btnStrong); color: var(--btnStrongText); }
    .voteB{ background: #c7b6c9; color: #201a26; }
    .vote:disabled{opacity:.55; cursor:not-allowed;}
    .footer{
      margin-top: 18px;
      display:flex;
      justify-content:flex-end;
      align-items:center;
      gap: 12px;
      min-height: 44px;
    }
    .submitBtn{
      border:0;
      background: #2b2b2b;
      color: white;
      padding: 12px 16px;
      border-radius: 14px;
      font-weight: 900;
      cursor:pointer;
    }
    .submitBtn:disabled{opacity:.55; cursor:not-allowed;}
    .small{
      font-size: 13px;
      color: rgba(0,0,0,.60);
    }
    @media (max-width: 900px){
      .grid{grid-template-columns: 1fr;}
      .voteRow{grid-template-columns: 1fr;}
      h1{font-size: 38px;}
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="shell">
      <div class="toprow">
        <div>
          <h1>Audio Preference Trainer</h1>
          <div class="sub">Listen to A and B. Choose which sounds more confident. Skip cycles chunks on that side.</div>
          <div class="err" id="err"></div>
        </div>
        <div class="progress" id="progress">Progress: 0/0</div>
      </div>

      <div class="grid">
        <div class="panel">
          <h2>A</h2>
          <div class="pill" id="aMeta">Loading…</div>
          <div class="playerbar">
            <button class="playbtn" id="playA">▶ Play A</button>
          </div>
          <div class="row">
            <button class="btn" id="skipA">Skip A Chunk</button>
          </div>
          <audio id="audioA" preload="none"></audio>
        </div>

        <div class="panel" style="background: var(--panel2);">
          <h2>B</h2>
          <div class="pill" id="bMeta">Loading…</div>
          <div class="playerbar">
            <button class="playbtn" id="playB">▶ Play B</button>
          </div>
          <div class="row">
            <button class="btn" id="skipB">Skip B Chunk</button>
          </div>
          <audio id="audioB" preload="none"></audio>
        </div>
      </div>

      <div class="voteRow">
        <button class="vote voteA" id="voteA">A IS BETTER</button>
        <button class="vote voteB" id="voteB">B IS BETTER</button>
      </div>

      <div class="footer" id="footer"></div>
      <div class="small" id="small"></div>
    </div>
  </div>

<script>
(function(){
  const qs = new URLSearchParams(window.location.search);
  const assignmentId = qs.get("assignmentId") || "";
  const workerId = qs.get("workerId") || "";
  const hitId = qs.get("hitId") || "";
  const turkSubmitTo = qs.get("turkSubmitTo") || "";

  let sid = qs.get("sid") || "";

  const preview = (assignmentId === "ASSIGNMENT_ID_NOT_AVAILABLE");
  const mturkReal = (!!assignmentId && !preview);

  if(!sid && !mturkReal){
    sid = localStorage.getItem("pref_sid") || "";
    if(!sid){
      sid = "sid_" + Math.random().toString(16).slice(2) + "_" + Date.now().toString(16);
      localStorage.setItem("pref_sid", sid);
    }
  }

  const err = document.getElementById("err");
  const progress = document.getElementById("progress");

  const aMeta = document.getElementById("aMeta");
  const bMeta = document.getElementById("bMeta");

  const audioA = document.getElementById("audioA");
  const audioB = document.getElementById("audioB");

  const playA = document.getElementById("playA");
  const playB = document.getElementById("playB");

  const skipA = document.getElementById("skipA");
  const skipB = document.getElementById("skipB");

  const voteA = document.getElementById("voteA");
  const voteB = document.getElementById("voteB");

  const footer = document.getElementById("footer");
  const small = document.getElementById("small");

  let STATE = null;

  function setEnabled(on){
    [playA, playB, skipA, skipB, voteA, voteB].forEach(b => b.disabled = !on);
  }

  function pauseBoth(){
    try{ audioA.pause(); }catch(e){}
    try{ audioB.pause(); }catch(e){}
    playA.textContent = "▶ Play A";
    playB.textContent = "▶ Play B";
  }

  function updateUI(st){
    STATE = st;

    if(st.message){
      err.textContent = st.message;
    }else{
      err.textContent = "";
    }

    progress.textContent = "Progress: " + st.votes_done + "/" + st.required_votes;

    audioA.src = st.a.url;
    audioB.src = st.b.url;

    aMeta.textContent = st.a.file_id + " (chunk " + st.a.chunk_idx_1based + "/" + st.a.chunk_total + ")";
    bMeta.textContent = st.b.file_id + " (chunk " + st.b.chunk_idx_1based + "/" + st.b.chunk_total + ")";

    small.textContent = "Session: " + st.session_id;

    footer.innerHTML = "";

    if(st.finished){
      setEnabled(false);
      pauseBoth();

      if(st.turk_submit_to && st.assignment_id && st.assignment_id !== "ASSIGNMENT_ID_NOT_AVAILABLE"){
        const form = document.createElement("form");
        form.method = "POST";
        form.action = st.turk_submit_to.replace(/\\/$/,"") + "/mturk/externalSubmit";

        const a1 = document.createElement("input");
        a1.type = "hidden";
        a1.name = "assignmentId";
        a1.value = st.assignment_id;

        const a2 = document.createElement("input");
        a2.type = "hidden";
        a2.name = "session_id";
        a2.value = st.session_id;

        const btn = document.createElement("button");
        btn.type = "submit";
        btn.className = "submitBtn";
        btn.textContent = "Submit HIT";

        form.appendChild(a1);
        form.appendChild(a2);
        form.appendChild(btn);
        footer.appendChild(form);
      }else{
        const done = document.createElement("div");
        done.className = "small";
        done.textContent = "Done. You can close this tab.";
        footer.appendChild(done);
      }
    }else{
      setEnabled(true);
    }
  }

  function togglePlay(audioEl, btn, label){
    if(!audioEl.src){
      return;
    }
    if(audioEl.paused){
      pauseBoth();
      audioEl.play();
      btn.textContent = "⏸ Pause " + label;
    }else{
      audioEl.pause();
      btn.textContent = "▶ Play " + label;
    }
  }

  playA.addEventListener("click", () => togglePlay(audioA, playA, "A"));
  playB.addEventListener("click", () => togglePlay(audioB, playB, "B"));

  audioA.addEventListener("ended", () => { playA.textContent = "▶ Play A"; });
  audioB.addEventListener("ended", () => { playB.textContent = "▶ Play B"; });

  async function api(path, opts){
    const res = await fetch(path, Object.assign({ headers: { "Content-Type":"application/json" } }, opts || {}));
    if(!res.ok){
      const t = await res.text();
      throw new Error(res.status + " " + t);
    }
    return await res.json();
  }

  async function start(){
    try{
      if(preview){
        err.textContent = "Preview mode: accept the HIT to begin.";
        setEnabled(false);
        return;
      }
      const url = new URL("/api/session", window.location.origin);
      if(assignmentId) url.searchParams.set("assignmentId", assignmentId);
      if(workerId) url.searchParams.set("workerId", workerId);
      if(hitId) url.searchParams.set("hitId", hitId);
      if(turkSubmitTo) url.searchParams.set("turkSubmitTo", turkSubmitTo);
      if(sid) url.searchParams.set("sid", sid);

      const st = await api(url.toString(), { method:"GET" });
      updateUI(st);
    }catch(e){
      err.textContent = "Error: " + e.message;
      setEnabled(false);
    }
  }

  async function doSkip(side){
    try{
      pauseBoth();
      const st = await api("/api/skip", { method:"POST", body: JSON.stringify({ session_id: STATE.session_id, side }) });
      updateUI(st);
    }catch(e){
      err.textContent = "Error: " + e.message;
    }
  }

  async function doVote(winner){
    try{
      pauseBoth();
      const st = await api("/api/vote", { method:"POST", body: JSON.stringify({ session_id: STATE.session_id, winner }) });
      updateUI(st);
    }catch(e){
      err.textContent = "Error: " + e.message;
    }
  }

  skipA.addEventListener("click", () => doSkip("A"));
  skipB.addEventListener("click", () => doSkip("B"));
  voteA.addEventListener("click", () => doVote(1));
  voteB.addEventListener("click", () => doVote(2));

  start();
})();
</script>
</body>
</html>
""".strip()


def _require_admin_token(token: str) -> None:
    expected = os.getenv("MTURK_PREF_ADMIN_TOKEN", "").strip()
    if expected:
        if (token or "").strip() != expected:
            raise HTTPException(401, "unauthorized")


SETTINGS = _resolve_storage_paths(load_settings())
MANIFEST_RAW = _read_json(Path(SETTINGS.manifest_path))
MANIFEST = normalize_manifest(MANIFEST_RAW, SETTINGS.chunks_base_url)
PAIRS_RAW = load_pairs(Path(SETTINGS.pair_schedule_path))
PAIRS, DROPPED_PAIR_COUNT = filter_pairs(PAIRS_RAW, MANIFEST)

if SETTINGS.total_comparisons is None:
    SETTINGS.total_comparisons = len(PAIRS)

DB = _db_connect(SETTINGS.db_path)
_db_init(DB)

app = FastAPI()


@app.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    return "ok"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _html()


def _load_session(session_id: str) -> Tuple[str, int, int, int, int, int, ComparisonState, Path, str, str, str, str]:
    cur = DB.execute(
        "SELECT slot, required_votes, votes_done, presented, local_index, state_json, csv_path, worker_id, assignment_id, hit_id, turk_submit_to FROM sessions WHERE session_id=?",
        (session_id,),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(404, "session not found")
    slot, required_votes, votes_done, presented, local_index, state_json, csv_path, worker_id, assignment_id, hit_id, turk_submit_to = row
    st = _state_from_json(json.loads(state_json))
    return (
        session_id,
        int(slot) if slot is not None else -1,
        int(required_votes),
        int(votes_done),
        int(presented),
        int(local_index),
        st,
        Path(str(csv_path)),
        str(worker_id or ""),
        str(assignment_id or ""),
        str(hit_id or ""),
        str(turk_submit_to or ""),
    )


def _save_session(
    session_id: str,
    votes_done: int,
    presented: int,
    local_index: int,
    st: ComparisonState,
) -> None:
    DB.execute(
        "UPDATE sessions SET updated_ts=?, votes_done=?, presented=?, local_index=?, state_json=? WHERE session_id=?",
        (_now(), int(votes_done), int(presented), int(local_index), json.dumps(_state_to_json(st)), session_id),
    )
    DB.commit()


def _advance(slot: int, local_index: int) -> Tuple[ComparisonState, int]:
    local_index += 1
    pair, local_index, _ = _pick_pair(PAIRS, slot, local_index, SETTINGS, DB)
    return _new_state(pair[0], pair[1]), local_index


def _repair_state(slot: int, local_index: int, st: ComparisonState) -> Tuple[ComparisonState, int, str]:
    tries = 0
    msg = ""
    while tries < min(2000, len(PAIRS) + 10):
        if _state_valid(st, MANIFEST):
            return st, local_index, msg
        st, local_index = _advance(slot, local_index)
        msg = "Skipped an invalid pair (missing from manifest)."
        tries += 1
    raise HTTPException(500, "No valid pairs available after repair attempts.")


@app.get("/api/session", response_class=JSONResponse)
def api_session(
    assignmentId: str = Query(default=""),
    workerId: str = Query(default=""),
    hitId: str = Query(default=""),
    turkSubmitTo: str = Query(default=""),
    sid: str = Query(default=""),
) -> JSONResponse:
    assignment_id = assignmentId.strip() or ""
    worker_id = workerId.strip() or ""
    hit_id = hitId.strip() or ""
    turk_submit_to = turkSubmitTo.strip() or ""
    sid_use = sid.strip() or ""

    session_id = _session_id_from_params(
        assignment_id if assignment_id else None,
        worker_id if worker_id else None,
        hit_id if hit_id else None,
        sid_use if sid_use else None,
    )

    cur = DB.execute("SELECT session_id FROM sessions WHERE session_id=?", (session_id,))
    exists = cur.fetchone() is not None

    if not exists:
        slot = _allocate_slot(DB, SETTINGS, len(PAIRS))
        required_votes = int(SETTINGS.comparisons_per_worker)
        votes_done = 0
        presented = 0
        local_index = 0

        pair, local_index, _ = _pick_pair(PAIRS, slot, local_index, SETTINGS, DB)
        st = _new_state(pair[0], pair[1])

        csvp = _session_csv_path(SETTINGS, session_id)
        _write_csv_header_if_missing(csvp)

        DB.execute(
            """
            INSERT INTO sessions(session_id, created_ts, updated_ts, slot, required_votes, votes_done, presented, local_index, state_json, csv_path, worker_id, assignment_id, hit_id, turk_submit_to)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                session_id,
                _now(),
                _now(),
                int(slot),
                int(required_votes),
                int(votes_done),
                int(presented),
                int(local_index),
                json.dumps(_state_to_json(st)),
                str(csvp),
                worker_id,
                assignment_id,
                hit_id,
                turk_submit_to,
            ),
        )
        DB.commit()
    else:
        DB.execute(
            "UPDATE sessions SET updated_ts=?, worker_id=?, assignment_id=?, hit_id=?, turk_submit_to=? WHERE session_id=?",
            (_now(), worker_id, assignment_id, hit_id, turk_submit_to, session_id),
        )
        DB.commit()

    session_id, slot, required_votes, votes_done, presented, local_index, st, csvp, worker_id, assignment_id, hit_id, turk_submit_to = _load_session(session_id)
    current_required = int(SETTINGS.comparisons_per_worker)
    if required_votes != current_required:
        DB.execute("UPDATE sessions SET required_votes=? WHERE session_id=?", (current_required, session_id))
        DB.commit()
        required_votes = current_required
    st, local_index, msg = _repair_state(slot, local_index, st)
    _save_session(session_id, votes_done, presented, local_index, st)

    finished = votes_done >= required_votes
    payload = _payload(
        SETTINGS,
        MANIFEST,
        len(PAIRS),
        session_id,
        assignment_id,
        worker_id,
        hit_id,
        turk_submit_to,
        required_votes,
        votes_done,
        presented,
        st,
        msg=msg,
        finished=finished,
    )
    return JSONResponse(payload)


@app.post("/api/skip", response_class=JSONResponse)
def api_skip(req: SkipReq) -> JSONResponse:
    session_id = (req.session_id or "").strip()
    if not session_id:
        raise HTTPException(400, "missing session_id")

    session_id, slot, required_votes, votes_done, presented, local_index, st, csvp, worker_id, assignment_id, hit_id, turk_submit_to = _load_session(session_id)

    st, local_index, msg0 = _repair_state(slot, local_index, st)

    if votes_done >= required_votes:
        payload = _payload(
            SETTINGS,
            MANIFEST,
            len(PAIRS),
            session_id,
            assignment_id,
            worker_id,
            hit_id,
            turk_submit_to,
            required_votes,
            votes_done,
            presented,
            st,
            msg=msg0,
            finished=True,
        )
        return JSONResponse(payload)

    side = (req.side or "").strip().upper()
    if side not in ("A", "B"):
        try:
            _log_event("skip", session_id, f"side={side}")
        except Exception:
            pass
        try:
            _log_event("session", session_id, f"assignment={assignment_id} worker={worker_id} hit={hit_id}")
        except Exception:
            pass
        raise HTTPException(400, "side must be A or B")

    a_list = MANIFEST[st.a_file]
    b_list = MANIFEST[st.b_file]

    msg = msg0

    if side == "A":
        if st.a_chunk_idx not in st.a_seen:
            st.a_seen.append(int(st.a_chunk_idx))
        st.a_chunk_idx = (st.a_chunk_idx + 1) % len(a_list)
        if len(set(st.a_seen)) >= len(a_list):
            st.a_exhausted = True
    else:
        if st.b_chunk_idx not in st.b_seen:
            st.b_seen.append(int(st.b_chunk_idx))
        st.b_chunk_idx = (st.b_chunk_idx + 1) % len(b_list)
        if len(set(st.b_seen)) >= len(b_list):
            st.b_exhausted = True

    if st.a_exhausted or st.b_exhausted:
        presented += 1

        if st.a_exhausted and st.b_exhausted:
            reason = "SKIPPED_ALL_CHUNKS_BOTH"
            msg = (msg + "\n" if msg else "") + "All chunks skipped on both sides. Loaded a new comparison."
        elif st.a_exhausted:
            reason = "SKIPPED_ALL_CHUNKS_A"
            msg = (msg + "\n" if msg else "") + "All chunks skipped on A. Loaded a new comparison."
        else:
            reason = "SKIPPED_ALL_CHUNKS_B"
            msg = (msg + "\n" if msg else "") + "All chunks skipped on B. Loaded a new comparison."

        _append_row(
            csvp,
            st.a_file,
            int(st.a_chunk_idx) + 1,
            len(a_list),
            st.b_file,
            int(st.b_chunk_idx) + 1,
            len(b_list),
            0,
            reason,
        )

        st, local_index = _advance(slot, local_index)
        st, local_index, msg2 = _repair_state(slot, local_index, st)
        if msg2:
            msg = (msg + "\n" if msg else "") + msg2

    _save_session(session_id, votes_done, presented, local_index, st)

    payload = _payload(
        SETTINGS,
        MANIFEST,
        len(PAIRS),
        session_id,
        assignment_id,
        worker_id,
        hit_id,
        turk_submit_to,
        required_votes,
        votes_done,
        presented,
        st,
        msg=msg,
        finished=False,
    )
    return JSONResponse(payload)


@app.post("/api/vote", response_class=JSONResponse)
def api_vote(req: VoteReq) -> JSONResponse:
    session_id = (req.session_id or "").strip()
    if not session_id:
        raise HTTPException(400, "missing session_id")

    session_id, slot, required_votes, votes_done, presented, local_index, st, csvp, worker_id, assignment_id, hit_id, turk_submit_to = _load_session(session_id)

    st, local_index, msg0 = _repair_state(slot, local_index, st)

    if votes_done >= required_votes:
        payload = _payload(
            SETTINGS,
            MANIFEST,
            len(PAIRS),
            session_id,
            assignment_id,
            worker_id,
            hit_id,
            turk_submit_to,
            required_votes,
            votes_done,
            presented,
            st,
            msg=msg0,
            finished=True,
        )
        return JSONResponse(payload)

    winner = int(req.winner)
    if winner not in (1, 2):
        try:
            _log_event("vote", session_id, f"winner={winner}")
        except Exception:
            pass
        raise HTTPException(400, "winner must be 1 (A) or 2 (B)")

    a_list = MANIFEST[st.a_file]
    b_list = MANIFEST[st.b_file]

    presented += 1
    votes_done += 1

    _append_row(
        csvp,
        st.a_file,
        int(st.a_chunk_idx) + 1,
        len(a_list),
        st.b_file,
        int(st.b_chunk_idx) + 1,
        len(b_list),
        winner,
        "0",
    )

    finished = votes_done >= required_votes
    msg = msg0

    if not finished:
        st, local_index = _advance(slot, local_index)
        st, local_index, msg2 = _repair_state(slot, local_index, st)
        if msg2:
            msg = (msg + "\n" if msg else "") + msg2
    else:
        msg = (msg + "\n" if msg else "") + "Completed. Thank you."

    _save_session(session_id, votes_done, presented, local_index, st)

    payload = _payload(
        SETTINGS,
        MANIFEST,
        len(PAIRS),
        session_id,
        assignment_id,
        worker_id,
        hit_id,
        turk_submit_to,
        required_votes,
        votes_done,
        presented,
        st,
        msg=msg,
        finished=finished,
    )
    return JSONResponse(payload)


@app.get("/api/status", response_class=JSONResponse)
def api_status(session_id: str = Query(default="")) -> JSONResponse:
    sid = session_id.strip()
    if sid:
        session_id, slot, required_votes, votes_done, presented, local_index, st, csvp, worker_id, assignment_id, hit_id, turk_submit_to = _load_session(sid)
    else:
        cur = DB.execute("SELECT session_id FROM sessions ORDER BY updated_ts DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return JSONResponse({"ok": False, "error": "no sessions", "dropped_pairs": DROPPED_PAIR_COUNT})
        session_id, slot, required_votes, votes_done, presented, local_index, st, csvp, worker_id, assignment_id, hit_id, turk_submit_to = _load_session(str(row[0]))

    exists = csvp.exists()
    lines = csvp.read_text(encoding="utf-8").splitlines() if exists else []
    return JSONResponse(
        {
            "ok": True,
            "session_id": session_id,
            "csv_path": str(csvp),
            "csv_exists": exists,
            "csv_lines": len(lines),
            "votes_done": int(votes_done),
            "required_votes": int(required_votes),
            "dropped_pairs": DROPPED_PAIR_COUNT,
            "preview_head": lines[:8],
        }
    )


@app.get("/api/storage", response_class=JSONResponse)
def api_storage(token: str = Query(default="")) -> JSONResponse:
    _require_admin_token(token)
    return JSONResponse(
        {
            "ok": True,
            "data_root": str(_data_root()),
            "db_path": SETTINGS.db_path,
            "output_sessions_dir": SETTINGS.output_sessions_dir,
            "sessions_meta_path": SETTINGS.sessions_meta_path,
        }
    )


@app.get("/api/sessions", response_class=JSONResponse)
def api_sessions(token: str = Query(default=""), limit: int = Query(default=200)) -> JSONResponse:
    _require_admin_token(token)
    root = Path(SETTINGS.output_sessions_dir)
    root.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(root.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True):
        st = p.stat()
        items.append({"name": p.name, "size": int(st.st_size), "mtime": float(st.st_mtime)})
        if len(items) >= max(1, int(limit)):
            break
    return JSONResponse({"ok": True, "dir": str(root), "count": len(items), "files": items})


@app.get("/api/download", response_class=FileResponse)
def api_download(sid: str = Query(default=""), name: str = Query(default=""), token: str = Query(default="")):
    _require_admin_token(token)
    sid = (sid or "").strip()
    name = (name or "").strip()
    if sid:
        p = _session_csv_path(SETTINGS, sid)
    elif name:
        p = Path(SETTINGS.output_sessions_dir) / Path(name).name
    else:
        raise HTTPException(400, "provide sid or name")
    if not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(p), media_type="text/csv", filename=p.name)


@app.get("/api/download_latest", response_class=FileResponse)
def api_download_latest(token: str = Query(default="")):
    _require_admin_token(token)
    root = Path(SETTINGS.output_sessions_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = list(root.glob("*.csv"))
    if not files:
        raise HTTPException(404, "no session csv files")
    p = max(files, key=lambda x: x.stat().st_mtime)
    return FileResponse(str(p), media_type="text/csv", filename=p.name)


def _cleanup_file(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


@app.get("/api/download_all.zip", response_class=FileResponse)
def api_download_all_zip(background_tasks: BackgroundTasks, token: str = Query(default="")):
    _require_admin_token(token)
    root = Path(SETTINGS.output_sessions_dir)
    root.mkdir(parents=True, exist_ok=True)
    files = list(root.glob("*.csv"))
    if not files:
        raise HTTPException(404, "no session csv files")

    tmp = tempfile.NamedTemporaryFile(prefix="sessions_", suffix=".zip", delete=False)
    tmp_path = tmp.name
    tmp.close()

    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(str(p), arcname=p.name)

    background_tasks.add_task(_cleanup_file, tmp_path)
    return FileResponse(tmp_path, media_type="application/zip", filename="sessions.zip")


from typing import Any


@app.get("/api/debug", response_class=JSONResponse)
def api_debug(token: str = Query(default="")) -> JSONResponse:
    _require_admin_token(token)
    keys = list(MANIFEST.keys())
    has_0908_manifest = any("speech_0908" in k for k in keys)
    has_0908_pairs = any(("speech_0908" in a) or ("speech_0908" in b) for a, b in PAIRS)
    return JSONResponse(
        {
            "ok": True,
            "data_root": str(_data_root()),
            "db_path": SETTINGS.db_path,
            "output_sessions_dir": SETTINGS.output_sessions_dir,
            "manifest_files": len(MANIFEST),
            "pairs_total": len(PAIRS),
            "dropped_pairs": DROPPED_PAIR_COUNT,
            "has_speech_0908_in_manifest": has_0908_manifest,
            "has_speech_0908_in_pairs": has_0908_pairs,
            "manifest_head": keys[:10],
            "pairs_head": PAIRS[:10],
        }
    )


@app.post("/api/admin/wipe", response_class=JSONResponse)
def api_admin_wipe(
    token: str = Query(default=""),
    delete_csv: int = Query(default=1),
    reset_db: int = Query(default=1),
) -> JSONResponse:
    _require_admin_token(token)
    global DB

    deleted = 0
    root = Path(SETTINGS.output_sessions_dir)
    root.mkdir(parents=True, exist_ok=True)

    if int(delete_csv) == 1:
        for fp in root.glob("*.csv"):
            try:
                fp.unlink()
                deleted += 1
            except Exception:
                pass

    if int(reset_db) == 1:
        try:
            DB.close()
        except Exception:
            pass

        dbp = Path(SETTINGS.db_path)
        for ext in ("", "-wal", "-shm"):
            try:
                Path(str(dbp) + ext).unlink(missing_ok=True)
            except Exception:
                pass

        DB = _db_connect(SETTINGS.db_path)
        _db_init(DB)
    else:
        DB.execute("DELETE FROM sessions")
        DB.execute("DELETE FROM meta")
        DB.commit()

    return JSONResponse(
        {
            "ok": True,
            "deleted_csv": deleted,
            "db_path": SETTINGS.db_path,
            "output_sessions_dir": str(root),
        }
    )


@app.get("/api/config_debug", response_class=JSONResponse)
def api_config_debug(token: str = Query(default="")) -> JSONResponse:
    _require_admin_token(token)
    env_cfg = os.getenv("MTURK_PREF_CONFIG", "").strip()
    cands = [
        env_cfg,
        "mturk_pref/config.json",
        "mturk_pref/settings.json",
        "config.json",
        "settings.json",
    ]
    out = []
    for c in cands:
        if not c:
            continue
        cp = Path(c)
        out.append({"path": c, "exists": cp.exists()})
    return JSONResponse(
        {
            "ok": True,
            "mturk_pref_config_env": env_cfg,
            "config_source_path": CONFIG_SOURCE_PATH,
            "comparisons_per_worker": int(SETTINGS.comparisons_per_worker),
            "total_comparisons": int(SETTINGS.total_comparisons) if SETTINGS.total_comparisons is not None else None,
            "candidates": out,
        }
    )


"__EVENTS_ENDPOINT_V1__"


@app.get("/api/events", response_class=JSONResponse)
def api_events(token: str = Query(default=""), limit: int = Query(default=200)) -> JSONResponse:
    _require_admin_token(token)
    lim = max(1, min(2000, int(limit)))
    cur = DB.execute(
        "SELECT ts, session_id, kind, detail FROM events ORDER BY ts DESC LIMIT ?",
        (lim,),
    )
    rows = cur.fetchall()
    out = []
    for ts, sid, kind, detail in rows:
        out.append(
            {
                "ts": float(ts),
                "session_id": str(sid or ""),
                "kind": str(kind or ""),
                "detail": str(detail or ""),
            }
        )
    return JSONResponse({"ok": True, "count": len(out), "events": out})


"__SESSIONS_DB_ENDPOINT_V1__"


@app.get("/api/sessions_db", response_class=JSONResponse)
def api_sessions_db(token: str = Query(default=""), limit: int = Query(default=200)) -> JSONResponse:
    _require_admin_token(token)
    lim = max(1, min(2000, int(limit)))
    cur = DB.execute(
        "SELECT session_id, updated_ts, votes_done, required_votes, worker_id, assignment_id, hit_id FROM sessions ORDER BY updated_ts DESC LIMIT ?",
        (lim,),
    )
    rows = cur.fetchall()
    out = []
    for sid, uts, vd, rv, wid, aid, hid in rows:
        out.append(
            {
                "session_id": str(sid or ""),
                "updated_ts": float(uts),
                "votes_done": int(vd or 0),
                "required_votes": int(rv or 0),
                "worker_id": str(wid or ""),
                "assignment_id": str(aid or ""),
                "hit_id": str(hid or ""),
            }
        )
    return JSONResponse({"ok": True, "count": len(out), "sessions": out})
