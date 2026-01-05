import os
import io
import json
import time
import base64
import subprocess
import csv
from typing import List, Dict, Any

import numpy as np
import librosa
import soundfile as sf
import requests

# Try to import parselmouth (Praat in Python)
try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    HAS_PARSEL = True
except ImportError:
    HAS_PARSEL = False
    print("⚠️ parselmouth not installed. Praat metrics will be skipped.")

# ======================================================================
# CONFIG
# ======================================================================

# Voicegain JWT – better to set as env var:
#   export VOICEGAIN_JWT="..."
JWT = os.getenv(
    "VOICEGAIN_JWT",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiIxZjBlMGEwNS1hNzFiLTRjOTMtOGIyMS05OTBmOTJmNWNmZjgiLCJhdWQiOiJodHRwczovL2FwaS52b2ljZWdhaW4uYWkvdjEiLCJzdWIiOiJkOTdmOGUzMi1hYWUyLTQ1OTktYWJmYi04Y2NlYTJkMDlhOWQifQ.4gjkbr6FFqn1jDEvIWxUCYUNmS2u0_bhRtWqx77VOrc"  # <-- replace or set env var
)

INPUT_FOLDER = "input"
WAV_FOLDER = "converted_wav"

SR = 16000
CHUNK_SEC = 30.0

VOICEGAIN_URL = "https://api.voicegain.ai/v1/asr/transcribe"

# Optional external APIs (leave as "YOUR_..." to disable safely)
SPEECHACE_KEY  = os.getenv("SPEECHACE_KEY", "d11wwwJu%2BRldnF2k6KIivFGtHz1I3MAz61Mk%2FWEz6ZWURLXyI5ko5EDtS%2FDR4A%2BwbU5MbKbxYPe4C378VoH2BSRQuHztCQdNlM4lpXejfHRAE6UfbKY%2FgjCetrnp9QLH")
SONIOX_KEY     = os.getenv("SONIOX_KEY", "f6c24d41668c93a864a4fb08000271a7689359a2face9ff41e5472d89efecfe8")
VOICESENSE_USER = os.getenv("VOICESENSE_USER", "")  # VoiceSense usually uses username/password + CLI/API
VOICESENSE_PASS = os.getenv("VOICESENSE_PASS", "")

FILLERS = ["um", "uh", "erm", "er", "uhh", "umm", "like", "you know", "sort of", "kinda"]

# ======================================================================
# FOLDERS
# ======================================================================

def ensure_folders():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(WAV_FOLDER, exist_ok=True)

# ======================================================================
# CONVERSION TO WAV (FFmpeg)
# ======================================================================

def convert_to_wav(input_path: str, output_path: str):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", str(SR),
        "-sample_fmt", "s16",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n❌ FFmpeg FAILED for {input_path}")
        print(result.stderr)
    else:
        print(f"✓ FFmpeg conversion OK: {output_path}")

# ======================================================================
# LOAD AUDIO
# ======================================================================

def load_audio(path: str, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    if np.max(np.abs(y)) < 0.01:
        print("⚠️ Audio very quiet — normalizing.")
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
    return y

# ======================================================================
# CHUNKING
# ======================================================================

def chunk_audio(y: np.ndarray, sr: int, chunk_sec: float) -> List[np.ndarray]:
    length = int(chunk_sec * sr)
    chunks = []
    for start in range(0, len(y), length):
        end = min(len(y), start + length)
        c = y[start:end]
        if len(c) > sr * 5:  # require > 5 seconds
            chunks.append(c)
    return chunks

# ======================================================================
# WAV BYTES FROM NUMPY
# ======================================================================

def wav_bytes(chunk: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, chunk, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

# ======================================================================
# VOICEGAIN STT
# ======================================================================

def transcribe_chunk(wav_bytes_data: bytes) -> str:
    b64 = base64.b64encode(wav_bytes_data).decode("ascii")

    body = {
        "audio": {"source": {"inline": b64}}
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {JWT}"
    }

    resp = requests.post(VOICEGAIN_URL, headers=headers, data=json.dumps(body))

    try:
        data = resp.json()
    except Exception:
        print("❌ Non-JSON response from Voicegain:", resp.text[:200])
        return ""

    if "result" not in data:
        print("❌ Unexpected Voicegain response:", json.dumps(data)[:250])
        return ""

    alts = data["result"].get("alternatives", [])
    if not alts:
        return ""
    return alts[0].get("utterance", "") or ""

# ======================================================================
# LOCAL / VOICEGAIN-BASED METRICS (YOUR ORIGINAL)
# ======================================================================

def compute_voicegain_local_metrics(chunk: np.ndarray, sr: int, text: str) -> Dict[str, Any]:
    duration = len(chunk) / sr

    # RMS
    rms = librosa.feature.rms(y=chunk)[0]
    RMS_mean = float(np.mean(rms))
    RMS_stability = float(np.std(rms))

    # F0 / pitch (micro & macro stability)
    try:
        f0 = librosa.yin(chunk, fmin=80, fmax=600, sr=sr)
        f0 = f0[np.isfinite(f0)]
        f0 = f0[f0 > 0]
    except Exception:
        f0 = np.array([])

    if len(f0) < 5:
        try:
            f0_py, _, _ = librosa.pyin(chunk, fmin=80, fmax=600, sr=sr)
            f0_py = f0_py[np.isfinite(f0_py)]
            f0_py = f0_py[f0_py > 0]
            f0 = f0_py
        except Exception:
            f0 = np.array([])

    if len(f0) > 2:
        pitch_micro = float(np.var(np.diff(f0)))  # micro jitters/peaks
        pitch_macro = float(np.var(f0))           # global contour variance
        intonation_slope = float(np.mean(f0[-10:]) - np.mean(f0[:10])) if len(f0) > 20 else 0.0
    else:
        pitch_micro = 0.0
        pitch_macro = 0.0
        intonation_slope = 0.0

    # Speech rate
    words = text.split()
    wpm = float(len(words) / duration * 60.0) if (len(words) > 0 and duration > 0) else 0.0

    # Fillers
    lower = text.lower()
    filler_count = sum(lower.count(f) for f in FILLERS)
    fillers_per_min = filler_count / (duration / 60.0) if duration > 0 else 0.0

    # Pauses based on energy
    thr = 0.05 * np.max(rms) if np.max(rms) > 0 else 0
    silent = rms < thr
    frame_hop = 512 / sr
    pauses = 0
    in_pause = False
    start = 0.0

    for i, s in enumerate(silent):
        t = i * frame_hop
        if s and not in_pause:
            in_pause = True
            start = t
        elif not s and in_pause:
            in_pause = False
            dur = t - start
            if dur > 0.15:
                pauses += 1

    # Articulation proxy: spectral centroid
    sc = librosa.feature.spectral_centroid(y=chunk, sr=sr)[0]
    articulation = float(np.mean(sc))

    return {
        "duration": duration,
        "RMS_mean": RMS_mean,
        "RMS_stability": RMS_stability,
        "pitch_micro": pitch_micro,
        "pitch_macro": pitch_macro,
        "intonation_slope": intonation_slope,
        "wpm": wpm,
        "fillers_per_min": fillers_per_min,
        "total_fillers": float(filler_count),
        "total_pauses": float(pauses),
        "articulation": articulation,
        "text": text
    }

# ======================================================================
# SPEECHACE (PRONUNCIATION / ARTICULATION)
# ======================================================================

def call_speechace_chunk(wav_bytes_data: bytes, duration: float) -> Dict[str, Any]:
    """
    SpeechAce pronunciation/fluency score for the chunk.
    Docs: https://docs.speechace.com/
    """
    if not SPEECHACE_KEY or SPEECHACE_KEY.startswith("YOUR_"):
        return {}

    try:
        base_url = "https://api.speechace.co/api/scoring/text/v9/json"
        params = {
            "key": SPEECHACE_KEY,
            "dialect": "en-us",
            "user_id": "chunk-user"
        }
        data = {"text": "dummy"}  # replace with recognized text if you want
        files = {"user_audio_file": ("chunk.wav", wav_bytes_data, "audio/wav")}

        r = requests.post(base_url, params=params, data=data, files=files, timeout=20)
        d = r.json()

        speech_score = d.get("speech_score", {})
        sa_score = speech_score.get("speechace_score", {})

        return {
            "duration": duration,
            "sa_pronunciation": float(sa_score.get("pronunciation", 0.0)),
            "sa_fluency": float(sa_score.get("fluency", 0.0)),
            "sa_overall": float(sa_score.get("overall", 0.0))
        }
    except Exception as e:
        print("SpeechAce chunk error:", e)
        return {}

# ======================================================================
# SONIOX (CONFIDENCE / FORMANT STABILITY – STUB)
# ======================================================================

def call_soniox_chunk(wav_bytes_data: bytes, duration: float) -> Dict[str, Any]:
    """
    Soniox STT + confidence (token-level). Docs: https://soniox.com/docs/stt/concepts/confidence-scores
    Exact HTTP endpoint & auth depend on your account; this is a stub.
    """
    if not SONIOX_KEY or SONIOX_KEY.startswith("YOUR_"):
        return {}

    # TODO: Implement real Soniox API call here once you have their exact REST/WebSocket spec.
    # Keep return shape so the pipeline doesn't break:
    return {
        # e.g. "soniox_mean_confidence": 0.0
    }

# ======================================================================
# VOICESENSE (GLOBAL CONFIDENCE – STUB)
# ======================================================================

def call_voicesense_chunk(wav_bytes_data: bytes, duration: float) -> Dict[str, Any]:
    """
    VoiceSense typically uses a 2-step 'upload audio' then 'predict scores' flow.
    Docs: https://apim-developer.voicesense.com
    """
    if not VOICESENSE_USER or not VOICESENSE_PASS:
        return {}

    # TODO: Implement using their upload + predictor endpoints.
    return {
        # e.g. "voicesense_confidence": 0.0
    }

# ======================================================================
# PRAAT (INTONATION / JITTER / SHIMMER via parselmouth)
# ======================================================================

def call_praat_chunk(wav_bytes_data: bytes, duration: float) -> Dict[str, Any]:
    """
    Uses parselmouth (Praat) directly in Python to get:
      - praat_terminal_slope   (intonation slope)
      - praat_f0_mean
      - praat_f0_std
      - praat_jitter_local
      - praat_shimmer_local
    """
    if not HAS_PARSEL:
        return {}

    try:
        # Load from bytes into numpy
        buf = io.BytesIO(wav_bytes_data)
        y, sr = sf.read(buf)
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        snd = parselmouth.Sound(y, sr)
        pitch = snd.to_pitch(time_step=0.01)

        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]

        if len(f0) < 10:
            f0_mean = 0.0
            f0_std = 0.0
            slope = 0.0
        else:
            f0_mean = float(np.mean(f0))
            f0_std  = float(np.std(f0))
            xs = np.arange(len(f0))
            slope = float(np.polyfit(xs, f0, 1)[0])

        # Jitter & shimmer
        pointProcess = praat_call(snd, "To PointProcess (periodic, cc)", 75, 600)
        jitter_local = praat_call(pointProcess, "Get jitter (local)", 0, 0, 75, 600, 1.3)
        shimmer_local = praat_call([snd, pointProcess], "Get shimmer (local)", 0, 0, 75, 600, 1.3, 1.6)

        return {
            "duration": duration,
            "praat_terminal_slope": slope,   # keeps your original key idea
            "praat_f0_mean": f0_mean,
            "praat_f0_std": f0_std,
            "praat_jitter_local": jitter_local,
            "praat_shimmer_local": shimmer_local
        }
    except Exception as e:
        print("Praat chunk error:", e)
        return {}

# ======================================================================
# COVAREP (JITTER / SHIMMER / HNR – PLACEHOLDER)
# ======================================================================

def call_covarep_chunk(wav_bytes_data: bytes, duration: float) -> Dict[str, Any]:
    """
    Real COVAREP integration requires the covarep toolbox / Python wrapper.
    Doc: http://covarep.github.io/covarep/
    This is left as a stub so the pipeline still runs.
    """
    # TODO: integrate real COVAREP if you set up their Python bindings.
    return {
        # e.g. "covarep_HNR": 0.0,
        #      "covarep_NAQ": 0.0,
        #      "covarep_CQ": 0.0
    }

# ======================================================================
# WEIGHTED AVERAGING
# ======================================================================

def wavg(values, weights):
    vals = [v for v, w in zip(values, weights)]
    ws = [w for v, w in zip(values, weights)]
    if sum(ws) == 0:
        return 0.0
    return float(np.average(vals, weights=ws))

def summarize_api_chunks(api_name: str, chunk_results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not chunk_results:
        return {}
    durations = [r.get("duration", 0.0) for r in chunk_results]
    metric_keys = set().union(*[set(r.keys()) for r in chunk_results]) - {"duration", "text"}
    summary = {}
    for k in metric_keys:
        vals = [r.get(k, 0.0) for r in chunk_results]
        summary[k] = wavg(vals, durations)
    return summary

def average_across_apis(api_summaries: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not api_summaries:
        return {}
    all_keys = set().union(*[s.keys() for s in api_summaries.values()])
    out = {}
    for k in all_keys:
        vals = [s[k] for s in api_summaries.values() if k in s]
        if vals:
            out[k] = float(np.mean(vals))
    return out

# ======================================================================
# PROCESS A SINGLE WAV FILE
# ======================================================================

def process_file(wav_path: str):
    print(f"\n=== Processing: {wav_path} ===")

    y = load_audio(wav_path, SR)
    chunks = chunk_audio(y, SR, CHUNK_SEC)
    print(f"Chunks: {len(chunks)}")

    api_chunks: Dict[str, List[Dict[str, Any]]] = {
        "voicegain_local": [],
        "speechace": [],
        "soniox": [],
        "voicesense": [],
        "praat": [],
        "covarep": []
    }

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1}/{len(chunks)} ---")

        data = wav_bytes(chunk, SR)

        # Voicegain + local metrics
        text = transcribe_chunk(data)
        vg = compute_voicegain_local_metrics(chunk, SR, text)
        api_chunks["voicegain_local"].append(vg)

        print(f"Text: {vg['text'][:80]}")
        print(f"RMS_mean={vg['RMS_mean']:.4f}, WPM={vg['wpm']:.2f}, "
              f"fillers/min={vg['fillers_per_min']:.2f}, pauses={vg['total_pauses']}")

        duration = vg["duration"]

        # SpeechAce pronunciation
        sa = call_speechace_chunk(data, duration)
        if sa:
            api_chunks["speechace"].append(sa)

        # Soniox (stub)
        sx = call_soniox_chunk(data, duration)
        if sx:
            api_chunks["soniox"].append(sx)

        # VoiceSense (stub)
        vs = call_voicesense_chunk(data, duration)
        if vs:
            api_chunks["voicesense"].append(vs)

        # Praat (parselmouth)
        pr = call_praat_chunk(data, duration)
        if pr:
            api_chunks["praat"].append(pr)

        # COVAREP (stub)
        cv = call_covarep_chunk(data, duration)
        if cv:
            api_chunks["covarep"].append(cv)

        # Small delay to be nice to APIs
        time.sleep(0.2)

    # Per-API summaries
    api_summaries: Dict[str, Dict[str, float]] = {}
    for api_name, chunks_list in api_chunks.items():
        if chunks_list:
            summary = summarize_api_chunks(api_name, chunks_list)
            api_summaries[api_name] = summary
            print(f"\n=== {api_name.upper()} SUMMARY ===")
            for k, v in summary.items():
                print(f"{k}: {v}")

    # Final averaged metrics
    final_avg = average_across_apis(api_summaries)

    print("\n==============================")
    print("  FINAL AVERAGED METRICS ACROSS APIS")
    print("==============================")
    for k, v in final_avg.items():
        print(f"{k}: {v}")

    return api_summaries, final_avg

# ======================================================================
# MAIN
# ======================================================================

def main():
    ensure_folders()
    files = [f for f in os.listdir(INPUT_FOLDER) if not f.startswith(".")]

    if not files:
        print("Place files inside the 'input/' folder.")
        return

    for f in files:
        in_f = os.path.join(INPUT_FOLDER, f)
        out_w = os.path.join(WAV_FOLDER, os.path.splitext(f)[0] + ".wav")

        print(f"\nConverting {f} → WAV")
        convert_to_wav(in_f, out_w)
        process_file(out_w)

if __name__ == "__main__":
    main()
