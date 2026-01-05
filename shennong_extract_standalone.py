#!/usr/bin/env python3
"""
Standalone Shennong feature extractor.
Usage:
    python shennong_extract_standalone.py path/to/audio.wav
"""

import io
import sys
import numpy as np
import soundfile as sf

try:
    from shennong import Audio, Pipeline
except Exception:
    print("❌ ERROR: Shennong is not installed.")
    print("Install with: pip install shennong")
    sys.exit(1)


def safe_mean(v):
    return float(np.mean(v)) if v is not None and len(v) else 0.0


def safe_std(v):
    return float(np.std(v)) if v is not None and len(v) else 0.0


def extract_shennong(audio_bytes: bytes):
    """Run Shennong pipelines on a raw WAV/MP3 audio file."""
    y, sr = sf.read(io.BytesIO(audio_bytes))

    if y.ndim > 1:
        y = y.mean(axis=1)

    audio = Audio(y, sr)
    duration = len(y) / sr

    ### ---- PITCH ----
    pitch_pipe = Pipeline("pitch")
    pitch_vals = pitch_pipe.process(audio).data.flatten()
    pitch_vals = pitch_vals[np.isfinite(pitch_vals)]
    pitch_vals = pitch_vals[pitch_vals > 0]

    ### ---- ENERGY ----
    energy_pipe = Pipeline("energy")
    energy_vals = energy_pipe.process(audio).data.flatten()
    energy_vals = energy_vals[np.isfinite(energy_vals)]

    ### ---- VOICE (jitter/shimmer) ----
    try:
        voice_pipe = Pipeline("voice")
        voice = voice_pipe.process(audio).data
        jitter = voice[:, 0]
        shimmer = voice[:, 1]
        jitter = jitter[np.isfinite(jitter)]
        shimmer = shimmer[np.isfinite(shimmer)]
    except Exception:
        jitter = np.array([])
        shimmer = np.array([])

    ### ---- FORMANTS ----
    try:
        form_pipe = Pipeline("formants")
        F = form_pipe.process(audio)
        f1 = F["F1"]
        f2 = F["F2"]
        f3 = F["F3"]
        f1 = f1[np.isfinite(f1)]
        f2 = f2[np.isfinite(f2)]
        f3 = f3[np.isfinite(f3)]
    except Exception:
        f1 = f2 = f3 = np.array([])

    return {
        "duration": duration,
        "pitch_mean": safe_mean(pitch_vals),
        "pitch_std": safe_std(pitch_vals),
        "energy_mean": safe_mean(energy_vals),
        "energy_std": safe_std(energy_vals),
        "jitter_local": safe_mean(jitter),
        "shimmer_local": safe_mean(shimmer),
        "formant1_mean": safe_mean(f1),
        "formant2_mean": safe_mean(f2),
        "formant3_mean": safe_mean(f3),
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python shennong_extract_standalone.py <audio_path>")
        sys.exit(1)

    audio_path = sys.argv[1]

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
    except Exception as e:
        print("❌ Could not load file:", e)
        sys.exit(1)

    feats = extract_shennong(audio_bytes)

    print("=== SHENNONG FEATURES ===")
    for k, v in feats.items():
        print(f"{k}: {v:.5f}")


if __name__ == "__main__":
    main()
