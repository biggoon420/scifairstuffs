#!/usr/bin/env python3
"""
Extract audio features using Shennong and write to CSV (one row per audio file).
Usage:
    python shennong_to_csv.py /path/to/audio1.wav /path/to/audio2.wav ... --out output.csv
"""

import sys, os
import csv
import soundfile as sf
import numpy as np

try:
    from shennong import Audio, Pipeline
except ImportError:
    print("Error: Shennong not installed or failed to import.")
    sys.exit(1)

def safe_mean(arr):
    return float(np.mean(arr)) if arr is not None and len(arr) else 0.0

def safe_std(arr):
    return float(np.std(arr)) if arr is not None and len(arr) else 0.0

def extract_features(filepath):
    y, sr = sf.read(filepath)
    if y.ndim > 1:
        y = y.mean(axis=1)
    duration = len(y) / sr
    audio = Audio(y, sr)

    feats = {"file": os.path.basename(filepath), "duration": duration}

    # PITCH
    pitch = Pipeline("pitch").process(audio).data.flatten()
    pitch = pitch[np.isfinite(pitch)]
    pitch = pitch[pitch > 0]
    feats.update({
        "pitch_mean": safe_mean(pitch),
        "pitch_std": safe_std(pitch),
    })

    # ENERGY
    energy = Pipeline("energy").process(audio).data.flatten()
    energy = energy[np.isfinite(energy)]
    feats.update({
        "energy_mean": safe_mean(energy),
        "energy_std": safe_std(energy),
    })

    # VOICE (jitter/shimmer)
    try:
        voice = Pipeline("voice").process(audio).data
        jitter = voice[:,0]
        shimmer = voice[:,1]
        feats["jitter_mean"] = safe_mean(jitter[np.isfinite(jitter)])
        feats["shimmer_mean"] = safe_mean(shimmer[np.isfinite(shimmer)])
    except Exception:
        feats["jitter_mean"] = 0.0
        feats["shimmer_mean"] = 0.0

    # FORMANTS
    try:
        F = Pipeline("formants").process(audio)
        for f in ["F1","F2","F3"]:
            vals = F.get(f, [])
            vals = vals[np.isfinite(vals)] if len(vals) else []
            feats[f.lower() + "_mean"] = safe_mean(vals)
    except Exception:
        feats["f1_mean"] = feats["f2_mean"] = feats["f3_mean"] = 0.0

    return feats

def main():
    import argparse
    p = argparse.ArgumentParser(description="Extract Shennong features -> CSV")
    p.add_argument("audio_files", nargs="+", help="Paths to audio files")
    p.add_argument("--out", default="shennong_features.csv", help="Output CSV path")
    args = p.parse_args()

    rows = []
    for f in args.audio_files:
        try:
            row = extract_features(f)
        except Exception as e:
            print("Error on", f, e)
            continue
        rows.append(row)

    if not rows:
        print("No features extracted.")
        return

    # write CSV
    keys = sorted(rows[0].keys())
    with open(args.out, "w", newline="") as csvfile:
        w = csv.DictWriter(csvfile, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("Wrote", len(rows), "rows to", args.out)

if __name__ == "__main__":
    main()
