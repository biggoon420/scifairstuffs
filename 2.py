"""
Minimal tester for SwiftF0 + Rev.ai SpeechAnalytics.
Run:  python -m src.factors.test <audio_path>
"""

import os
import sys
import base64
import json
import requests
import numpy as np
import soundfile as sf


def call_swiftf0_chunk_local(chunk, sr, duration):
    """Test SwiftF0 using the fields present in PitchResult."""
    try:
        from swift_f0 import SwiftF0
        model = SwiftF0()
        result = model.detect_from_array(chunk, sr)
        if hasattr(result, "pitch_hz"):
            f0 = np.array(result.pitch_hz, dtype=float)
        else:
            return {}
        f0 = f0[np.isfinite(f0)]
        f0 = f0[f0 > 0]
        if f0.size == 0:
            return {}
        return {
            "duration": duration,
            "swiftf0_mean": float(np.mean(f0)),
            "swiftf0_std": float(np.std(f0)),
            "swiftf0_min": float(np.min(f0)),
            "swiftf0_max": float(np.max(f0)),
            "swiftf0_range": float(np.max(f0) - np.min(f0)),
        }
    except Exception:
        return {}


def call_revai_chunk(wav_bytes_data, duration, key):
    """Rev.ai SpeechAnalytics using standard TLS."""
    if not key:
        return {}
    tmp = "tmp_rev_chunk.wav"
    try:
        with open(tmp, "wb") as f:
            f.write(wav_bytes_data)
        url = "https://api.rev.ai/speechanalytics/analyze"
        headers = {"Authorization": f"Bearer {key}"}
        files = {"media": open(tmp, "rb")}
        r = requests.post(url, headers=headers, files=files, timeout=60)
        if r.status_code != 200:
            print("Rev.ai status:", r.status_code)
            print("Rev.ai error:", r.text[:500])
            return {}
        J = r.json()
        m = J.get("acoustic_metrics", {}) or {}
        return {
            "duration": duration,
            "rev_speaking_rate": float(m.get("speaking_rate", 0.0)),
            "rev_pause_frequency": float(m.get("pause_frequency", 0.0)),
            "rev_loudness": float(m.get("loudness", 0.0)),
            "rev_articulation": float(m.get("articulation", 0.0)),
        }
    except Exception as e:
        print("EXCEPTION:", type(e).__name__, str(e))
        return {}
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def load_wav_bytes(path):
    """Load audio file as valid PCM16 WAV."""
    import soundfile as sf
    import io, wave

    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)

    duration = len(y) / sr if sr > 0 else 0.0

    y = np.clip(y, -1.0, 1.0)
    y16 = (y * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y16.tobytes())

    return buf.getvalue(), y, sr, duration


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.factors.test <audio_path>")
        sys.exit(1)
    wav_path = sys.argv[1]
    wav_bytes, y, sr, duration = load_wav_bytes(wav_path)
    swift = call_swiftf0_chunk_local(y, sr, duration)
    print("SwiftF0:", swift)
    revai_key = os.getenv("REVAI_API_KEY", "02znsSWd3AOpbKwjRUMChVmVo8bGjLkmVy_TVFR-B0Xbh_eWKUXlDmHFtQ5PjV3U4KxT2-nx9Fho8HUBxQllkYWXZchMg")
    rev = call_revai_chunk(wav_bytes, duration, revai_key)
    print("Rev.ai:", rev)
