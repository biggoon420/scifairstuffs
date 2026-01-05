import os
import io
import json
import time
import base64
import subprocess
from typing import List, Dict, Any

import numpy as np
import librosa
import soundfile as sf
import requests

try:
    import parselmouth
    from parselmouth.praat import call as praat_call
except ImportError:
    parselmouth = None
    praat_call = None



JWT = os.getenv(
    "VOICEGAIN_JWT",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiIxZjBlMGEwNS1hNzFiLTRjOTMtOGIyMS05OTBmOTJmNWNmZjgiLCJhdWQiOiJodHRwczovL2FwaS52b2ljZWdhaW4uYWkvdjEiLCJzdWIiOiJkOTdmOGUzMi1hYWUyLTQ1OTktYWJmYi04Y2NlYTJkMDlhOWQifQ.4gjkbr6FFqn1jDEvIWxUCYUNmS2u0_bhRtWqx77VOrc"
)

SONIOX_KEY = os.getenv(
    "SONIOX_KEY",
    "f6c24d41668c93a864a4fb08000271a7689359a2face9ff41e5472d89efecfe8"
)

ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_KEY", "90bcad6fec6741b981bc222f24006185")

VOICESENSE_USER = os.getenv("VOICESENSE_USER", "")
VOICESENSE_PASS = os.getenv("VOICESENSE_PASS", "")

INPUT_FOLDER = "input"
WAV_FOLDER = "converted_wav"

SR = 16000
CHUNK_SEC = 30.0
VOICEGAIN_URL = "https://api.voicegain.ai/v1/asr/transcribe"

FILLERS = ["um","uh","erm","er","uhh","umm","like","you know","sort of","kinda"]



def ensure_folders():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(WAV_FOLDER, exist_ok=True)



def convert_to_wav(input_path: str, output_path: str):
    cmd = ["ffmpeg","-y","-i",input_path,"-ac","1","-ar",str(SR),"-sample_fmt","s16",output_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(" FFmpeg FAILED:", r.stderr)
    else:
        print(f"✓ Converted → {output_path}")


def load_audio(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    if np.max(np.abs(y)) < 0.01 and np.max(np.abs(y)) > 0:
        print(" Normalizing quiet audio")
        y = y / np.max(np.abs(y))
    return y

def chunk_audio(y: np.ndarray, chunk_sec: float):
    hop = int(chunk_sec * SR)
    chunks = []
    for start in range(0, len(y), hop):
        c = y[start:start+hop]
        if len(c) > SR * 5:  # >5 sec
            chunks.append(c)
    return chunks

def wav_bytes(chunk: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, chunk, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ======================================================================
# VOICEGAIN STT
# ======================================================================

def transcribe_chunk(audio_bytes: bytes) -> str:
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    body = {"audio": {"source": {"inline": b64}}}

    headers = {
        "Authorization": f"Bearer {JWT}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        r = requests.post(VOICEGAIN_URL, headers=headers, data=json.dumps(body), timeout=60)
        d = r.json()
        alts = d.get("result", {}).get("alternatives", [])
        return alts[0].get("utterance","") if alts else ""
    except Exception as e:
        print(" Voicegain error:", e)
        return ""



def compute_voicegain_local_metrics(chunk: np.ndarray, text: str):
    duration = len(chunk) / SR

    rms = librosa.feature.rms(y=chunk)[0]
    RMS_mean = float(np.mean(rms))
    RMS_stability = float(np.std(rms))

    # Pitch
    try:
        f0 = librosa.yin(chunk, 80, 600, sr=SR)
        f0 = f0[np.isfinite(f0)]
        f0 = f0[f0>0]
    except:
        f0 = np.array([])

    if len(f0) < 5:
        try:
            f0p,_,_ = librosa.pyin(chunk,80,600,sr=SR)
            f0 = f0p[np.isfinite(f0p)]
            f0 = f0[f0>0]
        except:
            f0 = np.array([])

    if len(f0) > 2:
        pitch_micro = float(np.var(np.diff(f0)))
        pitch_macro = float(np.var(f0))
        intonation_slope = float(np.mean(f0[-10:]) - np.mean(f0[:10])) if len(f0)>20 else 0.0
    else:
        pitch_micro=pitch_macro=intonation_slope=0.0

    words = text.split()
    wpm = len(words)/duration*60 if duration>0 else 0

    lower = text.lower()
    filler_count = sum(lower.count(f) for f in FILLERS)
    fillers_per_min = filler_count/(duration/60) if duration>0 else 0

    # pauses
    thr = 0.05*np.max(rms) if np.max(rms)>0 else 0
    silent = rms<thr
    pauses=0
    in_pause=False
    hop = 512/SR
    start_t=0
    for i,s in enumerate(silent):
        t=i*hop
        if s and not in_pause:
            in_pause=True
            start_t=t
        elif not s and in_pause:
            in_pause=False
            if (t-start_t)>0.15:
                pauses+=1

    centroid = librosa.feature.spectral_centroid(y=chunk, sr=SR)[0]
    articulation = float(np.mean(centroid))

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



def call_soniox_chunk(wav_bytes_data: bytes, duration: float):
    """
    Soniox v3 async file API:
      1) POST /v1/files          (multipart, upload chunk)
      2) POST /v1/transcriptions (JSON, create transcription)
      3) Poll GET /v1/transcriptions/{id}
      4) GET  /v1/transcriptions/{id}/transcript  (tokens + confidence)
    Returns: {"duration": duration, "soniox_avg_confidence": <float>} or {} on error.
    """
    key = SONIOX_KEY
    if not key:
        return {}

    base_url = "https://api.soniox.com/v1"
    headers_auth = {"Authorization": f"Bearer {key}"}

    try:
        # 1) Upload file
        files = {
            "file": ("chunk.wav", io.BytesIO(wav_bytes_data), "audio/wav")
        }
        up = requests.post(
            f"{base_url}/files",
            headers=headers_auth,
            files=files,
            timeout=60
        )
        try:
            up_j = up.json()
        except Exception:
            print(" Soniox upload NON-JSON:", up.text[:200])
            return {}

        if "id" not in up_j:
            print("Soniox upload error:", up_j)
            return {}

        file_id = up_j["id"]

        # 2) Create transcription
        payload = {
            "file_id": file_id,
            "model": "stt-async-v3",  # current async model name
            # you can add language_hints if you want, e.g. ["en"]
            # "language_hints": ["en"],
        }
        tr = requests.post(
            f"{base_url}/transcriptions",
            headers={**headers_auth, "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60
        )

        try:
            tr_j = tr.json()
        except Exception:
            print("Soniox transcription NON-JSON:", tr.text[:200])
            return {}

        tid = tr_j.get("id")
        if not tid:
            print("Soniox create transcription error:", tr_j)
            return {}

        # 3) Poll until completed
        for _ in range(60):  # up to ~60s
            g = requests.get(
                f"{base_url}/transcriptions/{tid}",
                headers=headers_auth,
                timeout=30
            )
            gj = g.json()
            status = gj.get("status")
            if status == "completed":
                break
            if status == "error":
                print(" Soniox transcription failed:", gj.get("error_message"))
                # try to clean up file + transcription
                try:
                    requests.delete(
                        f"{base_url}/transcriptions/{tid}",
                        headers=headers_auth,
                        timeout=10
                    )
                except Exception:
                    pass
                try:
                    requests.delete(
                        f"{base_url}/files/{file_id}",
                        headers=headers_auth,
                        timeout=10
                    )
                except Exception:
                    pass
                return {}
            time.sleep(1)
        else:
            print(" Soniox transcription timeout")
            return {}

        # 4) Get transcript with tokens (for confidence)
        tt = requests.get(
            f"{base_url}/transcriptions/{tid}/transcript",
            headers=headers_auth,
            timeout=60
        )
        tt_j = tt.json()

        tokens = tt_j.get("tokens", [])
        confs = []

        # tokens have "confidence" field per docs
        for tok in tokens:
            c = tok.get("confidence")
            if c is not None:
                confs.append(c)

        avg_conf = float(np.mean(confs)) if confs else 0.0

        # Best-effort cleanup (doesn't matter if this fails)
        try:
            requests.delete(
                f"{base_url}/transcriptions/{tid}",
                headers=headers_auth,
                timeout=10
            )
        except Exception:
            pass
        try:
            requests.delete(
                f"{base_url}/files/{file_id}",
                headers=headers_auth,
                timeout=10
            )
        except Exception:
            pass

        return {
            "duration": duration,
            "soniox_avg_confidence": avg_conf
        }

    except Exception as e:
        print(" Soniox error:", e)
        return {}




def call_assemblyai_chunk(wav_bytes_data: bytes, duration: float):
    key = ASSEMBLYAI_KEY
    if not key:
        return {}

    headers={"authorization":key}

    try:
        # upload
        up = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, data=wav_bytes_data)
        uj = up.json()
        audio_url = uj.get("upload_url")
        if not audio_url:
            print(" AssemblyAI upload error:", uj)
            return {}

        # create transcript
        payload={
            "audio_url": audio_url,
            "disfluencies": True,
            "punctuate": True
        }

        tr = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            headers={**headers,"content-type":"application/json"},
            data=json.dumps(payload)
        )
        tj = tr.json()
        tid = tj.get("id")
        if not tid:
            print(" AssemblyAI create error:", tj)
            return {}

        # poll
        for _ in range(60):
            g=requests.get(f"https://api.assemblyai.com/v2/transcript/{tid}", headers=headers)
            gj=g.json()
            if gj.get("status")=="completed":
                break
            if gj.get("status") in ("error","failed"):
                print(" AssemblyAI failed:", gj)
                return {}
            time.sleep(2)
        else:
            print("AssemblyAI timeout")
            return {}

        words = gj.get("words",[]) or []

        # confidence
        confs=[w.get("confidence",0) for w in words]
        avg_conf=float(np.mean(confs)) if confs else 0

        # pauses
        pauses=0
        for w1,w2 in zip(words, words[1:]):
            if (w2.get("start",0)-w1.get("end",0))/1000 > 0.3:
                pauses+=1

        # filler rate
        filler_count=sum(1 for w in words if w.get("text","").lower() in FILLERS)
        fillers_per_min = filler_count/(duration/60) if duration>0 else 0

        wpm=len(words)/duration*60 if duration>0 else 0

        return {
            "duration": duration,
            "aa_avg_confidence": avg_conf,
            "aa_pauses": float(pauses),
            "aa_filler_rate": fillers_per_min,
            "aa_wpm": wpm
        }

    except Exception as e:
        print(" AssemblyAI chunk error:", e)
        return {}



def call_praat_chunk(wav_bytes_data: bytes, duration: float):
    """
    Expanded Praat metrics:
      - terminal_slope
      - jitter_local, shimmer_local
      - f0_mean, f0_std, f0_min, f0_max, f0_median
      - hnr_mean
      - cpp_mean
      - formants F1/F2/F3 mean & std
      - intensity_mean, intensity_std
      - voiced_percent
      - pitch_range
    """
    tmp = "tmp_praat_ext.wav"
    try:
        # write chunk to temp wav
        with open(tmp, "wb") as f:
            f.write(wav_bytes_data)

        snd = parselmouth.Sound(tmp)

        # PITCH
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]

        if len(f0_values) > 3:
            f0_mean  = float(np.mean(f0_values))
            f0_std   = float(np.std(f0_values))
            f0_min   = float(np.min(f0_values))
            f0_max   = float(np.max(f0_values))
            f0_median = float(np.median(f0_values))
            pitch_range = f0_max - f0_min
        else:
            f0_mean = f0_std = f0_min = f0_max = f0_median = pitch_range = 0.0

        # TERMINAL SLOPE
        # slope = mean(last 10 f0 frames) – mean(first 10 f0 frames)
        if len(f0_values) > 20:
            terminal_slope = float(np.mean(f0_values[-10:]) - np.mean(f0_values[:10]))
        else:
            terminal_slope = 0.0

        # HNR
        hnr = snd.to_harmonicity(time_step=0.01, minimum_pitch=75)
        hnr_mean = float(praat_call(hnr, "Get mean", 0, 0))

        # JITTER / SHIMMER
        pp = praat_call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter_local = float(praat_call(pp, "Get jitter (local)", 0,0,0.0001,0.02,1.3))
        shimmer_local = float(praat_call([snd, pp], "Get shimmer (local)",0,0,0.0001,0.02,1.3,1.6))

        # FORMANTS
        form = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5)
        F1, F2, F3 = [], [], []
        for t in np.arange(0, snd.duration, 0.01):
            try:
                F1.append(form.get_value_at_time(1, t))
                F2.append(form.get_value_at_time(2, t))
                F3.append(form.get_value_at_time(3, t))
            except:
                pass

        # clean NaNs/zeros
        def clean(vals):
            arr = np.array(vals)
            arr = arr[np.isfinite(arr)]
            arr = arr[arr > 0]
            return arr

        F1, F2, F3 = clean(F1), clean(F2), clean(F3)

        def stats(x):
            return (float(np.mean(x)), float(np.std(x))) if len(x) > 3 else (0.0, 0.0)

        F1_mean, F1_std = stats(F1)
        F2_mean, F2_std = stats(F2)
        F3_mean, F3_std = stats(F3)

        # INTENSITY
        intensity = snd.to_intensity(time_step=0.01)
        ints = intensity.values[0]
        ints = ints[np.isfinite(ints)]
        intensity_mean = float(np.mean(ints)) if len(ints) else 0.0
        intensity_std  = float(np.std(ints)) if len(ints) else 0.0

        # VOICED %
        voiced_frames = np.sum(f0_values > 0)
        voiced_percent = float(voiced_frames / len(f0_values)) if len(f0_values) else 0.0

        return {
            "duration": duration,
            "praat_terminal_slope": terminal_slope,
            "praat_jitter_local": jitter_local,
            "praat_shimmer_local": shimmer_local,
            "praat_f0_mean": f0_mean,
            "praat_f0_std": f0_std,
            "praat_f0_min": f0_min,
            "praat_f0_max": f0_max,
            "praat_f0_median": f0_median,
            "praat_pitch_range": pitch_range,
            "praat_hnr_mean": hnr_mean,
            "praat_intensity_mean": intensity_mean,
            "praat_intensity_std": intensity_std,
            "praat_formant1_mean": F1_mean,
            "praat_formant1_std": F1_std,
            "praat_formant2_mean": F2_mean,
            "praat_formant2_std": F2_std,
            "praat_formant3_mean": F3_mean,
            "praat_formant3_std": F3_std,
            "praat_voiced_percent": voiced_percent,
        }

    except Exception as e:
        print(" Praat EXT error:", e)
        return {}
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)





def call_covarep_chunk(wav_bytes_data: bytes, duration: float):
    if parselmouth is None:
        return {}

    tmp="tmp_cov.wav"
    try:
        with open(tmp,"wb") as f:
            f.write(wav_bytes_data)

        snd = parselmouth.Sound(tmp)
        pitch = snd.to_pitch(pitch_floor=75, pitch_ceiling=500)
        pp = praat_call(snd,"To PointProcess (periodic, cc)",75,500)

        jitter = praat_call(pp, "Get jitter (local)", 0,0,0.0001,0.02,1.3)
        shimmer = praat_call([snd,pp], "Get shimmer (local)",0,0,0.0001,0.02,1.3,1.6)
        hnr = snd.to_harmonicity(time_step=0.01, minimum_pitch=75)
        hnr_mean = praat_call(hnr,"Get mean",0,0)

        f0_mean = praat_call(pitch,"Get mean",0,0,"Hertz")
        f0_std  = praat_call(pitch,"Get standard deviation",0,0,"Hertz")

        return {
            "duration": duration,
            "cov_jitter_local": float(jitter),
            "cov_shimmer_local": float(shimmer),
            "cov_hnr_mean": float(hnr_mean),
            "cov_f0_mean": float(f0_mean),
            "cov_f0_std": float(f0_std)
        }

    except Exception as e:
        print(" Covarep error:", e)
        return {}
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)



def wavg(vals, weights):
    ws = weights
    return float(np.average(vals, weights=ws)) if sum(ws)>0 else 0

def summarize_api_chunks(chunks: List[Dict[str,Any]]):
    if not chunks:
        return {}
    durations=[c.get("duration",0) for c in chunks]
    metrics=set().union(*[c.keys() for c in chunks]) - {"duration","text"}

    out={}
    for m in metrics:
        vals=[c.get(m,0) for c in chunks]
        out[m]=wavg(vals, durations)
    return out



def process_file(path: str):
    print(f"\n=== Processing {path} ===")

    y = load_audio(path)
    chunks = chunk_audio(y, CHUNK_SEC)
    print("Chunks:", len(chunks))

    api_chunks = {
        "voicegain_local": [],
        "soniox": [],
        "assemblyai": [],
        "praat": [],
        "covarep": [],
    }

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1}/{len(chunks)} ---")
        audio_bytes = wav_bytes(chunk)

        text = transcribe_chunk(audio_bytes)
        vg = compute_voicegain_local_metrics(chunk, text)
        api_chunks["voicegain_local"].append(vg)

        print(f"Text: {vg['text'][:70]}")
        print(f"WPM={vg['wpm']:.1f}, Fillers/min={vg['fillers_per_min']:.2f}")

        dur = vg["duration"]

        sx = call_soniox_chunk(audio_bytes, dur)
        if sx: api_chunks["soniox"].append(sx)

        aa = call_assemblyai_chunk(audio_bytes, dur)
        if aa: api_chunks["assemblyai"].append(aa)

        pr = call_praat_chunk(audio_bytes, dur)
        if pr: api_chunks["praat"].append(pr)

        cv = call_covarep_chunk(audio_bytes, dur)
        if cv: api_chunks["covarep"].append(cv)

        time.sleep(0.2)


    # Summaries
    api_summaries={}
    for api, lst in api_chunks.items():
        if lst:
            summ=summarize_api_chunks(lst)
            api_summaries[api]=summ
            print(f"\n=== {api.upper()} SUMMARY ===")
            for k,v in summ.items():
                print(f"{k}: {v}")

    
    return api_summaries



def main():
    ensure_folders()
    files=[f for f in os.listdir(INPUT_FOLDER) if not f.startswith(".")]

    if not files:
        print("Put audio files in 'input/' folder.")
        return

    for f in files:
        in_f=os.path.join(INPUT_FOLDER,f)
        out_w=os.path.join(WAV_FOLDER, os.path.splitext(f)[0]+".wav")

        convert_to_wav(in_f, out_w)
        process_file(out_w)


if __name__=="__main__":
    main()
