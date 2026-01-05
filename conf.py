import time
import json
import requests
import os

REVAI_TOKEN = "02xNrB6gZAGltmbTozl1qEM4WnOpn3HvvLcVAIglsBoHJMV60UBtUjM-_sFXkWPrKgWzJ3MV_QO86YbnEHe4Soe1Akgus"
BASE_URL = "https://api.rev.ai/speechtotext/v1"


def submit_job(audio_path: str) -> str:
    url = f"{BASE_URL}/jobs"
    headers = {"Authorization": f"Bearer {REVAI_TOKEN}"}
    files = {"media": open(audio_path, "rb")}

    r = requests.post(url, headers=headers, files=files)
    r.raise_for_status()
    return r.json()["id"]


def wait_for_job(job_id: str):
    url = f"{BASE_URL}/jobs/{job_id}"
    headers = {"Authorization": f"Bearer {REVAI_TOKEN}"}

    while True:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        status = r.json()["status"]

        if status == "transcribed":
            return
        if status == "failed":
            raise RuntimeError("Transcription failed.")

        time.sleep(2)


def get_results(job_id: str):
    url = f"{BASE_URL}/jobs/{job_id}/transcript"
    headers = {
        "Authorization": f"Bearer {REVAI_TOKEN}",
        "Accept": "application/vnd.rev.transcript.v1.0+json",
    }
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()


def compute_total_confidence(results_json):
    """Return ONE score: average word confidence."""
    monologues = results_json.get("monologues", [])
    if not monologues:
        return 0.0

    elements = monologues[0].get("elements", [])
    words = [e for e in elements if e.get("type") == "text"]

    if not words:
        return 0.0

    confidences = [w.get("confidence", 0) for w in words]
    return sum(confidences) / len(confidences)


def process_audio(path):
    job_id = submit_job(path)
    wait_for_job(job_id)
    results = get_results(job_id)
    score = compute_total_confidence(results)

    # Print ONLY the total score
    print(f"\n🎤 Speech Confidence Score: {score:.3f}\n")


if __name__ == "__main__":
    process_audio("converted_wav/media-interpretation.wav")
