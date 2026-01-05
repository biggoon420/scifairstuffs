import os, json, time, uuid
import requests
import boto3
import azure.cognitiveservices.speech as speechsdk

from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import (
    Features,
    EmotionOptions,
    SentimentOptions
)



# ---- AZURE SPEECH ----
AZURE_SPEECH_KEY = "EC3CsGttIv5Th5W2RmiByBCjJLT3qZogSYROdDewHEBUXhqLjfanJQQJ99BLACYeBjFXJ3w3AAAYACOGrCn3"
AZURE_SPEECH_REGION = "eastus"   # use the region your resource is in

# ---- BEHAVIORAL SIGNALS ----
BS_CID = 10000200                # <-- paste your CID (int)
BS_API_KEY = "6faa22310efeb468a5d6e1c96ce21dc8"

# ---- AWS ----
AWS_REGION = "us-west-2"
S3_BUCKET = "my-call-input"

# ---- IBM NLU ----
IBM_NLU_APIKEY = "Bd0Kg5-KHWfNoxAMkkJxdjpZD7snethQHl2B1HnccajY"
IBM_NLU_URL = "https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/08a1f0d3-dac4-4706-b643-7ed2c3fd6e5c"

# ---- AUDIO ----
AUDIO_FILE = "/Users/ayaanb/voicegain_confidence/input/media-interpretation.wav"



def azure_acoustic_confidence(audio_path: str) -> dict:
    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION
    )
    speech_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

    pa_config = speechsdk.PronunciationAssessmentConfig(
        reference_text="",
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=False
    )
    pa_config.enable_prosody_assessment()

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )
    pa_config.apply_to(recognizer)

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.Canceled:
        raise RuntimeError(result.cancellation_details.error_details)

    pa_json = result.properties.get(
        speechsdk.PropertyId.SpeechServiceResponse_JsonResult
    )
    data = json.loads(pa_json)
    scores = data["NBest"][0]["PronunciationAssessment"]

    return {
        "azure_pronunciation": scores.get("PronScore"),
        "azure_fluency": scores.get("FluencyScore"),
        "azure_prosody": scores.get("ProsodyScore"),
        "azure_accuracy": scores.get("AccuracyScore"),
        "azure_completeness": scores.get("CompletenessScore"),
    }



def s3_upload(local_path: str, bucket: str, key: str) -> str:
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"

def transcribe_audio(s3_uri: str) -> str:
    transcribe = boto3.client("transcribe", region_name=AWS_REGION)
    job_name = f"job-{uuid.uuid4().hex[:10]}"

    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": s3_uri},
        LanguageCode="en-US"
    )

    while True:
        job = transcribe.get_transcription_job(
            TranscriptionJobName=job_name
        )["TranscriptionJob"]

        status = job["TranscriptionJobStatus"]
        if status == "COMPLETED":
            break
        if status == "FAILED":
            raise RuntimeError(job["FailureReason"])
        time.sleep(5)

    uri = job["Transcript"]["TranscriptFileUri"]
    r = requests.get(uri)
    return r.json()["results"]["transcripts"][0]["transcript"]



BS_BASE = "https://api.behavioralsignals.com/v5"

def bs_headers():
    return {
        "accept": "application/json",
        "X-Auth-Client": str(BS_CID),
        "X-Auth-Token": BS_API_KEY,
    }

def behavioral_confidence(audio_path: str, max_wait=1500):
    with open(audio_path, "rb") as f:
        r = requests.post(
            f"{BS_BASE}/detection/clients/{BS_CID}/processes/audio",
            headers=bs_headers(),
            files={"file": ("audio.wav", f, "audio/wav")},
            timeout=60,
        )
    r.raise_for_status()
    pid = r.json()["pid"]
    print(f"Behavioral Signals job submitted → pid={pid}")

    start = time.time()
    last = None

    while time.time() - start < max_wait:
        s = requests.get(
            f"{BS_BASE}/clients/{BS_CID}/processes/{pid}",
            headers=bs_headers(),
            timeout=20,
        )
        status = s.json().get("status")

        if status != last:
            print(f"Behavioral Signals status → {status}")
            last = status

        if status == 3:  # completed
            time.sleep(5)
            res = requests.get(
                f"{BS_BASE}/clients/{BS_CID}/processes/{pid}/results",
                headers=bs_headers(),
                timeout=20,
            )
            if res.status_code == 200:
                return res.json()

        if status == 4:
            raise RuntimeError("Behavioral Signals failed")

        time.sleep(5)

    return {"pid": pid, "status": "processing"}



def ibm_tone(text: str) -> dict:
    """
    IBM Watson NLU-based confidence-related signals.
    Uses supported Emotion + Sentiment models.
    """

    authenticator = IAMAuthenticator(IBM_NLU_APIKEY)

    nlu = NaturalLanguageUnderstandingV1(
        version="2022-04-07",
        authenticator=authenticator
    )
    nlu.set_service_url(IBM_NLU_URL)

    response = nlu.analyze(
        text=text,
        language="en",
        features=Features(
            emotion=EmotionOptions(document=True),
            sentiment=SentimentOptions(document=True)
        )
    ).get_result()

    # ---- Extract signals cleanly ----
    emotion = response["emotion"]["document"]["emotion"]
    sentiment = response["sentiment"]["document"]

    return {
        # Emotion probabilities (0–1)
        "emotion_joy": emotion.get("joy"),
        "emotion_confidence_proxy": emotion.get("joy"),  # joy ~ vocal confidence
        "emotion_fear": emotion.get("fear"),
        "emotion_sadness": emotion.get("sadness"),
        "emotion_anger": emotion.get("anger"),
        "emotion_disgust": emotion.get("disgust"),

        # Sentiment
        "sentiment_score": sentiment.get("score"),   # -1 → 1
        "sentiment_label": sentiment.get("label"),

        # Simple derived confidence heuristic
        "ibm_confidence_score": round(
            (
                (emotion.get("joy", 0) * 0.6)
                + ((1 - emotion.get("fear", 0)) * 0.25)
                + ((sentiment.get("score", 0) + 1) / 2 * 0.15)
            ),
            3
        )
    }



def main():
    print("\n=== AZURE ACOUSTIC CONFIDENCE ===")
    azure_scores = azure_acoustic_confidence(AUDIO_FILE)
    print(json.dumps(azure_scores, indent=2))

    print("\n=== AWS TRANSCRIBE ===")
    s3_key = f"audio/{uuid.uuid4().hex}.wav"
    s3_uri = s3_upload(AUDIO_FILE, S3_BUCKET, s3_key)
    transcript = transcribe_audio(s3_uri)
    print("Transcript:", transcript[:200])

    print("\n=== BEHAVIORAL SIGNALS ===")
    bs_scores = behavioral_confidence(AUDIO_FILE)
    print(json.dumps(bs_scores, indent=2)[:2000])

    print("\n=== IBM NLU ===")
    ibm_scores = ibm_tone(transcript)
    print(json.dumps(ibm_scores, indent=2)[:2000])


if __name__ == "__main__":
    main()
