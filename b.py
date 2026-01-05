from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions, SentimentOptions

IBM_API_KEY = "Bd0Kg5-KHWfNoxAMkkJxdjpZD7snethQHl2B1HnccajY"
IBM_URL = "https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/08a1f0d3-dac4-4706-b643-7ed2c3fd6e5c"

authenticator = IAMAuthenticator(IBM_API_KEY)
nlu = NaturalLanguageUnderstandingV1(
    version="2022-04-07",
    authenticator=authenticator
)
nlu.set_service_url(IBM_URL)

def ibm_confidence(text: str):
    response = nlu.analyze(
        text=text,
        features=Features(
            emotion=EmotionOptions(),
            sentiment=SentimentOptions()
        )
    ).get_result()

    emotion = response["emotion"]["document"]["emotion"]
    sentiment = response["sentiment"]["document"]

    return {
        # Negative confidence indicators
        "fear": emotion.get("fear", 0.0),
        "sadness": emotion.get("sadness", 0.0),

        # Assertive but unstable
        "anger": emotion.get("anger", 0.0),

        # Positive confidence proxies
        "joy": emotion.get("joy", 0.0),
        "sentiment_score": sentiment.get("score", 0.0),
        "sentiment_label": sentiment.get("label")
    }


# 🧪 TEST
if __name__ == "__main__":
    text = """
    I know exactly what I am doing.
    This decision is correct and I am fully confident.
    """

    print("=== IBM CONFIDENCE (REAL OUTPUT) ===")
    print(ibm_confidence(text))
