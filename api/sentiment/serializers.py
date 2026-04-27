"""
Serializers for the Sentiment Analysis API.

Handles request validation and response formatting.
"""

from rest_framework import serializers


class SentimentRequestSerializer(serializers.Serializer):
    """
    Validates the incoming request for sentiment evaluation.

    Expected JSON body:
        {"sentence": "I love this product!"}
    """
    sentence = serializers.CharField(
        required=True,
        allow_blank=False,
        max_length=5000,
        help_text="The text to analyze for sentiment."
    )


class SentimentResponseSerializer(serializers.Serializer):
    """
    Formats the sentiment prediction response.

    Response JSON:
        {
            "sentiment": "positive",
            "confidence": 0.92,
            "probability": 0.92
        }

    If input is empty after preprocessing:
        {
            "sentiment": "neutral",
            "confidence": 0.0,
            "probability": 0.5,
            "error": "Input text is empty after preprocessing"
        }
    """
    sentiment = serializers.CharField(help_text="Predicted sentiment: positive, negative, or neutral")
    confidence = serializers.FloatField(help_text="Confidence score (0-1) for the prediction")
    probability = serializers.FloatField(help_text="Raw probability of positive class (0-1)")
    error = serializers.CharField(required=False, allow_null=True, help_text="Error message if input was invalid")
