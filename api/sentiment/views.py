"""
Views for the Sentiment Analysis API.

Provides the POST /evaluate endpoint that takes a sentence
and returns its sentiment prediction using the trained
Logistic Regression model.
"""

import sys
import os
import logging

from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .serializers import SentimentRequestSerializer, SentimentResponseSerializer

# Add the parent directory to Python path so we can import sentiment_model
sys.path.insert(0, str(settings.PROJECT_ROOT))
from sentiment_model.classifier import SentimentClassifier

logger = logging.getLogger(__name__)

# =============================================================================
# Model Loading (loaded once at module import, not per request)
# =============================================================================

# Initialize and load the Sentiment Classifier from our local library
classifier = SentimentClassifier()

try:
    classifier.load(settings.MODEL_DIR)
except Exception as e:
    logger.error("Failed to load sentiment model: %s", e)


# =============================================================================
# API Endpoint
# =============================================================================

@api_view(['POST'])
def evaluate(request):
    """
    Evaluate the sentiment of a sentence.

    POST /evaluate
    Request body: {"sentence": "I love this product!"}

    Returns:
        200: {"sentiment": "positive", "confidence": 0.92, "probability": 0.92}
        400: {"error": "..."} if input is invalid
        503: {"error": "..."} if model is not loaded
    """
    if not classifier.is_trained:
        return Response(
            {'error': 'Model not loaded. Please ensure model files exist in the models/ directory.'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    serializer = SentimentRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(
            {'error': serializer.errors},
            status=status.HTTP_400_BAD_REQUEST
        )

    sentence = serializer.validated_data['sentence']

    # Delegate the evaluation entirely to the SentimentClassifier!
    try:
        result = classifier.evaluate(sentence)
        response_serializer = SentimentResponseSerializer(result)
        return Response(response_serializer.data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
