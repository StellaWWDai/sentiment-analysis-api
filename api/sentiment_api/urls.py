"""
URL configuration for the Sentiment Analysis API.
"""

from django.urls import path, include
from rest_framework.response import Response
from rest_framework.decorators import api_view


@api_view(['GET'])
def api_root(request):
    """API root — shows available endpoints."""
    return Response({
        'message': 'Sentiment Analysis API',
        'endpoints': {
            'POST /evaluate': 'Evaluate the sentiment of a sentence',
        },
        'example': {
            'request': {'sentence': 'I love this product!'},
            'response': {
                'sentiment': 'positive',
                'confidence': 0.92,
                'probability': 0.92
            }
        }
    })


urlpatterns = [
    path('', api_root, name='api-root'),
    path('', include('sentiment.urls')),
]
