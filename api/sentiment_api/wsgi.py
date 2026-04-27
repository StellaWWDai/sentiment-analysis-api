"""
WSGI config for the Sentiment Analysis API.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sentiment_api.settings')
application = get_wsgi_application()
