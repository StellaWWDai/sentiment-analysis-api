"""
URL routing for the sentiment app.
"""

from django.urls import path
from . import views

urlpatterns = [
    path('evaluate', views.evaluate, name='evaluate'),
]
