from django.urls import path
from . import views

urlpatterns = [
    path("", views.indexp, name="indexp")
]