from django.urls import include, path

from . import views

urlpatterns = [
    path("", views.indexd, name="indexd"),
    path("mdatos", views.mdatos, name="mdatos")
]
