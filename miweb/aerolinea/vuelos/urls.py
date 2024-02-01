from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("<int:vuelo_id>", views.vuelo, name="vuelo"),
    path("<int:vuelo_id>/book", views.book, name="book")
]