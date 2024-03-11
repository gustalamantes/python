
'''
urlpatterns = [
    url(r'^create/$', CreateUserAPIView.as_view()),
]
'''

from django.urls import path
from .views import CreateUserAPIView

urlpatterns = [
    path('create/', CreateUserAPIView.as_view()),
    #path('login/', LoginUserAPIView.as_view()),
]