from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, "hello/index.html")

def gus(request):
    return HttpResponse("Hola Gus!")

def saludo(request, name):
    return render(request, "hello/saludo.html", {
        "name": name.capitalize()})
