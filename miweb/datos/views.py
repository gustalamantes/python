import json
from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.

def indexd(request):
    return HttpResponse("Hello, world. You're at the quickstartapp index.")

def mdatos(request):
    data = {'Player': 'Antoine griezmann','Team': 'Atlético de Madrid', 'Age': 25}
    return HttpResponse(json.dumps(data, indent=4, sort_keys=True), content_type="application/json")