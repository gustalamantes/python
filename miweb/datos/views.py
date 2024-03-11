from django.http import HttpResponse
from django.shortcuts import render

#import warnings; 
#warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
from json import loads, dumps
from rest_framework.views import APIView
from rest_framework import serializers
from.models import User
from rest_framework.permissions import AllowAny 
from rest_framework.response import Response


# Create your views here.
def indexd(request):
    return render(request, "datos/indexd.html", {
        "datos" : "Hola",
        "fecha": "Fecha de impresion: " 
    })

#def indexd(request):
#    return HttpResponse("Hello, world. You're at the quickstartapp index.")

def mdatos(request):
    df = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    result = df.to_json(orient="split")
    data = loads(result) 
    #data = {'Player': 'Antoine griezmann','Team': 'Atlético de Madrid', 'Age': 25}
    return HttpResponse(dumps(data, indent=4, sort_keys=True), content_type="application/json")


#from rest_framework.response import Response
#from rest_framework import status
#from rest_framework.permissions import AllowAny 


# Create your views here.

class mdata(APIView):
    # Allow any user (authenticated or not) to access this url 
    permission_classes = (AllowAny,)
    def post(self, request):
        user = request.data
        if user:
            df = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
            result = df.to_json(orient="split")
            data = loads(result) 
            return HttpResponse(dumps(data, indent=4, sort_keys=True), content_type="application/json")

        #serializer = UserSerializer(data=user)
        #serializer.is_valid(raise_exception=True)
        #serializer.save()
        
        