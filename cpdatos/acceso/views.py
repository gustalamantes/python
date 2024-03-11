from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, get_user_model
from rest_framework import HTTP_HEADER_ENCODING, exceptions

import pandas as pd
from json import loads, dumps


def index(request):
    user = request.user
    #act = user.is_authenticated
    #print(act)
    '''
    if user is None:
        raise exceptions.AuthenticationFailed('Invalid username/password.')    
    if not user.is_active:
        raise exceptions.AuthenticationFailed('User inactive or deleted.')
    if not user.is_authenticated:
        raise exceptions.AuthenticationFailed('User no firmado.')
    if not request.session:
        raise exceptions.AuthenticationFailed('Invalid session.')
    '''
    if user is None:
        return HttpResponse("Invalid username/password.")    
    if not user.is_active:
        return HttpResponse("User inactive or deleted.")
    if not user.is_authenticated:
        return HttpResponse("User no firmado.")
    if not request.session:
        return HttpResponse("Invalid session.")
    
    muser = str(user)
    print(muser)
    if muser == "gus" or muser == "gus2":
        arch = "./acceso/static/acceso/mpg_ggplot2.csv" if muser == "gus"  else "./acceso/static/acceso/mtcars.csv"
        df = pd.read_csv(arch)
        result = df.to_json(orient="split")
        data = loads(result) 
        return HttpResponse(dumps(data, indent=4, sort_keys=True), content_type="application/json")
    else:
        return HttpResponse("Acceso no permitido")

