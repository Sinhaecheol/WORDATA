from django.shortcuts import render, get_object_or_404
from analysis_text.views import wordata
from analysis_text.models import Userinput

#처음 뜨는 메인페이지
def home(request):
    return render(request, 'main/index.html')

def login(request):
    return render(request, 'main/login.html')

def signup(request):
    return render(request, 'main/signup.html')
