from django.urls import path
from . import views

app_name = 'analysis_text'
urlpatterns = [
    path('userinputform/', views.userinputform, name='userinputform'),
    path('form_list/', views.form_list, name='form_list'),
    path('dataframe/<int:form_id>/', views.dataframe, name='dataframe'),
    #path('wordlist/', views.wordlist, name='wordlist'),
]
