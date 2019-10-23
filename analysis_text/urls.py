from django.urls import path
from . import views

app_name = 'analysis_text'
urlpatterns = [
    path('userinputform/', views.userinputform, name='userinputform'),
    path('form_list/', views.form_list, name='form_list'),
    path('dataframe/<int:form_id>/', views.dataframe, name='dataframe'),
    path('suneung_words/', views.suneung_words, name='suneung_words'),
    path('pyeonggawon_words/', views.pyeonggawon_words, name='pyeonggawon_words'),
    path('dataframe_words/', views.dataframe_words, name='dataframe_words'),
]
