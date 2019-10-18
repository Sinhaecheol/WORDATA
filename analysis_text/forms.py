from django import forms
from .models import Userinput

class UserinputForm(forms.ModelForm):
    class Meta:
        model = Userinput
        fields = ['file', 'frequency', 'word_except', 'times']