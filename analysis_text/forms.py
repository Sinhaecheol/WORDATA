from django import forms
from .models import Userinput

#사용자가 파일을 올리고 분석할 단어의 범위와 단어의 개수를 정하는 form
class UserinputForm(forms.ModelForm):
    class Meta:
        model = Userinput
        fields = ['file', 'frequency', 'word_except', 'times']
        #'file': 파일을 받을 field
        #'frequency': 분석할 파일의 단어 빈도수를 정하는 field. frequency이하의 단어는 단어장 형성에 포함되지 않는다.
        #'word_except': 전처리할 단어를 정하는 field
        #'times': 단어장에 들어갈 단어 개수를 정하는 field