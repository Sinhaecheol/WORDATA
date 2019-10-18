from django.db import models
import os
# Create your models here.
class Dataframe(models.Model):
    word = models.CharField(max_length=50)
    part_of_speech = models.CharField(max_length=20)
    meaning = models.CharField(max_length=40)
    example_sentence = models.TextField()
    sentence_interpretation = models.TextField()
    word_of_frequency = models.IntegerField(default=0)

class Userinput(models.Model):
    file = models.FileField(upload_to='documents/') #파일을 저장하는 공간
    frequency = models.IntegerField(default=0)
    word_except = models.BooleanField(default=False)
    times = models.IntegerField(default=20)
    
    