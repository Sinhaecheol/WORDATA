from django.db import models
import os
# Create your models here.
class Dataframe(models.Model):
    numbering = models.IntegerField(default=0)
    word = models.CharField(max_length=50)
    part_of_speech = models.CharField(max_length=20)
    meaning = models.CharField(max_length=40)
    example_sentence = models.TextField(blank=True, null=True)
    sentence_interpretation = models.TextField(blank=True, null=True)
    word_of_frequency = models.IntegerField(default=0)

    def __str__(self):
        return self.word

class Userinput(models.Model):
    file = models.FileField(upload_to='documents/') #파일을 저장하는 공간
    frequency = models.IntegerField(default=0)
    word_except = models.IntegerField(default=False)
    times = models.IntegerField(default=20)

    def __str__(self):
        return self.file.path
    
    