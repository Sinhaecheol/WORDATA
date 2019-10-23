from django.db import models
import os

#사용자가 올리는 파일이 저장되고 분석에 관한 세부사항이 저장되는 테이블
class Userinput(models.Model):
    file = models.FileField(upload_to='documents/')
    frequency = models.IntegerField(default=10)
    word_except = models.IntegerField(default=False)
    times = models.IntegerField(default=20)

    def __str__(self):
        return os.path.basename(self.file.path)

#사용자가 분석을 요청한 단어가 저장되는 테이블
class Dataframe(models.Model):
    numbering = models.IntegerField(default=0, blank=True, null=True)
    word = models.CharField(max_length=50, blank=True, null=True)
    part_of_speech = models.CharField(max_length=20, blank=True, null=True)
    meaning = models.CharField(max_length=40, blank=True, null=True)
    example_sentence = models.TextField(blank=True, null=True)
    sentence_interpretation = models.TextField(blank=True, null=True)
    word_of_frequency = models.IntegerField(default=0, blank=True, null=True)

    def __str__(self):
        return str(self.word)

#수능 단어가 저장되는 테이블
class Suneung(models.Model):
    numbering = models.IntegerField(default=0, blank=True, null=True)
    word = models.CharField(max_length=50, blank=True, null=True)
    part_of_speech = models.CharField(max_length=20, blank=True, null=True)
    meaning = models.CharField(max_length=40, blank=True, null=True)
    example_sentence = models.TextField(blank=True, null=True)
    sentence_interpretation = models.TextField(blank=True, null=True)
    word_of_frequency = models.IntegerField(default=0, blank=True, null=True)

    def __str__(self):
        return str(self.word)

#평가원 단어가 저장되는 테이블
class Pyeonggawon(models.Model):
    numbering = models.IntegerField(default=0, blank=True, null=True)
    word = models.CharField(max_length=50, blank=True, null=True)
    part_of_speech = models.CharField(max_length=20, blank=True, null=True)
    meaning = models.CharField(max_length=40, blank=True, null=True)
    example_sentence = models.TextField(blank=True, null=True)
    sentence_interpretation = models.TextField(blank=True, null=True)
    word_of_frequency = models.IntegerField(default=0, blank=True, null=True)

    def __str__(self):
        return str(self.word)