from django.shortcuts import render, get_object_or_404, redirect
from .models import Userinput, Dataframe
from .forms import UserinputForm

# import modules
# read data
import docx
from PyPDF2 import PdfFileReader

# Analysis
from nltk import *
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import re
import sys
import os

# Crawling
import requests
from bs4 import BeautifulSoup

# UnZip
def getZip(zipFile):
    unzip = zipfile.ZipFile(zipFile)
    unzip.extractall('zipfolder')
    unzip.close()
    path = os.getcwd()
#     path = re.sub(path[path.rindex('\h')+1:], '', zipFile)
    path += '\zipfolder'
    print("*Zip 해제 완료")
    return path
    
#하위 dir에 있는 파일들을 list로 반환한다.
def getFiles(path):
    files = []
    for a, b, c in os.walk(path):
        for f in c:
            file = os.path.join(a, f)
            files.append(file)
    print("*파일 스캔 완료")
    return files

#function of readding data
def getTextPDF(pdfFileName):
    with open(pdfFileName, 'rb') as f:
        read_pdf = PdfFileReader(f)  
        text = []
        for i in range(0, read_pdf.getNumPages()):
            text.append(read_pdf.getPage(i).extractText())
    return '\n'.join(text)

def getTextWord(wordFileName):
    doc = docx.Document(wordFileName)
    fullText = []
    for text in doc.paragraphs:
        fullText.append(text.text)
    return '\n'.join(fullText)

def read_file(file):
    text = []
    if file[-3:] == 'pdf':
        text.append(getTextPDF(file))
    elif file[-4:] == 'docx':
        text.append(getTextWord(file))
    return text

def removeFolder(file):
    if os.path.isfile(file):
        os.remove(file)
        
#텍스트 분석
def wordata(file, n, wordExcept = False, lenth = False):
    FileName = file.path
    mode = ''
    textHeap = []

    if FileName[-3:] == 'zip':
        FileName = getZip(FileName)
        FileName = getFiles(FileName)
        mode = 'Zip'

    #mode에 따라 선택적 처리
    print("*파일 읽기를 시작합니다.")


    if mode == 'Zip':
        for file in FileName:
            textHeap += read_file(file)

    else:
        textHeap = read_file(FileName)

    #집 해제된 폴더안의 파일 삭제
    for file in FileName:
        removeFolder(file)
        
    #Tokenization
    print("*텍스트 분석을 시작합니다.")
    tokenizer = WordPunctTokenizer()
    TokenizedWords = []

    for text in textHeap:
        TokenizedWords += tokenizer.tokenize(text)
    print("*문서 안의 전체 단어 개수: {}" .format(len(TokenizedWords))) 
    
    for file in FileName:
        removeFolder(file)

    #불용어 제거

    #불용어 load
    errorWords = pd.read_csv("C:\\Users\\haecheol\\Desktop\\WORDATA\\errorword\\errorWords.csv", header = None)

    stop_words = set(stopwords.words('english')) # NLTK에서 기본적으로 정의하고 있는 불용어
    stop_words = stop_words | set(pd.Series(errorWords[0]).to_list())

    # 과정별 단어 제거
    if wordExcept == 1:
        elementWord = pd.read_csv("C:\\Users\\haecheol\\Desktop\\WORDATA\\errorword\\초등.csv", header = None)
        stop_words = stop_words | set(pd.Series(elementWord[0]).to_list())
        print('초등 영단어 800 제거 성공')
    elif wordExcept == 2:
        middleWord = pd.read_csv("C:\\Users\\haecheol\\Desktop\\WORDATA\\errorword\\중등2000.csv", header = None)
        stop_words = stop_words | set(pd.Series(middleWord[0]).to_list())
    elif wordExcept == 3:
        highWord = pd.read_csv("C:\\Users\\haecheol\\Desktop\\WORDATA\\errorword\\고등3000.csv", header = None)
        stop_words = stop_words | set(pd.Series(highWord[0]).to_list())

    np_words = np.array(TokenizedWords) # Tokenized words를 numpy array type으로 형 변환
    delete_index = [] # 불용어 index번호를 저장할 list

    print("*1차 불용어 제거를 시작합니다.")
    for i in range(len(np_words)):
        np_words[i] = re.sub("[^a-zA-Z]", "", np_words[i])

        if (np_words[i] in stop_words) == True: 
            delete_index.append(i)
        if len(np_words[i]) <= 1:
            delete_index.append(i)

    TrimmedWords = np.delete(np_words, delete_index) #불용어 index를 삭제
    print('제거 후 단어 수: {}' .format(len(TrimmedWords)))
    
    
     # 품사 태깅
    tagged_list = pos_tag(TrimmedWords)
    verb =[]
    noun = []
    adject = []
    adverb = []
    for w in tagged_list:
        if w[1]=='VB' or  w[1]=='VBD' or w[1] =='VBG' or w[1] == 'VBN' or w[1] == 'VBP' or w[1]=='JJ':
            verb.append(w) 
        elif w[1]=='NNS' or w[1] == 'NNPS' or w[1]== 'NN':
            if len(w[0]) > 3 and w[0][-3] == 'ing': #만약 현재분사로써 대문자인 -ing 형이 온다면 아래를 실행
                verb.append(w)
            else:
                noun.append(w)
        elif w[1] =='JJ' or w[1]=='JJR' or w[1] == 'JJS':
            adject.append(w)
        elif w[1]=='RBR' or w[1] == 'RBS' or w[1]=='RB':
            adverb.append(w)

    verb = untag(verb)
    noun = untag(noun)
    adject = untag(adject)
    adverb = untag(adverb)

    restoredVerb = [] # 동사 원형 복원
    for v in verb:
        restoredVerb.append(WordNetLemmatizer().lemmatize(v.lower(), pos='v'))                       
    restoredNoun = [WordNetLemmatizer().lemmatize(w, pos='n') for w in noun]  #명사 원형 복원
    restoredAdject = [WordNetLemmatizer().lemmatize(w, pos='a') for w in adject]  #형용사 원형 복원
    restoredAdverb = [WordNetLemmatizer().lemmatize(w, pos='r') for w in adverb]  #부사 원형 복원

    #복원된 데이터 통합
    combinedWords = restoredVerb + restoredNoun + restoredAdject + restoredAdverb
    print("*필터된 단어의 개수: {}" .format(len(combinedWords)))
    
    np_words = np.array(combinedWords) # Tokenized words를 numpy array type으로 형 변환
    delete_index_2 = [] # 불용어 index번호를 저장할 list

    print("*2차 불용어 제거를 시작합니다.")
    for i in range(len(np_words)):
    #     np_words[i] = np_words[i].lower()  #모든 단어를 소문자로 변경
        if (np_words[i] in stop_words) == True: 
            delete_index_2.append(i)
        if len(np_words[i]) >= 20 or len(np_words[i]) <= 2:
            delete_index_2.append(i)
    TrimmedWords = np.delete(np_words, delete_index_2) #불용어 index를 삭제
    print('제거 후 단어 수: {}' .format(len(TrimmedWords)))
    resultWords = TrimmedWords
    
    overNum = n  #빈도수 개수 이상 단어를 뽑아냄

    print('*중복된 단어의 갯수를 셉니다.')
    cleansing = pd.Series(resultWords).value_counts()
    removedOverlabWords = pd.DataFrame()
    removedOverlabWords['Word'] = cleansing.index
    removedOverlabWords['value count'] = cleansing.values
    removedOverlabWords = removedOverlabWords[removedOverlabWords['value count']> overNum ]
    print("*** 단어 분석 완료 ***")
    print('{}개 이상의 빈도수 단어를 추출합니다.'.format(overNum))
    print("최종 단어 수 : {}" .format(removedOverlabWords['Word'].count()))
    
    print('*단어장 형성을 시작합니다.')
  
   #lenth 파라미터를 받음
    if lenth == False:
        WordsLenth = removedOverlabWords['Word'].to_list()
    else:
        lenth -= 1
        WordsLenth = removedOverlabWords['Word'].loc[:lenth].to_list()

    word_url=[] #단어 검색 후 첫번째 url
    dataFrame = pd.DataFrame(columns=['번호','단어', '품사', '뜻','예문', '해석','빈도수'])
    word_number = 1
    dictlink='https://endic.naver.com' #네이버사전 홈페이지 url



    for i in range(len(WordsLenth)): #테스트 단어 갯수만큼 반복
        url="https://endic.naver.com/search.nhn?sLn=kr&query="+ WordsLenth[i] #단어검색 뒤에 영단어를 붙혀 url 넘겨줌
        res=requests.get(url).text #url을 requests한걸 text로 가져와 res 에 저장
        soup=BeautifulSoup(res,"lxml") #res를 beautiful soup에 넘겨줌

        for link in soup.findAll("a",href=re.compile("^(/enkrEntry)")): #a 태그의 href를 가져올건데 (/enkrEntry) 이부분이 가져올 url들에 반복되어 그 href를 다찾음
             if 'href' in link.attrs: #만약 link의 모든속성에 href가 있따면
                    word=link.attrs['href'] #href의 모든속성을 word에 저장 즉 그 사전검색하면 나오는 두번째 url 저장
                    word=dictlink+word #네이버 사전 홈페이지+2번째 페이지 url을 넘겨준다.
                    word_url.append(word) #합친 url을 word_url의 리스트에 넣음
                    break #첫번째 word_url만 뽑아서 멈춤



    selecter = [
        ['span.fnt_k06',
         'p.bg span.fnt_e07._ttsText',
         'dd.first p span.fnt_k10',
         'span.fnt_syn',
         '#content > div.word_view > div.tit > h3'],
        ['#zoom_content > div:nth-child(6) > dl > dt.first.mean_on.meanClass > em > span.fnt_k06',
         '#zoom_content > div:nth-child(6) > dl > dd:nth-child(2) > p.bg > span',
         '#zoom_content > div:nth-child(6) > dl > dd:nth-child(2) > p:nth-child(2) > span',
         '#ajax_hrefContent > li:nth-child(2) > a',
         '#content > div.word_view > div.tit > h3'],
        ['#zoom_content > div:nth-child(7) > dl > dt > em > span.fnt_k06',
        '#zoom_content > div:nth-child(7) > dl > dd.first > p.bg > span',
        '#zoom_content > div:nth-child(7) > dl > dd.first > p:nth-child(2) > span',
        '#ajax_hrefContent > li:nth-child(3) > a',
        '#content > div.word_view > div.tit > h3']]


    def makeVoca(dataFrame, soup, selecter, word_name, number, freq):
        global word_url

        words = soup.select(selecter[0]) #단어 뜻
        if len(words)==0:
            return

        examples = soup.select(selecter[1]) #예문
        if len(examples)==0:
            example = None
        else:
            example=examples[0].get_text().strip()

        inperprets = soup.select(selecter[2]) #예문 해석
        if len(inperprets) == 0:
            interpretation = None
        else:
            interpretation=inperprets[0].get_text().strip()
        parts = soup.select(selecter[3]) #품사
        if len(parts) == 0:
            part = None
        else:
            part=parts[0].get_text().strip()

        voca = soup.select(selecter[4])#두번째 단어
        Words = voca[0].get_text().strip()
        meaning=words[0].get_text().strip() #단어의 첫번째만 태그 제거하여 리스트에 넘김

        dataFrame.loc[len(dataFrame)] = [number, word_name, part, meaning, example, interpretation, freq]
        return dataFrame

    for j in range(len(WordsLenth)): #단어갯수 만큼 반복
        response=requests.get(word_url[j]).text #url넘기고 요청하여 텍스트로 넘김
        soup_2 = BeautifulSoup(response, "lxml") # lxml을 이용하여 beautifulsoup으로 넘겨줌
        for i in range(len(selecter)):
            makeVoca(dataFrame, soup_2, selecter[i], WordsLenth[j], word_number, removedOverlabWords['value count'].loc[j] )
        word_number += 1  
    print("***단어장 형성 완료***")
    return dataFrame
    
def userinputform(request):
    if request.method == 'POST':
        form = UserinputForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('main:home')
    else:
        form = UserinputForm()
    return render(request, 'analysis_text/userinputform.html', {'form': form})

def form_list(request):
    forms = Userinput.objects.all()
    return render(request, 'analysis_text/form_list.html', {'forms': forms})

def dataframe(request, form_id):
    form = get_object_or_404(Userinput, id=form_id)
    dataFrame = wordata(form.file, form.frequency, form.word_except, form.times)
    for i in range(len(dataFrame['단어'].to_list())):
        db_dataframe = Dataframe(
        numbering = dataFrame.loc[i].to_list()[0],
        word = dataFrame.loc[i].to_list()[1],
        part_of_speech = dataFrame.loc[i].to_list()[2],
        meaning = dataFrame.loc[i].to_list()[3],
        example_sentence = dataFrame.loc[i].to_list()[4],
        sentence_interpretation = dataFrame.loc[i].to_list()[5],
        word_of_frequency = dataFrame.loc[i].to_list()[6],
        )
        db_dataframe.save()

    words = Dataframe.objects.all()
    return render(request, 'analysis_text/dataframe.html', {'words': words})

def wordlist(request):
    words = Dataframe.objects.all()
    return render(request, 'analysis_text/wordlist.html', {'words': words})

# def wordlist_detail(request, list_id):