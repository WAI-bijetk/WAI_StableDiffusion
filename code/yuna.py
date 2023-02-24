import json
import nltk
import os
import sys
import re
import urllib.request
from translate import Translator
from typing import List, Dict
import argparse    
from nltk.data import load
from nltk.tokenize import word_tokenize  

nltk.download('punkt')
client_id = "H4q5ybS5cfAmEl1hDUqC" # 개발자센터에서 발급받은 Client ID 값
client_secret = "w5KyEQhM2l" # 개발자센터에서 발급받은 Client Secret  

def papago_api(input_text, client, client_secret):
    encText = urllib.parse.quote(input_text)
    data = "source=ko&target=en&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()     
    if(rescode==200):
        response_body = response.read() 
        translated_text = response_body.decode('utf-8')
        dic = json.loads(translated_text)         
        text = dic['message']['result']['translatedText']         
        return text      
    else:
        print("Error Code:" + rescode)

def cleaned_text(translated_text, *args): 
    stop_words_list = list()  
    with open('stopwords.txt', 'r') as f:
        i = [line.strip() for line in f.readlines()]
        stop_words_list.append(i)   
    stop_words_list = stop_words_list[0]
    
    tokend_list = word_tokenize(translated_text)
    tokend_tagging_list = nltk.pos_tag(tokend_list)
    wanted_tagging_list = ["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBP", "VBZ"]  
    result_dic = dict()
    for tokened, tag in tokend_tagging_list:
        if tag in wanted_tagging_list:
            if tag not in result_dic:
                result_dic[tag] = [tokened]
            else:
                result_dic[tag].append(tokened)            
    prompt_text = " ".join(list(" ".join(v) for v in result_dic.values()))
    magic_words = ",".join(list(args))
    return prompt_text +" "+magic_words

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='papgo_api_tutorial')      
    parser.add_argument('--prompt','--p', type=str, default='아침에 커피 마시기') # default='아침에 커피 마시기'
    args = parser.parse_args()  
    translated_text = papago_api(args.prompt, client_id, client_secret)    
    prompt_text = cleaned_text(translated_text, "HDO", "DDDD")
    print(prompt_text)