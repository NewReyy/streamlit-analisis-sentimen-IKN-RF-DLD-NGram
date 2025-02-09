import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
import string
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemover, ArrayDictionary
nltk.download('stopwords')
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings("ignore")

"""# CLEANING"""

import re
import string

def cleaningText(text):
    # menghilangkan url
    text = re.sub(r'https?:\/\/\S+','',text)
    # menghilangkan mention, link, hastag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    #menghilangkan karakter byte (b')
    text = re.sub(r'(b\'{1,2})',"", text)
    # menghilangkan yang bukan huruf
    text = re.sub('[^a-zA-Z]', ' ', text)
    # menghilangkan digit angka
    text = re.sub(r'\d+', '', text)
    #menghilangkan tanda baca
    text = text.translate(str.maketrans("","",string.punctuation))
    # menghilangkan whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    # Memecah teks menjadi kata-kata terpisah
    text = text.split()

    return ' '.join(text)  # Mengembalikan teks yang sudah diproses kembali menjadi string

"""# TOKENIZE"""

def tokenizingText(text): # Tokenizing or splitting a string, text into a list of tokens
    text = word_tokenize(text)
    return text

nltk.download('punkt_tab')

"""# SLANGWORD"""

kamus_normalisasi = pd.read_csv("slang.csv")

kata_normalisasi_dict = {}

for index, row in kamus_normalisasi.iterrows():
    if row[0] not in kata_normalisasi_dict:
        kata_normalisasi_dict[row[0]] = row[1]

def normalisasi_kata(document):
    return [kata_normalisasi_dict[term] if term in kata_normalisasi_dict else term for term in document]

"""# FILTERING"""

import re

def filteringText(text):  # Remove stopwords in a text
    listStopwords = stopwords.words('indonesian')

    filtered = []
    for txt in text:
        if txt not in listStopwords:  # Cek stopwords dan validitas kata
            filtered.append(txt)

    return filtered

"""# STEMMING"""

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemmingText(token_list):
    # Daftar pengecualian
    exception_list = {
            "pengen": "pengen",
            "makan": "makan",
            "minum": "minum",
            "bermain": "main",
            "mengapa": "mengapa",
            "ingin": "ingin",
            "akan": "akan",
            "dengan": "dengan",
            "tidak": "tidak"
        }

    # stopword
    with open('kamus.txt') as kamus:
        word = kamus.readlines()
        list_stopword = [line.replace('\n', "") for line in word]
    dictionary = ArrayDictionary(list_stopword)
    stopword = StopWordRemover(dictionary)

    # Inisialisasi list untuk menyimpan hasil stemming
    stemmed_token_list = []

    # Proses setiap token dalam list
    for token in token_list:
        # Cek apakah token ada dalam daftar pengecualian
        if token in exception_list:
            stemmed_token_list.append(exception_list[token])
        else:
            # Hapus stopword
            token = stopword.remove(token)

            # Jika token tidak termasuk dalam stopword, lakukan stemming
            if token:
                stemmed_token = stemmer.stem(token)
                stemmed_token_list.append(stemmed_token)

    return stemmed_token_list