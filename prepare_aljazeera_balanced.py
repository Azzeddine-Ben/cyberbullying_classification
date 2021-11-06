# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 21:06:51 2021

@author: IT Doctor
"""

import numpy as np
import pandas as pd
import re, itertools, html, string, unicodedata
from nltk.tokenize import RegexpTokenizer
from string import digits
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

arabic_punctuations = '''٠١٢٣٤٥٦٧٨٩·¿¡¡¸\u202b\u200f\u200f#`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ٠\n°•﴿\u200f«\t\u202c\u202a\u202b¸¸·´·´¨¸·´»x\xa0♡\u200c💕💐﴾🏆٫бинлодензиндаваҳаётзериназоратифбрамрикост‘'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida

                         """, re.VERBOSE)

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def decoding_html(text):
  return html.unescape(text)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def standardization(text):
  text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
  return text

def denoise_text(text):
    text = remove_numbers(text)
    # text = remove_arabic_numbers(text)
    text = preprocessingArabic(text)
    text = decoding_html(text)
    text = standardization(text)
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

############################################## Arabic Text Preprocessing

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text


def preprocessingArabic(text):
#     preprocessedText = remove_diacritics(normalize_arabic(text))
    preprocessedText = normalize_arabic(text)
    preprocessedText = remove_diacritics(text)
    return preprocessedText


def tokenizeArabic(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokenizedText = tokenizer.tokenize(text)
    return tokenizedText

def remove_numbers(text):
  numbers_removed = text.lstrip(digits)
  numbers_removed = ''.join([i for i in numbers_removed if not i.isdigit()])
  return numbers_removed

def remove_arabic_numbers(text):
  strr = "٠١٢٣٤٥٦٧٨٩";  # Only digit in this string
  numbers_removed = ''.join([i for i in text if not i in strr])
  return numbers_removed

def remove_less_than_one(text):
  if text != 'ﺁ':
    text = re.sub(r'\b\w{1}\b', '', text)
  return text

########################################
########### Reading the dataset
dataset_name = 'aljazeera_balanced'

ds = pd.read_excel("/content/drive/My Drive/Colab Notebooks/AJCommentsClassification-CF.XLSX", header = 0)

df_agg_labelOne = ds[ds['languagecomment'] == -1]
df_agg_labelOne = df_agg_labelOne.sort_values(by='words', ascending=False)
df_agg_labelOne = df_agg_labelOne.iloc[:5120]
df_agg_labelOne.languagecomment = df_agg_labelOne.languagecomment.map({-1 : -2})
frames = [ds, df_agg_labelOne]
ds = pd.concat(frames)
ds = pd.concat([ds[ds["languagecomment"]==0], ds[ds["languagecomment"]==-2]])
print(ds.languagecomment.value_counts())

print("Maximum sentence length : {}".format(max(list(ds.words))))
d = {'post': list(ds.body), 'label': list(ds.languagecomment.map({0 : 0, -2 : 1}))}

df1 = pd.DataFrame(data=d)
dff = df1.dropna()

########## Preprocessing Dataset
print("############ Preprocessing ... \n")
data_preprocessed = []
for post in dff.post:
    # data_preprocessed.append(''.join(remove_punctuation(''.join((denoise_text(post))))))
    data_preprocessed.append(''.join(remove_less_than_one(''.join(remove_punctuation(''.join((denoise_text(post))))))))
d = {'post': data_preprocessed, 'label': dff.label}
dataset = pd.DataFrame(data=d)

dataset = dataset[dataset['post'] != '' ]
dataset = dataset[dataset['post'] != ' ' ]
# print(len(dataset.label), dataset.label.value_counts())
print(dataset['label'].value_counts(normalize = True))

post_length = np.array([len(post.split(" ")) for post in dataset.post])
MAX_LEN = int(np.percentile(post_length, 95))
print("Post lentgh at 95 percentile {}".format(MAX_LEN))


### Train/test data split
print("#### Splitting dataset ... \n")
X_train, X_test, y_train, y_test = train_test_split(dataset['post'],dataset['label'], test_size=0.2, random_state=59)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=59)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

### Conversion to numpy arrays
print("#### Converting to numpy arrays ... \n")
X_train = np.asarray(X_train)
X_valid = np.asarray(X_valid)
y_train = np.asarray(y_train)
y_valid = np.asarray(y_valid)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

path ='/content/drive/My Drive/'

### Saving files
print('Saving Dataset Files ... \n')
         
pickle.dump( X_train, open( path + "X_train_" + dataset_name, "wb" ))
pickle.dump( y_train, open( path + "y_train_" + dataset_name, "wb" ))        

pickle.dump( X_valid, open( path + "X_valid_" + dataset_name, "wb" ))
pickle.dump( y_valid, open( path + "y_valid_" + dataset_name, "wb" )) 

pickle.dump( X_test, open( path + "X_test_" + dataset_name, "wb" ))
pickle.dump( y_test, open( path + "y_test_" + dataset_name, "wb" ))
       
print("Terminated.")
