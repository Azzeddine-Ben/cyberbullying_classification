# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:37:04 2021

@author: IT Doctor
"""

import collections as coll
import math
import string
import nltk
import textstat
import numpy as np
import pandas as pd
from matplotlib import style
from nltk.corpus import cmudict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from lexical_diversity import lex_div as ld

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

import os 
import argparse
import pickle
import load_data_module
import warnings
from tqdm import tqdm


############################################# LEXICAL FEATURES
def syllable_count_Manual(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("e"):
                count -= 1
    if count == 0:
        count += 1
    return count

# ---------------------------------------------------------------------
# COUNTS NUMBER OF SYLLABLES

def syllable_count(word):
    global cmuDictionary
    d = cmuDictionary
    try:
        syl = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except:
        syl = syllable_count_Manual(word)
    return syl

    # ----------------------------------------------------------------------------


# removing stop words plus punctuation. 
def Avg_wordLength(str): 
    str.translate(string.punctuation)
    tokens = word_tokenize(str, language='english')
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    stop = stopwords.words('english') + st
    words = [word for word in tokens if word not in stop]
    return np.average([len(word) for word in words])


# ----------------------------------------------------------------------------


# returns avg number of characters in a sentence
def Avg_SentLenghtByCh(text):   
    tokens = sent_tokenize(text)
    return np.average([len(token) for token in tokens])


# ----------------------------------------------------------------------------

# returns avg number of words in a sentence
def Avg_SentLenghtByWord(text):
    tokens = sent_tokenize(text)
    return np.average([len(token.split()) for token in tokens])


# ----------------------------------------------------------------------------
# RETURNS NORMALIZED COUNT OF FUNCTIONAL WORDS FROM A Framework for
# Authorship Identification of Online Messages: Writing-Style Features and Classification Techniques

def CountFunctionalWords(text): 
    functional_words = """a between in nor some upon
    about both including nothing somebody us
    above but inside of someone used
    after by into off something via
    all can is on such we
    although cos it once than what
    am do its one that whatever
    among down latter onto the when
    an each less opposite their where
    and either like or them whether
    another enough little our these which
    any every lots outside they while
    anybody everybody many over this who
    anyone everyone me own those whoever
    anything everything more past though whom
    are few most per through whose
    around following much plenty till will
    as for must plus to with
    at from my regarding toward within
    be have near same towards without
    because he need several under worth
    before her neither she unless would
    behind him no should unlike yes
    below i nobody since until you
    beside if none so up your
    """

    functional_words = functional_words.split()
    words = RemoveSpecialCHs(text)
    count = 0

    for i in text:
        if i in functional_words:
            count += 1

    return count / len(words)


# ---------------------------------------------------------------------------

def avg_character_per_word(text):
  return textstat.avg_character_per_word(text)
 
# ---------------------------------------------------------------------------

def avg_syllables_per_word(text):
  return textstat.avg_syllables_per_word(text)

# ---------------------------------------------------------------------------

def lex_dvrst(text):
  tok = ld.tokenize(text)
  flt = ld.flemmatize(text)

  simpleTTR = ld.ttr(flt)
  rootTTR   = ld.root_ttr(flt)
  logTTR    = ld.log_ttr(flt)
  massTTR   = ld.maas_ttr(flt)
  mean_segmental_TTR = ld.msttr(flt)
  moving_avg_TTR = ld.mattr(flt)
  hypogeometric_distribution = ld.hdd(flt)
  MTLD  = ld.mtld(flt)
  MTLD2 = ld.mtld_ma_wrap(flt)
  MTLD3 = ld.mtld_ma_bid(flt)

  return [simpleTTR, rootTTR, logTTR, massTTR, mean_segmental_TTR, moving_avg_TTR,
          hypogeometric_distribution, MTLD, MTLD2, MTLD3]

# ---------------------------------------------------------------------------

######################################## Readability features
def get_readability_features(text):
  flesh_reading     = textstat.flesch_reading_ease(text)
  flesh_kincaid     = textstat.flesch_kincaid_grade(text)
  coleman_liau      = textstat.coleman_liau_index(text)
  smog_index        = textstat.smog_index(text)
  readability_index = textstat.automated_readability_index(text)
  dale_cahll        = textstat.dale_chall_readability_score(text)
  linsear_formula   = textstat.linsear_write_formula(text)
  gunning_fox       = textstat.gunning_fog(text)
  fernandez_huerta  = textstat.fernandez_huerta(text)
  szigriszt_pazos   = textstat.szigriszt_pazos(text)
  gutierrez_polini  = textstat.gutierrez_polini(text)
  crawford          = textstat.crawford(text)
  lix               = textstat.lix(text)
  polysyllabcount   = textstat.polysyllabcount(text)
  reading_time      = textstat.reading_time(text)
  rix               = textstat.rix(text)

  return[flesh_reading, flesh_kincaid, coleman_liau, smog_index, readability_index, dale_cahll,
         linsear_formula, gunning_fox, szigriszt_pazos, fernandez_huerta, #text_standard
         gutierrez_polini, crawford, lix, polysyllabcount, reading_time, rix]

# ---------------------------------------------------------------------------

############################### STYLOMETRIC FEATURES
# also returns Honore Measure R
def hapaxLegemena(text):
    words = RemoveSpecialCHs(text)
    V1 = 0
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    for word in freqs:
        if freqs[word] == 1:
            V1 += 1
    N = len(words)
    V = float(len(set(words)))
    R = 100 * math.log(N) / max(1, (1 - (V1 / V)))
    # h = V1 / N
    return R


# ---------------------------------------------------------------------------

def hapaxDisLegemena(text):
    words = RemoveSpecialCHs(text)
    count = 0
    # Collections as coll Counter takes an iterable collapse duplicate and counts as
    # a dictionary how many equivelant items has been entered
    freqs = coll.Counter()
    freqs.update(words)
    for word in freqs:
        if freqs[word] == 2:
            count += 1

    # h = count / float(len(words))
    S = count / float(len(set(words)))
    return S


# ---------------------------------------------------------------------------

# c(w)  = ceil (log2 (f(w*)/f(w))) f(w*) frequency of most commonly used words f(w) frequency of word w
# measure of vocabulary richness and connected to zipfs law, f(w*) const rak kay zips law say rank nikal rahay hein
def AvgWordFrequencyClass(text):
    words = RemoveSpecialCHs(text)
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    maximum = float(max(list(freqs.values())))
    return np.average([math.floor(math.log((maximum + 1) / (freqs[word]) + 1, 2)) for word in words])


# --------------------------------------------------------------------------
# logW = V-a/log(N)
# N = total words , V = vocabulary richness (unique words) ,  a=0.17
# we can convert into log because we are only comparing different texts
# def BrunetsMeasureW(text):
#     words = RemoveSpecialCHs(text)
#     a = 0.17
#     V = float(len(set(words)))
#     N = len(words)
#     B = (V - a) / (math.log(N))
#     return B

# ------------------------------------------------------------------------
def RemoveSpecialCHs(text):
    text = word_tokenize(text)
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']

    words = [word for word in text if word not in st]
    return words


# -------------------------------------------------------------------------
# K  10,000 * (M - N) / N**2
# , where M  Sigma i**2 * Vi.
def YulesCharacteristicK(text):
    words = RemoveSpecialCHs(text)
    N = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    vi = coll.Counter()
    vi.update(freqs.values())
    M = sum([(value * value) * vi[value] for key, value in freqs.items()])
    K = 10000 * (M - N) / math.pow(N, 2)
    return K


# -------------------------------------------------------------------------

############################################## Sentiment Features

def getSentimentValues(cmnt):
    sentiment_list = []

    blob_cmnt = TextBlob(cmnt)
    sentiment_list.append(blob_cmnt.sentiment.polarity)
    sentiment_list.append(blob_cmnt.sentiment.subjectivity)

    analyzer = SentimentIntensityAnalyzer()
    vader_sentiments = analyzer.polarity_scores(cmnt)
    sentiment_list.append(vader_sentiments['compound'])
    sentiment_list.append(vader_sentiments['pos'])
    sentiment_list.append(vader_sentiments['neu'])
    sentiment_list.append(vader_sentiments['neg'])

    return sentiment_list

################### Robust scaling function
def RobustScaling(features):
  robust_scaler = RobustScaler()
  robust_scaler.fit(features)
  scaled_features = robust_scaler.transform(features)
  return robust_scaler, scaled_features 
############################################## FEATURE EXTRACTION FUNCTION
# returns a feature vector of text
def FeatureExtration(dataset):
  
    # cmu dictionary for syllables
    global cmuDictionary
    cmuDictionary = cmudict.dict()

    readability_matrix, lexical_matrix, stylometric_matrix, sentiment_matrix = [], [], [], []

    for text in tqdm(dataset):
      text_lexical_features, text_stylometric_features = [], []
      ########## Extract lexical features
      text_lexical_features = [syllable_count(text), Avg_wordLength(text), Avg_SentLenghtByCh(text), 
                               Avg_SentLenghtByWord(text), CountFunctionalWords(text), avg_character_per_word(text), 
                               avg_syllables_per_word(text)               
                              ]
      text_lex_dvrst = lex_dvrst(text)
      for measure in text_lex_dvrst:
        text_lexical_features.append(measure)
      lexical_matrix.append(text_lexical_features)

      ########### Extract readability features
      readability_matrix.append(get_readability_features(text))

      ########### Extract styolemtric features
      text_stylometric_features = [hapaxLegemena(text), hapaxDisLegemena(text), AvgWordFrequencyClass(text),
                                    YulesCharacteristicK(text)
                                   ]
      stylometric_matrix.append(text_stylometric_features)

      ########### Extract sentiment features
      sentiment_matrix.append(getSentimentValues(text))


    return np.asarray(readability_matrix), np.asarray(lexical_matrix), np.asarray(stylometric_matrix), np.asarray(sentiment_matrix)


if __name__ == '__main__':
    
    warnings.filterwarnings("ignore")
                        
    nltk.download('cmudict')
    nltk.download('stopwords')
    nltk.download('punkt')
    
    style.use("ggplot")
    cmuDictionary = None

    # Create the parser
    my_parser = argparse.ArgumentParser()
    
    # Add the arguments
    my_parser.add_argument('dataset_name',
                           metavar='dataset_name',
                           type=str,
                           help='Name of the dataset to extract features')
    my_parser.add_argument('eda_dataset',
                           metavar='eda_dataset',
                           type=str,
                           help="y/n to load the eda version of the dataset")
    
    # Execute the parse_args() method
    args = my_parser.parse_args()
    dataset_name = args.dataset_name
    eda_dataset = args.eda_dataset
    path ='/content/drive/MyDrive/'
    
    if eda_dataset == 'y':
        X_train, X_test = load_data_module.load_eda_data(dataset_name)
        y_train, y_test = load_data_module.load_eda_labels(dataset_name)
        liwc_train_dataset = pd.read_csv(path + 'liwc_eda_'+ dataset_name +'_train.csv')
        liwc_test_dataset  = pd.read_csv(path + 'liwc_eda_'+ dataset_name +'_test.csv')
        directory = dataset_name + '_eda_features'
    elif eda_dataset == 'n':
        X_train, X_test = load_data_module.load_data(dataset_name)
        y_train, y_test = load_data_module.load_labels(dataset_name)
        liwc_train_dataset = pd.read_csv(path + 'liwc_' + dataset_name +'_train.csv')
        liwc_test_dataset  = pd.read_csv(path + 'liwc_' + dataset_name +'_test.csv')
        directory = dataset_name + '_features'
        
    liwc_train_data = liwc_train_dataset[liwc_train_dataset.columns[2:-16]].to_numpy()
    liwc_test_data  = liwc_test_dataset[liwc_test_dataset.columns[2:-16]].to_numpy()
    
    floatConvert = lambda x: float(x.replace(',','.')) if isinstance(x, int) != True else x
    vfunc = np.vectorize(floatConvert)
    liwc_train_data, liwc_test_data = vfunc(liwc_train_data), vfunc(liwc_test_data)
    
    print("Extracting features ... \n")

    X_test_readability, X_test_lexical, X_test_stylometric, X_test_sentiments        = FeatureExtration(X_test)
    readability_features, lexical_features, stylometric_features, sentiment_features = FeatureExtration(X_train)
    
    print('Extraction Done.\n')
    
    lexical_features = np.nan_to_num(lexical_features) ### Replaces NaN values with zeros
    
    ### Scaling Features values
    stylometric_scaler, scaled_stylometric_features =  RobustScaling(stylometric_features)
    test_stylometric = stylometric_scaler.transform(X_test_stylometric)
    
    readability_scaler, scaled_readability_features =  RobustScaling(readability_features)
    test_readability = readability_scaler.transform(X_test_readability)
    
    lexical_scaler, scaled_lexical_features =  RobustScaling(lexical_features)
    test_lexical = lexical_scaler.transform(X_test_lexical)
    
    liwc_scaler, scaled_liwc_features =  RobustScaling(liwc_train_data)
    test_liwc = liwc_scaler.transform(liwc_test_data)
    
    sentiments_scaler, scaled_sentiments_features =  RobustScaling(sentiment_features)
    test_sentiments = sentiments_scaler.transform(X_test_sentiments)
    
    print('Splitting data ...\n')
    ## Train/validation data split
    X_train, X_valid, y_train, y_valid       = train_test_split(X_train, y_train, test_size=0.2, random_state=59)
    X_train_stylometric, X_valid_stylometric = train_test_split(stylometric_features, test_size=0.2, random_state=59)
    X_train_lexical, X_valid_lexical         = train_test_split(lexical_features, test_size=0.2, random_state=59)
    X_train_readability, X_valid_readability = train_test_split(readability_features, test_size=0.2, random_state=59)
    X_train_liwc, X_valid_liwc               = train_test_split(liwc_train_data, test_size=0.2, random_state=59)
    X_train_sentiments, X_valid_sentiments   = train_test_split(sentiment_features, test_size = 0.2, random_state=59)
    print('Splitting Done.\n')

    if (os.path.exists(directory)) == False:  
        os.mkdir(directory)
        train_dir, valid_dir, test_dir = directory + '/train', directory + '/valid', directory + '/test'
        os.mkdir(train_dir)
        os.mkdir(valid_dir)
        os.mkdir(test_dir)
        
        pickle.dump( X_train, open( train_dir + "/X_train", "wb" ) )
        pickle.dump( y_train, open( train_dir + "/y_train", "wb" ) )        
        pickle.dump( X_train_stylometric, open( train_dir + "/X_train_stylometric", "wb" ) )        
        pickle.dump( X_train_lexical, open( train_dir + "/X_train_lexical", "wb" ) )       
        pickle.dump( X_train_readability, open( train_dir + "/X_train_readability", "wb" ) )         
        pickle.dump( X_train_liwc, open( train_dir + "/X_train_liwc", "wb" ) )      
        pickle.dump( X_train_sentiments, open( train_dir + "/X_train_sentiments", "wb" ) )
        
        pickle.dump( X_valid, open( valid_dir + "/X_valid", "wb" ) )
        pickle.dump( y_valid, open( valid_dir + "/y_valid", "wb" ) )
        pickle.dump( X_valid_stylometric, open( valid_dir + "/X_valid_stylometric", "wb" ))
        pickle.dump( X_valid_lexical, open( valid_dir + "/X_valid_lexical", "wb" ) )
        pickle.dump( X_valid_readability, open( valid_dir + "/X_valid_readability", "wb" ))
        pickle.dump( X_valid_liwc, open( valid_dir + "/X_valid_liwc", "wb" ) )     
        pickle.dump( X_valid_sentiments, open( valid_dir + "/X_valid_sentiments", "wb" ) )
        
        pickle.dump( test_stylometric, open( test_dir + "/X_test_stylometric", "wb" ))
        pickle.dump( test_lexical, open( test_dir + "/X_test_lexical", "wb" ) )
        pickle.dump( test_readability, open( test_dir + "/X_test_readability", "wb" ))
        pickle.dump( test_liwc, open( test_dir + "/X_test_liwc", "wb" ) )     
        pickle.dump( test_sentiments, open( test_dir + "/X_test_sentiments", "wb" ) )
        
                    
                    