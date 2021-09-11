# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:22:20 2021

@author: IT Doctor
"""

import pickle

def load_data(dataset_name):
    pickle_in = open(dataset_name + '/X_train_' + dataset_name, 'rb' )
    X_train   = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(dataset_name +'/X_test_' + dataset_name, 'rb')
    X_test    = pickle.load(pickle_in)
    pickle_in.close()
    
    return X_train, X_test

def load_eda_data(dataset_name):
    pickle_in = open(dataset_name +'/eda_X_train_' + dataset_name, 'rb' )
    X_train   = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(dataset_name +'/eda_X_test_' + dataset_name, 'rb')
    X_test    = pickle.load(pickle_in)
    pickle_in.close()
    return X_train, X_test

def load_labels(dataset_name):
    pickle_in = open(dataset_name +'/y_train_' + dataset_name, 'rb' )
    y_train   = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(dataset_name +'/y_test_' + dataset_name, 'rb')
    y_test    = pickle.load(pickle_in)
    pickle_in.close()
    
    return y_train, y_test

def load_eda_labels(dataset_name):
    pickle_in = open(dataset_name +'/eda_y_train_' + dataset_name, 'rb' )
    y_train   = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(dataset_name+'/eda_y_test_' + dataset_name, 'rb')
    y_test    = pickle.load(pickle_in)
    pickle_in.close()
    return y_train, y_test

def load_train_features(dataset_name, eda):
    if eda == 'y':
        dataset_name = dataset_name + '_eda'
    pickle_in = open(dataset_name + '_features/train/X_train', 'rb')
    X_train = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(dataset_name + '_features/train/X_train_stylometric', 'rb')
    X_train_stylometric = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(dataset_name + '_features/train/X_train_readability', 'rb')
    X_train_readability = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(dataset_name + '_features/train/X_train_lexical', 'rb')
    X_train_lexical = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(dataset_name + '_features/train/X_train_liwc', 'rb')
    X_train_liwc = pickle.load(pickle_in)
    pickle_in.close()    

    pickle_in = open(dataset_name + '_features/train/X_train_sentiments', 'rb')
    X_train_sentiments= pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(dataset_name + '_features/train/y_train', 'rb')
    y_train = pickle.load(pickle_in)
    pickle_in.close()
    
    return X_train, X_train_stylometric, X_train_readability, X_train_lexical, X_train_liwc, X_train_sentiments, y_train
    
def load_valid_features(dataset_name, eda): 
    if eda == 'y':
        dataset_name = dataset_name + '_eda'
    
    pickle_in = open(dataset_name + '_features/valid/X_valid', 'rb')
    X_valid = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(dataset_name + '_features/valid/X_valid_stylometric', 'rb')
    X_valid_stylometric = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(dataset_name + '_features/valid/X_valid_readability', 'rb')
    X_valid_readability = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(dataset_name + '_features/valid/X_valid_lexical', 'rb')
    X_valid_lexical = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(dataset_name + '_features/valid/X_valid_liwc', 'rb')
    X_valid_liwc = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(dataset_name + '_features/valid/X_valid_sentiments', 'rb')
    X_valid_sentiments = pickle.load(pickle_in)
    pickle_in.close()
      
    pickle_in = open(dataset_name + '_features/valid/y_valid', 'rb')
    y_valid = pickle.load(pickle_in)
    pickle_in.close()
    
    return X_valid, X_valid_stylometric, X_valid_readability, X_valid_lexical, X_valid_liwc, X_valid_sentiments, y_valid 
    
def load_test_features(dataset_name, eda):
    if eda == 'y':
        dataset_name = dataset_name + '_eda'
    
    pickle_in = open(dataset_name + '_features/test/X_test', 'rb')
    X_test = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(dataset_name + '_features/test/X_test_stylometric', 'rb')
    X_test_stylometric = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(dataset_name + '_features/test/X_test_readability', 'rb')
    X_test_readability = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(dataset_name + '_features/test/X_test_lexical', 'rb')
    X_test_lexical = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(dataset_name + '_features/test/X_test_liwc', 'rb')
    X_test_liwc = pickle.load(pickle_in)
    pickle_in.close()
    
    pickle_in = open(dataset_name + '_features/test/X_test_sentiments', 'rb')
    X_test_sentiments = pickle.load(pickle_in)
    pickle_in.close()
      
    pickle_in = open(dataset_name + '_features/test/y_test', 'rb')
    y_test = pickle.load(pickle_in)
    pickle_in.close()
    
    return X_test, X_test_stylometric, X_test_readability, X_test_lexical, X_test_liwc, X_test_sentiments, y_test     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    