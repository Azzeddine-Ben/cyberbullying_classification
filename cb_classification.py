# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:14:36 2021

@author: IT Doctor
"""

import numpy as np
import tensorflow as tf
import random as python_random
import warnings
import argparse

from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer
from transformers import TFBertModel, TFRobertaModel, TFDistilBertModel
from transformers import logging as hf_logging

import load_data_module
import clf_models
from results_prediction_module import print_learning_curves, predict_and_visualize

#### Inputs encoding function
def hf_model_encode(data, maximum_length, tokenizer) :
  input_ids = []
  attention_masks = []
  for i in range(len(data)):
      encoded = tokenizer.encode_plus(      
        data[i],
        add_special_tokens=True,
        max_length=maximum_length,
        pad_to_max_length=True,
        truncation=True,
        return_attention_mask=True,       
      )     
      input_ids.append(encoded['input_ids'])
      attention_masks.append(encoded['attention_mask'])
  return np.array(input_ids),np.array(attention_masks)

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(123)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(1234)

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
hf_logging.set_verbosity_error()

if __name__ == '__main__':
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
    my_parser.add_argument('nlp_model',
                           metavar='nlp_model',
                           type=str,
                           help='Name of the NLP model to load')
    my_parser.add_argument('clf_model',
                           metavar='clf_model',
                           type=str,
                           help='Name of the classification model (MCNN/BLSTM-MCNN)')
    
    args = my_parser.parse_args()
    dataset_name = args.dataset_name
    eda = args.eda_dataset
    nlp_model = args.nlp_model
    clf_model = args.clf_model
    
    if nlp_model == 'bert':
        #### Loading tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=False)
        #### Loading TF BERT model
        hf_model = TFBertModel.from_pretrained('bert-large-uncased')
    elif nlp_model == 'roberta':
        tokenizer  = RobertaTokenizer.from_pretrained('roberta-large-uncased', do_lower_case=False)
        hf_model = TFRobertaModel.from_pretrained('roberta-large-uncased')     
    elif nlp_model == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=False)
        hf_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')   
        
    X_train, X_train_stylometric, X_train_readability, X_train_lexical, X_train_liwc, X_train_sentiments, y_train = load_data_module.load_train_features(dataset_name, eda)
    X_valid, X_valid_stylometric, X_valid_readability, X_valid_lexical, X_valid_liwc, X_valid_sentiments, y_valid = load_data_module.load_valid_features(dataset_name, eda)
    X_test, X_test_stylometric, X_test_readability, X_test_lexical, X_test_liwc, X_test_sentiments, y_test = load_data_module.load_test_features(dataset_name, eda)   
    
    MAX_LEN = 40    
    ##### Preparing train and test data
    X_train_ids, X_train_masks = hf_model_encode(X_train, MAX_LEN, tokenizer)
    X_valid_ids, X_valid_masks = hf_model_encode(X_valid, MAX_LEN, tokenizer)
    X_test_ids, X_test_masks   = hf_model_encode(X_test, MAX_LEN, tokenizer)
    
    if clf_model == 'mcnn':
        model = clf_models.mcnn_model(hf_model)
    elif clf_model =='blstm_mcnn':
        model = clf_models.blstm_mcnn_model(hf_model)

    model.summary()
    
    ## Training 
    history = model.fit(
        [X_train_ids, X_train_masks, X_train, X_train_stylometric, X_train_lexical, X_train_readability, 
         X_train_liwc, X_train_sentiments], y_train, 
        epochs=4, 
        batch_size=10, 
        validation_data=([X_valid_ids, X_valid_masks, X_valid, X_valid_stylometric, X_valid_lexical, X_valid_readability,
                          X_valid_liwc, X_valid_sentiments], y_valid)
        )
    
    print_learning_curves(history)
    predict_and_visualize(model, [X_test_ids, X_test_masks, X_test, X_test_stylometric, X_test_lexical, X_test_readability,
                                  X_test_liwc, X_test_sentiments], y_test)












