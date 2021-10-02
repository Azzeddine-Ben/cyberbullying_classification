# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:14:36 2021

@author: IT Doctor
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import random as python_random
import warnings
import argparse
import os
from tensorflow import keras
import tensorflow_addons as tfa

from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer
from transformers import TFBertModel, TFRobertaModel, TFDistilBertModel
from transformers import logging as hf_logging

from sklearn.utils import class_weight
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
    my_parser.add_argument('cost_sensitive',
                           metavar='cost_sensitive',
                           type=str,
                           help='Cost sensitive method (cw: class weights / focal: focal loss)',
                           nargs='?'
                           )
    
    args = my_parser.parse_args()
    dataset_name = args.dataset_name
    eda = args.eda_dataset
    nlp_model = args.nlp_model
    clf_model = args.clf_model
    cs_method = args.cost_sensitive
    
    if nlp_model == 'bert':
        #### Loading tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=False)
        #### Loading TF BERT model
        hf_model = TFBertModel.from_pretrained('bert-large-uncased')
    elif nlp_model == 'roberta':
        tokenizer  = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=False)
        hf_model = TFRobertaModel.from_pretrained('roberta-large')     
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
    
    if eda == 'y':
        mchkp = keras.callbacks.ModelCheckpoint('./' + dataset_name + '_eda_' + nlp_model + '_' + clf_model + '.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only = True)
        model.compile(keras.optimizers.Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(
            [X_train_ids, X_train_masks, X_train], 
            y_train, 
            epochs=4, 
            batch_size=10, 
            validation_data=([X_valid_ids, X_valid_masks, X_valid], y_valid),
            callbacks=[mchkp]
            )
        path = dataset_name + '_eda_' + nlp_model + '_' + clf_model
        
    elif eda == 'n' and cs_method == 'cw':
        mchkp = keras.callbacks.ModelCheckpoint('./' + dataset_name + '_' + nlp_model + '_' + clf_model + '_cw.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only = True)
        model.compile(keras.optimizers.Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
        cw = class_weight.compute_class_weight('balanced',  [0, 1], y_train)
        cw_dict = {0: cw[0], 1: cw[1]}
        
        history = model.fit(
            [X_train_ids, X_train_masks, X_train],
            y_train, 
            epochs=4, 
            batch_size=10, 
            validation_data=([X_valid_ids, X_valid_masks, X_valid], y_valid),
            callbacks=[mchkp],
            class_weight=cw_dict
            )
        path = dataset_name + '_' + nlp_model + '_' + clf_model + '_cw'
        
    elif eda == 'n' and cs_method == 'focal':
        mchkp = keras.callbacks.ModelCheckpoint('./' + dataset_name + '_' + nlp_model + '_' + clf_model + '_focal_.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only = True)
        model.compile(keras.optimizers.Adam(lr=6e-6), loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.925, gamma=0.99), metrics=['accuracy'])
        history = model.fit(
            [X_train_ids, X_train_masks, X_train],
            y_train, 
            epochs=4, 
            batch_size=10, 
            validation_data=([X_valid_ids, X_valid_masks, X_valid], y_valid),
            callbacks=[mchkp]
            )        
        path = dataset_name + '_' + nlp_model + '_' + clf_model + '_focal'
    
    
# =============================================================================
#     Create a new directory with the model and dataset names
#         - Contains the learning curves figures
#         - The clssification report as a csv file 
#         - The confusion matrix figure
#         - The trained model
#     
# =============================================================================

# =============================================================================
#     if eda == 'y':
#         path = dataset_name + '_eda_' + nlp_model + '_' + clf_model
#     else:
#         path = dataset_name + '_' + nlp_model + '_' + clf_model
# =============================================================================
        
    os.mkdir(path)
    print_learning_curves(history, path)
    clf_report, confusion_matrix_fig = predict_and_visualize(model, [X_test_ids, X_test_masks, X_test], y_test)
    
    ### Saving the classification report as a CSV file
    clf_report_df = pd.DataFrame(clf_report).transpose()
    clf_report_df.to_csv(path + '/classification_report.csv') 
    ### Saving the confusion matrix figure
    confusion_matrix_fig.savefig(path + '/confusion_matrix')
    ### Saving the model
    model.save_weights(path + '/saved_weights')











