# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 14:53:41 2022

@author: IT Doctor_cb_classification_features
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
from sklearn.preprocessing import MinMaxScaler
import load_data_module, clf_models, iffl_loss
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
  return np.array(input_ids), np.array(attention_masks)

#### Features Scaling Function
def FeaturesScaling(features):
  minmax_scaler = MinMaxScaler()
  minmax_scaler.fit(features)
  scaled_features = minmax_scaler.transform(features)
  return minmax_scaler, scaled_features.astype(dtype='float32')  


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
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased', do_lower_case=False)
        hf_model = TFDistilBertModel.from_pretrained('distilbert-base-cased')   
        
    X_train, X_train_stylometric, X_train_readability, X_train_lexical, X_train_liwc, X_train_sentiments, y_train = load_data_module.load_train_features(dataset_name, eda)
    X_valid, X_valid_stylometric, X_valid_readability, X_valid_lexical, X_valid_liwc, X_valid_sentiments, y_valid = load_data_module.load_valid_features(dataset_name, eda)
    X_test, X_test_stylometric, X_test_readability, X_test_lexical, X_test_liwc, X_test_sentiments, y_test = load_data_module.load_test_features(dataset_name, eda)   
    
    #======================================================================
    #================== Features Scaling ==================================
    #======================================================================

    stylometric_scaler, X_train_stylometric =  FeaturesScaling(X_train_stylometric)
    X_test_stylometric = stylometric_scaler.transform(X_test_stylometric)
    X_valid_stylometric = stylometric_scaler.transform(X_valid_stylometric)

    readability_scaler, X_train_readability =  FeaturesScaling(X_train_readability)
    X_test_readability = readability_scaler.transform(X_test_readability)
    X_valid_readability = readability_scaler.transform(X_valid_readability)

    lexical_scaler, X_train_lexical =  FeaturesScaling(X_train_lexical)
    X_test_lexical = lexical_scaler.transform(X_test_lexical)
    X_valid_lexical = lexical_scaler.transform(X_valid_lexical)

    # liwc_scaler, X_train_liwc =  FeaturesScaling(X_train_liwc)
    # X_test_liwc = liwc_scaler.transform(X_test_liwc)
    # X_valid_liwc = liwc_scaler.transform(X_valid_liwc)

    sentiments_scaler, X_train_sentiments =  FeaturesScaling(X_train_sentiments)
    X_test_sentiments = sentiments_scaler.transform(X_test_sentiments)
    X_valid_sentiments = sentiments_scaler.transform(X_valid_sentiments)

    #=======================================================================

    MAX_LEN = 40    
    ##### Preparing train and test data
    X_train_ids, X_train_masks = hf_model_encode(X_train, MAX_LEN, tokenizer)
    X_valid_ids, X_valid_masks = hf_model_encode(X_valid, MAX_LEN, tokenizer)
    X_test_ids, X_test_masks   = hf_model_encode(X_test, MAX_LEN, tokenizer)
    
    #### Added clf models along with input features
    if clf_model == 'mcnn_with_features':
        model = clf_models.mcnn_model_with_features(hf_model)
    elif clf_model == 'blstm_mcnn_with_features':
        model = clf_models.blstm_mcnn_model_with_features(hf_model)
    elif clf_model == 'mcnn_blstm_with_features':
        model = clf_models.mcnn_blstm_model_with_features(hf_model)

    model.summary()
    
    if eda == 'y':
        chkp_path = './' + dataset_name + '_eda_' + nlp_model + '_' + clf_model + '.h5'
        mchkp = keras.callbacks.ModelCheckpoint(chkp_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only = True)
        model.compile(keras.optimizers.Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(
            [X_train_ids, X_train_masks, X_train, X_train_lexical, X_train_readability, X_train_sentiments, X_train_stylometric], 
            y_train, 
            epochs=4, 
            batch_size=10, 
            validation_data=([X_valid_ids, X_valid_masks, X_valid, X_valid_lexical, X_valid_readability, X_valid_sentiments, X_valid_stylometric], y_valid),
            callbacks=[mchkp]
            )
        path = dataset_name + '_eda_' + nlp_model + '_' + clf_model
        
    elif eda == 'n' and cs_method == 'cw':
        chkp_path = './' + dataset_name + '_' + nlp_model + '_' + clf_model + '_cw.h5'
        mchkp = keras.callbacks.ModelCheckpoint(chkp_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only = True)
        model.compile(keras.optimizers.Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])
        cw = class_weight.compute_class_weight(class_weight = 'balanced',  classes = [0, 1], y = y_train)
        cw_dict = {0: cw[0], 1: cw[1]}
        
        history = model.fit(
            [X_train_ids, X_train_masks, X_train, X_train_lexical, X_train_readability, X_train_sentiments, X_train_stylometric],
            y_train, 
            epochs=4, 
            batch_size=10, 
            validation_data=([X_valid_ids, X_valid_masks, X_valid, X_valid_lexical, X_valid_readability, X_valid_sentiments, X_valid_stylometric], y_valid),
            callbacks=[mchkp],
            class_weight=cw_dict
            )
        path = dataset_name + '_' + nlp_model + '_' + clf_model + '_cw'
        
    elif eda == 'n' and cs_method == 'focal':
        chkp_path = './' + dataset_name + '_' + nlp_model + '_' + clf_model + '_focal_.h5'
        mchkp = keras.callbacks.ModelCheckpoint(chkp_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only = True)
        model.compile(keras.optimizers.Adam(lr=6e-6), loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.925, gamma=0.99), metrics=['accuracy'])
        history = model.fit(
            [X_train_ids, X_train_masks, X_train, X_train_lexical, X_train_readability, X_train_sentiments, X_train_stylometric],
            y_train, 
            epochs=4, 
            batch_size=10, 
            validation_data=([X_valid_ids, X_valid_masks, X_valid, X_valid_lexical, X_valid_readability, X_valid_sentiments, X_valid_stylometric], y_valid),
            callbacks=[mchkp]
            )        
        path = dataset_name + '_' + nlp_model + '_' + clf_model + '_focal'
        
    elif eda == 'n' and cs_method == 'iffl':
        chkp_path = './' + dataset_name + '_' + nlp_model + '_' + clf_model + '_iffl.h5'
        cw = class_weight.compute_class_weight('balanced',  [0, 1], y_train)
        cw_dict = {0: cw[0], 1: cw[1]}
        mchkp = keras.callbacks.ModelCheckpoint(chkp_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only = True)
        model.compile(keras.optimizers.Adam(lr=6e-6), loss= iffl_loss.if_focal_loss(alpha=0.65, gamma=0.55, class_weights=cw_dict), metrics=['accuracy'])
         
        history = model.fit(
            [X_train_ids, X_train_masks, X_train, X_train_lexical, X_train_readability, X_train_sentiments, X_train_stylometric],
            y_train, 
            epochs=4, 
            batch_size=10, 
            validation_data=([X_valid_ids, X_valid_masks, X_valid, X_valid_lexical, X_valid_readability, X_valid_sentiments, X_valid_stylometric], y_valid),
            callbacks=[mchkp]
            )        
        path = dataset_name + '_' + nlp_model + '_' + clf_model + '_iffl'        
    
    
# =============================================================================
#     Create a new directory with the model and dataset names
#         - Contains the learning curves figures
#         - The clssification report as a csv file 
#         - The confusion matrix figure
#         - The trained model
#     
# =============================================================================
     
    os.mkdir(path)
    print_learning_curves(history, path)
    model.load_weights(chkp_path)
    clf_report, confusion_matrix_fig = predict_and_visualize(model, [X_test_ids, X_test_masks, X_test, X_test_lexical, X_test_readability, X_test_sentiments, X_test_stylometric], y_test)
    
    ### Saving the classification report as a CSV file
    clf_report_df = pd.DataFrame(clf_report).transpose()
    clf_report_df.to_csv(path + '/classification_report.csv') 
    ### Saving the confusion matrix figure
    confusion_matrix_fig.savefig(path + '/confusion_matrix')
    ### Saving the model
    model.save_weights(path + '/saved_weights')