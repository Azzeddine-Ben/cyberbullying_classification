# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:54:47 2021

@author: IT Doctor
"""

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import argparse
import load_data_module
import pickle
import os
import tensorflow_text

### Generate Arabic USE embeddings
def generate_embeddings(dataset):
    ## load hub module
    embed_type = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    
    dataset_embeddings = []
    for text in dataset:
        if (text is not None) and (text != '') and (text != ' '):
            emb = embed_type(tf.constant([text], dtype=tf.string)).numpy()[0]
            dataset_embeddings.append(emb)
        else:
            dataset_embeddings.append(np.zeros(512))
    return dataset_embeddings

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    
    # Add the arguments
    my_parser.add_argument('dataset_name',
                           metavar='dataset_name',
                           type=str,
                           help='Name of the dataset to extract features')

    
    args = my_parser.parse_args()
    dataset_name = args.dataset_name

    #### Load data
    X_train, X_test = load_data_module.load_data(dataset_name)
    
    ##### Generate USE embeddings
    X_train_use, X_test_use = generate_embeddings(X_train), generate_embeddings(X_test)
    
    ##### Saving data
    directory = dataset_name + '_use_embeddings'
    if (os.path.exists('')) == False:  
     os.mkdir(directory)
     pickle.dump( X_train_use, open(directory + "/X_train_use", "wb" ) )
     pickle.dump( X_test_use, open(directory + "/X_test_use", "wb" ) )  
     
     
     